//! OpenAI-compatible chat-completions client (Z.ai). Wire conversion is
//! hand-rolled to keep the request shape perfectly stable for prefix caching:
//! one system string head, a constant tools array, then messages.

use crate::session::{Message, Role, ToolCall};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

const MAX_ATTEMPTS: u32 = 3;

#[derive(Clone)]
pub struct Llm {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

pub struct ChatOptions<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    /// Z.ai `reasoning_effort` (takes effect with thinking enabled).
    /// `None` omits both it and the `thinking` switch: provider defaults.
    pub effort: Option<&'a str>,
    /// Preserved thinking: send `clear_thinking: false` so prior-turn
    /// `reasoning_content` stays in context instead of being stripped.
    pub preserve_thinking: bool,
}

#[derive(Debug, Default)]
pub struct Reply {
    pub text: String,
    /// Chain-of-thought of this response; kept only in preserve mode.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug)]
pub enum ChatError {
    /// Provider refused the request because the context does not fit.
    ContextOverflow(String),
    /// Non-retryable client error.
    Fatal(String),
    /// Network / 5xx / 429 — already retried, still failing.
    Transient(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::ContextOverflow(e) | ChatError::Fatal(e) | ChatError::Transient(e) => {
                write!(f, "{e}")
            }
        }
    }
}

impl Llm {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Self { http, base_url: base_url.into(), api_key: api_key.into() })
    }

    pub async fn chat(
        &self,
        msgs: &[Message],
        tools: &Value,
        opts: &ChatOptions<'_>,
    ) -> Result<Reply, ChatError> {
        let mut attempt = 1;
        loop {
            match self.chat_once(msgs, tools, opts).await {
                Err(ChatError::Transient(e)) if attempt < MAX_ATTEMPTS => {
                    tracing::warn!(attempt, error = %e, "transient LLM error, retrying");
                    tokio::time::sleep(Duration::from_secs(2u64 << (attempt - 1))).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    async fn chat_once(
        &self,
        msgs: &[Message],
        tools: &Value,
        opts: &ChatOptions<'_>,
    ) -> Result<Reply, ChatError> {
        let mut body = json!({
            "model": opts.model,
            "messages": to_wire(msgs),
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": opts.max_tokens,
        });
        if let Some(t) = opts.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(effort) = opts.effort {
            body["thinking"] = json!({ "type": "enabled" });
            body["reasoning_effort"] = json!(effort);
        }
        if opts.preserve_thinking {
            body["clear_thinking"] = json!(false);
        }

        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ChatError::Transient(format!("{e:#}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ChatError::Transient(format!("{e:#}")))?;
        if !status.is_success() {
            return Err(classify_error(status, &text));
        }

        let parsed: ApiResponse =
            serde_json::from_str(&text).map_err(|e| ChatError::Fatal(format!("malformed response: {e}")))?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ChatError::Fatal("empty choices".into()))?;
        let usage = parsed.usage;
        if let Some(u) = usage {
            // Hit rate on `cached_tokens` is how we verify prefix caching
            // actually works against the provider.
            tracing::info!(
                prompt = u.prompt_tokens,
                cached = u.cached_tokens,
                completion = u.completion_tokens,
                "llm usage"
            );
        }
        Ok(Reply {
            text: content_text(&choice.message.content),
            reasoning: choice.message.reasoning_content.unwrap_or_default(),
            tool_calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    name: c.function.name,
                    arguments: c.function.arguments.unwrap_or_else(|| "{}".into()),
                })
                .collect(),
            finish_reason: choice.finish_reason,
        })
    }
}

fn classify_error(status: reqwest::StatusCode, body: &str) -> ChatError {
    let lower = body.to_lowercase();
    if status.as_u16() == 400 && (lower.contains("context") || lower.contains("too long") || lower.contains("maximum")) {
        ChatError::ContextOverflow(format!("{}: {}", status, body))
    } else if status.is_server_error() || status.as_u16() == 429 {
        ChatError::Transient(format!("{}: {}", status, body))
    } else {
        ChatError::Fatal(format!("{}: {}", status, body))
    }
}

/// Content may arrive as a plain string or as a list of typed parts.
fn content_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.as_str().map(str::to_string).or_else(|| p.get("text")?.as_str().map(str::to_string)))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: RespMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RespMessage {
    #[serde(default)]
    content: Option<Value>,
    /// Chain-of-thought of the response (thinking models).
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RespToolCall>>,
}

#[derive(Deserialize)]
struct RespToolCall {
    id: String,
    function: RespFunction,
}

#[derive(Deserialize)]
struct RespFunction {
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

fn to_wire(msgs: &[Message]) -> Vec<Value> {
    msgs.iter().map(to_wire_msg).collect()
}

fn to_wire_msg(m: &Message) -> Value {
    match m.role {
        Role::System => json!({ "role": "system", "content": m.text }),
        Role::Tool => json!({
            "role": "tool",
            "content": m.text,
            "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
        }),
        Role::User => {
            if m.images.is_empty() {
                json!({ "role": "user", "content": m.text })
            } else {
                let mut parts = vec![json!({ "type": "text", "text": m.text })];
                parts.extend(m.images.iter().map(|url| {
                    json!({ "type": "image_url", "image_url": { "url": url } })
                }));
                json!({ "role": "user", "content": parts })
            }
        }
        Role::Assistant => {
            // Text-less turns send "" — never null: some gateways reject
            // null outright and the official samples replay "" verbatim.
            let mut v = json!({ "role": "assistant", "content": m.text });
            // CoT passback on every reasoning-carrying turn (required for
            // interleaved thinking with tools; must stay verbatim, in order).
            if !m.reasoning.is_empty() {
                v["reasoning_content"] = json!(m.reasoning);
            }
            if !m.tool_calls.is_empty() {
                v["tool_calls"] = json!(m.tool_calls.iter().map(|c| json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                }))
                .collect::<Vec<_>>());
            }
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Message;

    #[test]
    fn wire_shape_stable_prefix() {
        let msgs = vec![
            Message::system("sys"),
            Message::user_with_images("look", vec!["data:image/png;base64,AAA".into()]),
            Message::assistant_with_calls(
                "",
                vec![ToolCall { id: "c1".into(), name: "bash".into(), arguments: "{\"command\":\"ls\"}".into() }],
            )
            .with_reasoning("need to list files"),
            Message::tool_result("c1", "out"),
        ];
        let wire = to_wire(&msgs);
        assert_eq!(wire[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(wire[1]["content"][0]["type"], "text");
        assert_eq!(wire[1]["content"][1]["image_url"]["url"], "data:image/png;base64,AAA");
        assert_eq!(wire[2]["content"], json!(""));
        assert_eq!(wire[2]["reasoning_content"], "need to list files");
        assert_eq!(wire[2]["tool_calls"][0]["function"]["name"], "bash");
        assert_eq!(wire[3]["tool_call_id"], "c1");
    }

    #[test]
    fn wire_omits_empty_reasoning() {
        let wire = to_wire(&[Message::assistant_with_calls("hi", vec![])]);
        assert!(wire[0].get("reasoning_content").is_none());
    }

    #[test]
    fn content_parts_concat() {
        let v = json!(["text", {"type":"text","text":" tail"}]);
        assert_eq!(content_text(&Some(v)), "text tail");
    }
}
