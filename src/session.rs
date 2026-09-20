//! Conversation state: message surface plus the append-only JSONL log it is
//! derived from. Compaction never rewrites history — it appends a `Compact`
//! op that shadows a span of the surface.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

/// Estimated cost of one image block, in model tokens. Rough by design:
/// the estimate only drives the compaction threshold.
const IMAGE_TOKENS: u64 = 1500;
const CHARS_PER_TOKEN: u64 = 4;
const MESSAGE_OVERHEAD_TOKENS: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON as produced by the model; parsed lazily at dispatch time.
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub text: String,
    /// Chain-of-thought of an assistant message (preserved thinking). Kept
    /// verbatim and replayed in order; empty when preservation is off.
    #[serde(default)]
    pub reasoning: String,
    /// Data URLs attached to a user message (vision input).
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, text: text.into(), reasoning: String::new(), images: vec![], tool_calls: vec![], tool_call_id: None }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, text: text.into(), reasoning: String::new(), images: vec![], tool_calls: vec![], tool_call_id: None }
    }

    pub fn user_with_images(text: impl Into<String>, images: Vec<String>) -> Self {
        Self { role: Role::User, text: text.into(), reasoning: String::new(), images, tool_calls: vec![], tool_call_id: None }
    }

    pub fn assistant_with_calls(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self { role: Role::Assistant, text: text.into(), reasoning: String::new(), images: vec![], tool_calls, tool_call_id: None }
    }

    pub fn tool_result(call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self { role: Role::Tool, text: text.into(), reasoning: String::new(), images: vec![], tool_calls: vec![], tool_call_id: Some(call_id.into()) }
    }

    /// Attach chain-of-thought (builder style); only used when preserve
    /// thinking is on, so the session log stays lean otherwise.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = reasoning.into();
        self
    }
}

/// Durable operations; the surface is their fold.
#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Op {
    Append { message: Message },
    /// Drop `shadowed` entries right after the system head and put the
    /// summary user message in their place.
    Compact { shadowed: usize, summary: String },
}

pub const SUMMARY_PREAMBLE: &str = "This is an automatically generated checkpoint condensing an earlier span of the conversation. Treat it as established background and build on it without restating it. Continue directly from the messages that follow, without acknowledging this checkpoint.";

fn summary_message(summary: &str) -> Message {
    Message::user(format!("{SUMMARY_PREAMBLE}\n\n<compacted-summary>\n{summary}\n</compacted-summary>"))
}

/// Pure fold of log ops into a surface. `compact` shadows a span but the log
/// keeps everything, so replaying from scratch always reproduces the state.
fn replay(ops: impl IntoIterator<Item = Op>) -> Vec<Message> {
    let mut surface: Vec<Message> = Vec::new();
    for op in ops {
        apply_op(&mut surface, op);
    }
    surface
}

fn apply_op(surface: &mut Vec<Message>, op: Op) {
    match op {
        Op::Append { message } => surface.push(message),
        Op::Compact { shadowed, summary } => {
            if surface.len() > 1 {
                let end = (1 + shadowed).min(surface.len());
                surface.drain(1..end);
            }
            let summary = summary_message(&summary);
            if surface.len() > 1 {
                surface.insert(1, summary);
            } else {
                surface.push(summary);
            }
        }
    }
}

/// A crashed turn can leave assistant tool calls without results; providers
/// reject that, so synthesize error results on load.
fn repair_dangling_calls(surface: &mut Vec<Message>) {
    let mut i = 0;
    while i < surface.len() {
        if surface[i].role != Role::Assistant {
            i += 1;
            continue;
        }
        let calls = surface[i].tool_calls.clone();
        let mut j = i + 1;
        for call in calls {
            let answered = surface.get(j).is_some_and(|m| {
                m.role == Role::Tool && m.tool_call_id.as_deref() == Some(call.id.as_str())
            });
            if !answered {
                surface.insert(
                    j,
                    Message::tool_result(call.id.clone(), "Error: tool call was interrupted before a result was recorded."),
                );
            }
            j += 1;
        }
        i = j;
    }
}

pub fn message_tokens(m: &Message) -> u64 {
    (m.text.chars().count() + m.reasoning.chars().count()) as u64 / CHARS_PER_TOKEN
        + MESSAGE_OVERHEAD_TOKENS
        + m.images.len() as u64 * IMAGE_TOKENS
}

pub fn estimate_tokens(msgs: &[Message]) -> u64 {
    msgs.iter().map(message_tokens).sum()
}

/// Number of entries after the system head that can be shadowed while
/// keeping a tail worth at least `retain_tokens`. Never splits an
/// assistant-tool_call / tool-result pair. `None` when there is nothing
/// meaningful to compact.
pub fn shadow_count(msgs: &[Message], retain_tokens: u64) -> Option<usize> {
    let mut boundary = msgs.len();
    let mut acc = 0u64;
    while boundary > 1 {
        let t = message_tokens(&msgs[boundary - 1]);
        if acc > 0 && acc + t > retain_tokens {
            break;
        }
        acc += t;
        boundary -= 1;
    }
    while boundary > 1 && msgs[boundary].role == Role::Tool {
        boundary -= 1;
    }
    (boundary > 1).then_some(boundary - 1)
}

pub struct Session {
    surface: Vec<Message>,
    log: std::fs::File,
}

impl Session {
    pub fn load(log_path: &Path) -> Result<Self> {
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("open session log {}", log_path.display()))?;
        let surface = if log_path.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            Vec::new()
        } else {
            let file = std::fs::File::open(log_path)
                .with_context(|| format!("read session log {}", log_path.display()))?;
            let ops = std::io::BufReader::new(file)
                .lines()
                .map(|l| serde_json::from_str::<Op>(&l?).context("decode session log line"))
                .collect::<Result<Vec<Op>>>()?;
            let mut surface = replay(ops);
            repair_dangling_calls(&mut surface);
            surface
        };
        Ok(Self { surface, log })
    }

    pub fn surface(&self) -> &[Message] {
        &self.surface
    }

    /// Estimated tokens of `surface[from..]`. Combined with the provider's
    /// real `prompt_tokens` for the prefix before `from`, this tracks the
    /// context size: exact where it matters (the stable prefix), heuristic
    /// only for the short fresh tail.
    pub fn tail_tokens(&self, from: usize) -> u64 {
        self.surface[from.min(self.surface.len())..]
            .iter()
            .map(message_tokens)
            .sum()
    }

    pub fn append(&mut self, message: Message) -> Result<()> {
        Self::write(&mut self.log, &Op::Append { message: message.clone() })?;
        self.surface.push(message);
        Ok(())
    }

    pub fn compact(&mut self, shadowed: usize, summary: String) -> Result<()> {
        Self::write(&mut self.log, &Op::Compact { shadowed, summary: summary.clone() })?;
        apply_op(&mut self.surface, Op::Compact { shadowed, summary });
        Ok(())
    }

    fn write(log: &mut std::fs::File, op: &Op) -> Result<()> {
        let line = serde_json::to_string(op)?;
        log.write_all(line.as_bytes())
            .and_then(|_| log.write_all(b"\n"))
            .and_then(|_| log.flush())
            .context("append to session log")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_pair(i: usize) -> Vec<Message> {
        vec![
            Message::assistant_with_calls(
                "",
                vec![ToolCall { id: format!("c{i}"), name: "bash".into(), arguments: "{}".into() }],
            ),
            Message::tool_result(format!("c{i}"), "ok"),
        ]
    }

    #[test]
    fn replay_roundtrip() {
        let ops = vec![
            Op::Append { message: Message::system("sys") },
            Op::Append { message: Message::user("hi") },
            Op::Append { message: Message::assistant_with_calls("hello", vec![]) },
            Op::Compact { shadowed: 1, summary: "user said hi".into() },
            Op::Append { message: Message::user("more") },
        ];
        let surface = replay(ops);
        assert_eq!(surface.len(), 4);
        assert_eq!(surface[0].role, Role::System);
        assert!(surface[1].text.contains("<compacted-summary>"));
        assert_eq!(surface[2].text, "hello");
        assert_eq!(surface[3].text, "more");
    }

    #[test]
    fn tail_tokens_counts_only_the_tail() {
        let ops = vec![
            Op::Append { message: Message::system("sys") },
            Op::Append { message: Message::user("0123456789".repeat(40)) }, // 400 chars
            Op::Append { message: Message::user("x") },
        ];
        let log = std::fs::File::create(std::env::temp_dir().join("rp-tail-tokens.jsonl")).unwrap();
        let session = Session { surface: replay(ops), log };
        let (head, tail) = (session.tail_tokens(2), session.tail_tokens(0));
        assert!(head < tail);
        assert_eq!(head, message_tokens(&Message::user("x")));
        assert_eq!(session.tail_tokens(99), 0);
    }

    #[test]
    fn repair_inserts_missing_tool_results() {
        let mut surface = vec![
            Message::system("s"),
            Message::assistant_with_calls(
                "",
                vec![ToolCall { id: "c1".into(), name: "bash".into(), arguments: "{}".into() }],
            ),
            Message::user("next"),
        ];
        repair_dangling_calls(&mut surface);
        assert_eq!(surface.len(), 4);
        assert_eq!(surface[2].role, Role::Tool);
        assert_eq!(surface[2].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn repair_keeps_answered_pairs() {
        let mut surface = vec![Message::system("s")];
        surface.extend(tool_pair(1));
        repair_dangling_calls(&mut surface);
        assert_eq!(surface.len(), 3);
    }

    #[test]
    fn shadow_respects_retain_and_pairs() {
        let mut msgs = vec![Message::system("s")];
        for i in 0..10 {
            msgs.push(Message::user(format!("u{i}{}", "x".repeat(400))));
        }
        let shadowed = shadow_count(&msgs, 200).unwrap();
        assert!(shadowed > 0 && shadowed < 10);

        // A boundary landing between a call and its result moves back.
        let mut msgs = vec![Message::system("s")];
        for i in 0..3 {
            msgs.extend(tool_pair(i));
        }
        msgs.push(Message::user("tail"));
        let shadowed = shadow_count(&msgs, 1).unwrap();
        let kept = &msgs[1 + shadowed];
        assert_ne!(kept.role, Role::Tool);
    }

    #[test]
    fn shadow_nothing_when_only_head() {
        let msgs = vec![Message::system("s"), Message::user("only")];
        assert_eq!(shadow_count(&msgs, 1), None);
    }

    #[test]
    fn session_log_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rpbot-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.jsonl");
        let mut s = Session::load(&path).unwrap();
        s.append(Message::system("sys")).unwrap();
        s.append(Message::user("hi")).unwrap();
        s.compact(1, "user said hi".into()).unwrap();
        // The in-memory surface must equal what a fresh replay produces.
        assert_eq!(s.surface().len(), 2);
        assert_eq!(s.surface()[0].role, Role::System);
        assert!(s.surface()[1].text.contains("compacted-summary"));
        s.append(Message::user("more")).unwrap();
        drop(s);
        let s = Session::load(&path).unwrap();
        assert_eq!(s.surface().len(), 3);
        assert!(s.surface()[1].text.contains("compacted-summary"));
        assert_eq!(s.surface()[2].text, "more");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
