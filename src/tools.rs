//! Agent tools: filesystem/terminal plus Telegram output. Handlers never
//! fail — every outcome becomes model-readable text, so a tool error can
//! never take down the turn.

use crate::tg::Telegram;
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

const BASH_DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const BASH_MAX_TIMEOUT: Duration = Duration::from_secs(600);
const STREAM_TAIL_BYTES: usize = 64 * 1024;
const READ_MAX_LINES: usize = 2000;
const READ_MAX_LINE_CHARS: usize = 2000;
const READ_MAX_BYTES: usize = 50 * 1024;
const TELEGRAM_LIMIT: usize = 4000;

/// `tg` is `None` while compacting: the summarizer may keep notes on disk
/// but must not be able to message anyone.
pub struct Tools<'a> {
    pub tg: Option<&'a Telegram>,
    pub workspace: &'a Path,
}

pub fn schemas(telegram: bool) -> Value {
    let mut tools = vec![bash_schema(), read_schema(), write_schema(), edit_schema()];
    if telegram {
        tools.push(send_message_schema());
        // tools.push(send_typing_schema());
    }
    Value::Array(tools)
}

/// Strip a schema set down to the filesystem/terminal half.
pub fn file_only_schemas(all: &Value) -> Value {
    Value::Array(
        all.as_array()
            .expect("tools array")
            .iter()
            .filter(|t| {
                t["function"]["name"]
                    .as_str()
                    .is_some_and(|n| !n.starts_with("send_"))
            })
            .cloned()
            .collect(),
    )
}

fn bash_schema() -> Value {
    schema(
        "bash",
        "Run a shell command in the workspace directory (bash -c). Non-zero exit is not an error; output includes stdout, stderr and the exit code.",
        json!({
            "command": { "type": "string", "description": "Shell command to run." },
            "timeout_ms": { "type": "integer", "description": "Optional timeout, 120000 by default, 600000 max." }
        }),
        &["command"],
    )
}

fn read_schema() -> Value {
    schema(
        "read_file",
        "Read a text file with line numbers. Use offset to page through big files.",
        json!({
            "path": { "type": "string", "description": "File path, relative to the workspace directory unless absolute." },
            "offset": { "type": "integer", "description": "1-based first line to read, default 1." },
            "limit": { "type": "integer", "description": "Max lines to read, default 2000." }
        }),
        &["path"],
    )
}

fn write_schema() -> Value {
    schema(
        "write_file",
        "Create or overwrite a file (parent directories are created automatically).",
        json!({
            "path": { "type": "string", "description": "File path, relative to the workspace directory unless absolute." },
            "content": { "type": "string", "description": "Full file content." }
        }),
        &["path", "content"],
    )
}

fn edit_schema() -> Value {
    schema(
        "edit_file",
        "Replace an exact literal snippet in a file. old_string must occur exactly once unless replace_all is set. Read the file first.",
        json!({
            "path": { "type": "string", "description": "File path, relative to the workspace directory unless absolute." },
            "old_string": { "type": "string" },
            "new_string": { "type": "string" },
            "replace_all": { "type": "boolean", "description": "Replace every occurrence, default false." }
        }),
        &["path", "old_string", "new_string"],
    )
}

fn send_message_schema() -> Value {
    schema(
        "send_message",
        "Send a Telegram message. Long texts are split automatically. Plain text by default.",
        json!({
            "chat_id": { "type": "integer", "description": "Target chat id (a user id for DMs, a negative id for groups). Note that any message sent to the user's DM won't be visible for other users." },
            "thread_id": { "type": "integer", "description": "Forum topic (message_thread_id) to post into, if the chat has topics." },
            "text": { "type": "string" },
            "parse_mode": { "type": "string", "description": "Optional: HTML, MarkdownV2 or Markdown." },
            "reply_to_message_id": { "type": "integer", "description": "Optional message to reply to." }
        }),
        &["chat_id", "text"],
    )
}

#[allow(dead_code)] // disabled in schemas() for now, kept for easy re-enable
fn send_typing_schema() -> Value {
    schema(
        "send_typing",
        "Show a typing indicator in a chat for a few seconds. Call it before composing long replies.",
        json!({ "chat_id": { "type": "integer" } }),
        &["chat_id"],
    )
}

fn schema(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    })
}

/// Parse raw model-produced arguments; invalid JSON is reported back to the
/// model instead of crashing the turn.
pub fn parse_arguments(raw: &str) -> Result<Value> {
    serde_json::from_str(raw).map_err(|e| anyhow!("arguments were not valid JSON: {e}"))
}

impl<'a> Tools<'a> {
    pub async fn execute(&self, name: &str, args: Value) -> String {
        if !args.is_object() {
            return "Error: tool arguments must be a JSON object.".into();
        }
        match name {
            "bash" => self.bash(&args).await,
            "read_file" => self.read_file(&args),
            "write_file" => self.write_file(&args),
            "edit_file" => self.edit_file(&args),
            "send_message" => self.send_message(&args).await,
            "send_typing" => self.send_typing(&args).await,
            other => format!("Error: unknown tool `{other}`."),
        }
    }

    async fn bash(&self, args: &Value) -> String {
        let Some(command) = str_arg(args, "command") else {
            return err_arg("command");
        };
        let timeout = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(|ms| Duration::from_millis(ms.min(BASH_MAX_TIMEOUT.as_millis() as u64)))
            .unwrap_or(BASH_DEFAULT_TIMEOUT);

        let mut child = match tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(self.workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return format!("Error: failed to spawn bash: {e}"),
        };
        let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        use tokio::io::AsyncReadExt;
        let run = async {
            let (mut obuf, mut ebuf) = (Vec::new(), Vec::new());
            // Both streams are drained concurrently: waiting for stdout
            // alone would deadlock once the child fills the stderr pipe.
            let _ = tokio::join!(out.read_to_end(&mut obuf), err.read_to_end(&mut ebuf),);
            let status = child.wait().await;
            (obuf, ebuf, status)
        };
        match tokio::time::timeout(timeout, run).await {
            Ok((o, e, status)) => {
                let mut report = render_stream(&o, "");
                if !e.is_empty() {
                    report.push('\n');
                    report.push_str(&render_stream(&e, "[stderr]\n"));
                }
                match status {
                    Ok(s) if s.success() => report,
                    Ok(s) => {
                        report.push_str(&format!("\n[exit code: {}]", s.code().unwrap_or(-1)));
                        report
                    }
                    Err(e) => format!("{report}\nError: {e}"),
                }
            }
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await; // reap, don't leave a zombie
                format!("[timed out after {}ms]", timeout.as_millis())
            }
        }
    }

    fn read_file(&self, args: &Value) -> String {
        let Some(path) = str_arg(args, "path") else {
            return err_arg("path");
        };
        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(READ_MAX_LINES as u64)
            .max(1) as usize;
        match std::fs::read(self.resolve(path)) {
            Ok(bytes) => render_read(&bytes, offset, limit),
            Err(e) => format!("Error: {e}"),
        }
    }

    fn write_file(&self, args: &Value) -> String {
        let (Some(path), Some(content)) = (str_arg(args, "path"), str_arg(args, "content")) else {
            return err_arg("path, content");
        };
        let path = self.resolve(path);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return format!("Error: {e}");
        }
        match std::fs::write(&path, content) {
            Ok(()) => format!("Wrote {} bytes to {}.", content.len(), path.display()),
            Err(e) => format!("Error: {e}"),
        }
    }

    fn edit_file(&self, args: &Value) -> String {
        let (Some(path), Some(old), Some(new)) = (
            str_arg(args, "path"),
            str_arg(args, "old_string"),
            str_arg(args, "new_string"),
        ) else {
            return err_arg("path, old_string, new_string");
        };
        let replace_all = args
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if old == new {
            return "Error: old_string and new_string are identical.".into();
        }
        let path = self.resolve(path);
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return format!("Error: {e}"),
        };
        let count = content.matches(old).count();
        match count {
            0 => "Error: old_string not found in file.".into(),
            n if n > 1 && !replace_all => {
                format!("Error: old_string occurs {n} times; make it unique or set replace_all.")
            }
            _ => {
                let updated = if replace_all {
                    content.replace(old, new)
                } else {
                    content.replacen(old, new, 1)
                };
                match std::fs::write(&path, updated) {
                    Ok(()) => "Edited.".into(),
                    Err(e) => format!("Error: {e}"),
                }
            }
        }
    }

    async fn send_message(&self, args: &Value) -> String {
        let Some(tg) = self.tg else {
            return "Error: Telegram tools are unavailable right now.".into();
        };
        let (Some(text), Some(chat_id)) = (
            str_arg(args, "text"),
            args.get("chat_id").and_then(Value::as_i64),
        ) else {
            return err_arg("chat_id, text");
        };
        let thread_id = args.get("thread_id").and_then(Value::as_i64);
        let parse_mode = args.get("parse_mode").and_then(Value::as_str);
        let reply_to = args.get("reply_to_message_id").and_then(Value::as_i64);
        let chunks = split_text(text, TELEGRAM_LIMIT);
        for chunk in &chunks {
            if let Err(e) = tg
                .send_message(chat_id, thread_id, chunk, parse_mode, reply_to)
                .await
            {
                return format!("Error: {e}");
            }
        }
        let target = match thread_id {
            Some(t) => format!("chat {chat_id} topic {t}"),
            None => format!("chat {chat_id}"),
        };
        format!("Sent {} message(s) to {target}.", chunks.len())
    }

    async fn send_typing(&self, args: &Value) -> String {
        let Some(tg) = self.tg else {
            return "Error: Telegram tools are unavailable right now.".into();
        };
        let Some(chat_id) = args.get("chat_id").and_then(Value::as_i64) else {
            return err_arg("chat_id");
        };
        match tg.send_typing(chat_id).await {
            Ok(()) => "Typing indicator shown.".into(),
            Err(e) => format!("Error: {e}"),
        }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.workspace.join(p)
        }
    }
}

fn str_arg<'v>(args: &'v Value, key: &str) -> Option<&'v str> {
    args.get(key).and_then(Value::as_str)
}

fn err_arg(keys: &str) -> String {
    format!("Error: missing or invalid argument(s): {keys}.")
}

fn render_stream(bytes: &[u8], prefix: &str) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = if bytes.len() > STREAM_TAIL_BYTES {
        let cut = bytes.len() - STREAM_TAIL_BYTES;
        let tail = &text[cut..];
        format!("[... {cut} earlier bytes truncated ...]\n{tail}")
    } else {
        text.into_owned()
    };
    format!("{prefix}{text}")
}

fn render_read(bytes: &[u8], offset: usize, limit: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let total_lines = text.lines().count();
    let mut out = String::new();
    let mut shown = 0;
    let mut size = 0;
    let mut capped = false;
    for (i, line) in text.lines().enumerate().skip(offset - 1) {
        if shown >= limit {
            capped = true;
            break;
        }
        let rendered: String = if line.chars().count() > READ_MAX_LINE_CHARS {
            let mut head: String = line.chars().take(READ_MAX_LINE_CHARS).collect();
            head.push('…');
            head
        } else {
            line.to_string()
        };
        size += rendered.len() + 1;
        if size > READ_MAX_BYTES {
            capped = true;
            break;
        }
        out.push_str(&format!("{}: {rendered}\n", i + 1));
        shown += 1;
    }
    let footer = if capped {
        format!(
            "(Showing lines {}-{} of {total_lines}. Use offset={} to continue.)\n",
            offset,
            offset + shown - 1,
            offset + shown
        )
    } else if offset > 1 {
        format!("(End of file - total {total_lines} lines.)\n")
    } else {
        String::new()
    };
    format!("<content>\n{out}</content>\n{footer}")
}

pub fn split_text(text: &str, limit: usize) -> Vec<String> {
    if text.len() <= limit {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut rest = text;
    while rest.len() > limit {
        // Never slice mid-UTF-8: Telegram texts are frequently Cyrillic.
        let hard = floor_char_boundary(rest, limit);
        let low = floor_char_boundary(rest, hard.saturating_sub(500));
        let cut = rest[low..hard]
            .rfind('\n')
            .map(|i| low + i + 1)
            .unwrap_or(hard);
        let (head, tail) = rest.split_at(cut);
        chunks.push(head.trim_end().to_string());
        rest = tail;
    }
    if !rest.trim_end().is_empty() {
        chunks.push(rest.trim_end().to_string());
    }
    chunks
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(dir: &Path) -> Tools<'_> {
        Tools {
            tg: None,
            workspace: dir,
        }
    }

    #[test]
    fn schemas_and_file_only_filter() {
        let all = schemas(true);
        for name in [
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "send_message",
        ] {
            assert!(
                all.as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["function"]["name"] == name),
                "missing {name}"
            );
        }
        let files = file_only_schemas(&all);
        assert_eq!(files.as_array().unwrap().len(), 4);
        assert!(schemas(false).as_array().unwrap().len() == 4);
    }

    #[test]
    fn read_render_pages_and_caps_lines() {
        let bytes = b"one\ntwo\nthree".as_slice();
        let out = render_read(bytes, 1, 2);
        assert!(out.contains("1: one"));
        assert!(out.contains("2: two"));
        assert!(out.contains("Use offset=3 to continue"));
        let out = render_read(bytes, 3, 5);
        assert!(out.contains("3: three"));
        assert!(out.contains("End of file"));
    }

    #[test]
    fn edit_requires_unique_match() {
        let dir = std::env::temp_dir().join(format!("rpbot-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.txt");
        std::fs::write(&path, "aa bb aa").unwrap();
        let t = tools(&dir);
        let out = t.edit_file(&json!({ "path": "f.txt", "old_string": "aa", "new_string": "cc" }));
        assert!(out.contains("occurs 2 times"), "{out}");
        let out = t.edit_file(&json!({ "path": "f.txt", "old_string": "aa", "new_string": "cc", "replace_all": true }));
        assert_eq!(out, "Edited.");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "cc bb cc");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_then_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rpbot-tools2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = tools(&dir);
        let out = t.write_file(&json!({ "path": "notes/a.md", "content": "# Notes" }));
        assert!(out.starts_with("Wrote"), "{out}");
        let out = t.read_file(&json!({ "path": "notes/a.md" }));
        assert!(out.contains("1: # Notes"), "{out}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn telegram_tools_offline_report_error() {
        let dir = std::env::temp_dir();
        let t = tools(&dir);
        let out = t.edit_file(&json!({ "path": "x", "old_string": "a", "new_string": "b" }));
        assert!(out.starts_with("Error:"));
        let out = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(t.send_typing(&json!({ "chat_id": 1 })));
        assert!(out.contains("unavailable"), "{out}");
    }

    #[test]
    fn splitting_prefers_newlines() {
        let text = "a\n".repeat(2000);
        for chunk in split_text(&text, 4000) {
            assert!(chunk.len() <= 4000);
        }
        assert_eq!(split_text("short", 4000).len(), 1);
    }

    #[test]
    fn splitting_survives_multibyte_text() {
        // "ы" is 2 bytes in UTF-8; a naive byte-cut at `limit` would panic.
        let text = "ы".repeat(5000);
        let chunks = split_text(&text, 4000);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks.concat().len(), 10_000);
    }

    #[test]
    fn invalid_arguments_reported() {
        let e = parse_arguments("{not json").unwrap_err();
        assert!(e.to_string().contains("not valid JSON"));
    }
}
