//! Conversation state: the message surface plus the per-session JSONL files
//! it is derived from. The system prompt lives in config, never in the log;
//! compaction rotates to a fresh session file seeded with the checkpoint,
//! so files stay bounded and old generations remain on disk as archives.
//! `migrate` converts the legacy single-file layout on first load.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

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

/// Pure fold of log ops into a surface. Legacy-format machinery: old
/// single-file logs carried the system prompt at index 0 and `Compact` ops
/// shadowing a span after it; kept only so `migrate` can replay them.
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

/// Number of leading messages that can be shadowed while keeping a tail
/// worth at least `retain_tokens`. Never splits an assistant-tool_call /
/// tool-result pair. `None` when there is nothing meaningful to compact.
pub fn shadow_count(msgs: &[Message], retain_tokens: u64) -> Option<usize> {
    let mut boundary = msgs.len();
    let mut acc = 0u64;
    while boundary > 0 {
        let t = message_tokens(&msgs[boundary - 1]);
        if acc > 0 && acc + t > retain_tokens {
            break;
        }
        acc += t;
        boundary -= 1;
    }
    while boundary > 0 && msgs[boundary].role == Role::Tool {
        boundary -= 1;
    }
    (boundary > 0).then_some(boundary)
}

/// One session file per compaction generation: `state/sessions/<ts>.jsonl`
/// holds plain `Append` ops of the post-compaction surface, `state/current`
/// names the active file. The system prompt is never stored — config
/// supplies it at request time, so a restart applies a new prompt at once.
/// Compaction rotates to a fresh file seeded with `[checkpoint, tail]`,
/// which also keeps every file's size bounded.
pub struct Session {
    surface: Vec<Message>,
    state_dir: PathBuf,
    file: Option<(PathBuf, std::fs::File)>,
}

impl Session {
    pub fn load(state_dir: &Path) -> Result<Self> {
        let pointer = state_dir.join("current");
        let (surface, file) = match std::fs::read_to_string(&pointer) {
            Ok(name) => {
                let path = state_dir.join("sessions").join(name.trim());
                let file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .with_context(|| format!("open session file {}", path.display()))?;
                let ops = std::io::BufReader::new(std::fs::File::open(&path)?)
                    .lines()
                    .map(|l| serde_json::from_str::<Op>(&l?).context("decode session line"))
                    .collect::<Result<Vec<Op>>>()?;
                (replay(ops), Some((path, file)))
            }
            Err(_) if state_dir.join("session.jsonl").exists() => {
                let s = Self::migrate(state_dir)?;
                return Ok(s);
            }
            Err(_) => (Vec::new(), None),
        };
        let mut surface = surface;
        repair_dangling_calls(&mut surface);
        Ok(Self { surface, state_dir: state_dir.to_path_buf(), file })
    }

    /// Convert a legacy single-file log (system prompt stored at the head,
    /// `Compact` ops shadowing a span after it) into the per-session layout.
    /// The stored prompt is dropped; the logical surface is re-seeded into a
    /// fresh file and the legacy one is renamed aside.
    fn migrate(state_dir: &Path) -> Result<Self> {
        let legacy = state_dir.join("session.jsonl");
        let ops = std::io::BufReader::new(std::fs::File::open(&legacy)?)
            .lines()
            .map(|l| serde_json::from_str::<Op>(&l?).context("decode legacy session line"))
            .collect::<Result<Vec<Op>>>()?;
        let mut surface = replay(ops);
        while surface.first().is_some_and(|m| m.role == Role::System) {
            surface.remove(0);
        }
        repair_dangling_calls(&mut surface);
        let mut session = Self { surface, state_dir: state_dir.to_path_buf(), file: None };
        let carried = session.surface.clone();
        session.rotate_into(&carried)?;
        std::fs::rename(&legacy, state_dir.join("session.jsonl.migrated"))
            .context("rename legacy session log")?;
        tracing::info!("migrated legacy session.jsonl into per-session layout");
        Ok(session)
    }

    /// Close the current file and make `messages` the content of a fresh one.
    fn rotate_into(&mut self, messages: &[Message]) -> Result<()> {
        self.file = None;
        let dir = self.state_dir.join("sessions");
        std::fs::create_dir_all(&dir)?;
        // Millisecond stamps collide when rotations land in the same instant
        // (e.g. a compaction right after the first append) — probe for a free
        // name instead of truncating the existing generation.
        let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%3f").to_string();
        let name = (0u32..)
            .map(|n| if n == 0 { format!("{stamp}.jsonl") } else { format!("{stamp}.{n}.jsonl") })
            .find(|name| !dir.join(name).exists())
            .expect("unbounded probe over free names");
        let path = dir.join(&name);
        let mut file = std::fs::File::create(&path)
            .with_context(|| format!("create session file {}", path.display()))?;
        for m in messages {
            Self::write(&mut file, &Op::Append { message: m.clone() })?;
        }
        std::fs::write(self.state_dir.join("current"), &name)?;
        self.file = Some((path, file));
        Ok(())
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
        if self.file.is_none() {
            self.rotate_into(&[])?;
        }
        let (_, log) = self.file.as_mut().expect("rotated above");
        Self::write(log, &Op::Append { message: message.clone() })?;
        self.surface.push(message);
        Ok(())
    }

    /// Replace the shadowed prefix with the checkpoint by rotating: the new
    /// session file starts as `[summary, kept tail…]`, the old file remains
    /// on disk as an archive generation.
    pub fn compact(&mut self, shadowed: usize, summary: String) -> Result<()> {
        let mut carried = Vec::with_capacity(1 + self.surface.len() - shadowed);
        carried.push(summary_message(&summary));
        carried.extend_from_slice(&self.surface[shadowed..]);
        self.rotate_into(&carried)?;
        self.surface = carried;
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
            Op::Append { message: Message::user("0123456789".repeat(40)) }, // 400 chars
            Op::Append { message: Message::user("x") },
        ];
        let session = Session {
            surface: replay(ops),
            state_dir: std::env::temp_dir(),
            file: None,
        };
        let (head, tail) = (session.tail_tokens(1), session.tail_tokens(0));
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
        let msgs: Vec<_> = (0..10)
            .map(|i| Message::user(format!("u{i}{}", "x".repeat(400))))
            .collect();
        let shadowed = shadow_count(&msgs, 200).unwrap();
        assert!(shadowed > 0 && shadowed < 10);

        // A boundary landing between a call and its result moves back.
        let mut msgs = Vec::new();
        for i in 0..3 {
            msgs.extend(tool_pair(i));
        }
        msgs.push(Message::user("tail"));
        let shadowed = shadow_count(&msgs, 1).unwrap();
        assert_ne!(msgs[shadowed].role, Role::Tool);
    }

    #[test]
    fn shadow_none_when_nothing_to_shadow() {
        assert_eq!(shadow_count(&[Message::user("only")], 1), None);
        assert_eq!(shadow_count(&[], 1000), None);
    }

    #[test]
    fn compaction_rotates_to_a_fresh_session_file() {
        let dir = std::env::temp_dir().join(format!("rpbot-rot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut s = Session::load(&dir).unwrap();
        s.append(Message::user("hi")).unwrap();
        s.append(Message::user("hello")).unwrap();
        s.compact(2, "user said hi".into()).unwrap();
        // The surface is now just the checkpoint; the old file is an archive.
        assert_eq!(s.surface().len(), 1);
        assert!(s.surface()[0].text.contains("<compacted-summary>"));
        assert_eq!(std::fs::read_dir(dir.join("sessions")).unwrap().count(), 2);
        s.append(Message::user("more")).unwrap();
        drop(s);
        // A fresh load follows the pointer to the newest generation.
        let s = Session::load(&dir).unwrap();
        assert_eq!(s.surface().len(), 2);
        assert!(s.surface()[0].text.contains("<compacted-summary>"));
        assert_eq!(s.surface()[1].text, "more");
        assert!(!dir.join("session.jsonl").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn legacy_log_migrates_and_drops_stored_prompt() {
        let dir = std::env::temp_dir().join(format!("rpbot-mig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let line = |op: &Op| serde_json::to_string(op).unwrap();
        let ops = [
            Op::Append { message: Message::system("old prompt") },
            Op::Append { message: Message::user("hi") },
            Op::Append { message: Message::user("hello") },
            Op::Compact { shadowed: 2, summary: "greeted".into() },
            Op::Append { message: Message::user("more") },
        ];
        std::fs::write(
            dir.join("session.jsonl"),
            ops.iter().map(&line).collect::<Vec<_>>().join("\n") + "\n",
        )
        .unwrap();

        let s = Session::load(&dir).unwrap();
        // No stored prompt, checkpoint preserved, tail kept.
        assert_eq!(s.surface().len(), 2);
        assert!(s.surface().iter().all(|m| m.role != Role::System));
        assert!(s.surface()[0].text.contains("<compacted-summary>"));
        assert_eq!(s.surface()[1].text, "more");
        // The legacy file is renamed aside; the live state is in the new layout.
        assert!(dir.join("session.jsonl.migrated").exists());
        assert!(dir.join("current").exists());
        assert_eq!(std::fs::read_dir(dir.join("sessions")).unwrap().count(), 1);
        // Migration is one-shot: the next load reads the new layout.
        drop(s);
        let s = Session::load(&dir).unwrap();
        assert_eq!(s.surface().len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
