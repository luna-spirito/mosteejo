//! Registry of Telegram channels the agent addresses by name — every DM,
//! group and forum topic is its own channel — plus append-only per-channel
//! event logs in `workspace/chat/`. The Bot API can neither enumerate chats
//! nor fetch a topic's name on demand, so names are learned from updates
//! that passed the ingress filter and frozen at first sight (a later rename
//! changes nothing, which also keeps log file names stable).

use crate::tg::{MessageReaction, TgMessage, render_reactions, timestamp};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Channel address: chat id plus the forum topic it belongs to, if any.
pub type Key = (i64, Option<i64>);

#[derive(Debug, Clone)]
struct Channel {
    /// Display name shown to the model; unique across channels.
    name: String,
    kind: String,
    /// Log file name inside `workspace/chat/`, derived from the name.
    log: String,
}

/// Owned by the agent (single writer); mutated in `collect_events` and read
/// by tools through a shared borrow.
pub struct Channels {
    path: PathBuf,
    chat_dir: PathBuf,
    channels: BTreeMap<Key, Channel>,
}

impl Channels {
    pub fn load(state_dir: &Path, workspace_dir: &Path) -> Self {
        let mut channels = Self {
            path: state_dir.join("channels.json"),
            chat_dir: workspace_dir.join("chat"),
            channels: std::fs::read_to_string(state_dir.join("channels.json"))
                .ok()
                .and_then(|s| serde_json::from_str::<Vec<Row>>(&s).ok())
                .map(rows_into_map)
                .unwrap_or_default(),
        };
        channels.migrate_legacy(state_dir);
        channels
    }

    fn save(&self) {
        let rows: Vec<Row> = self
            .channels
            .iter()
            .map(|((chat_id, thread), c)| Row {
                chat_id: *chat_id,
                thread: *thread,
                name: c.name.clone(),
                kind: c.kind.clone(),
                log: c.log.clone(),
            })
            .collect();
        if let Ok(line) = serde_json::to_string(&rows)
            && let Err(e) = std::fs::write(&self.path, line)
        {
            tracing::warn!(error = %e, "could not persist channel registry");
        }
    }

    fn insert(&mut self, key: Key, name: String, kind: &str) -> bool {
        let name = unique_name(&self.channels, &name);
        let log = unique_log(&self.channels, &sanitize(&name));
        self.channels.insert(key, Channel { name, kind: kind.into(), log });
        true
    }

    /// Name a new channel would get, before registration: the learned topic
    /// title, else well-known fallbacks.
    fn candidate_name(&self, msg: &TgMessage) -> String {
        if msg.message_thread_id.is_some()
            && let Some(title) = msg.topic_title()
        {
            return title.to_string();
        }
        match msg.message_thread_id {
            // Thread 1 is the always-present General topic of a forum.
            Some(1) => "General".into(),
            Some(t) => format!("topic {t}"),
            None => match msg.chat.kind.as_str() {
                "private" => msg.chat.display_name(),
                _ => msg.chat.label(),
            },
        }
    }

    /// A freshly seen topic title replaces only an auto fallback name;
    /// a real name stays (renames are ignored by design).
    fn learn_topic(&mut self, key: Key, title: &str) -> bool {
        let Some(channel) = self.channels.get_mut(&key) else {
            return false;
        };
        if is_fallback(&key.1, &channel.name) && channel.name != title {
            channel.name = title.to_string();
            return true;
        }
        false
    }

    /// Record a passing message: register the channel, learn its topic
    /// title, append the log line. Edits are new lines, not replacements —
    /// the log is append-only history.
    pub fn observe_message(&mut self, msg: &TgMessage, edited: bool) {
        let key = (msg.chat.id, msg.message_thread_id);
        let mut changed = match self.channels.contains_key(&key) {
            true => false,
            false => self.insert(key, self.candidate_name(msg), &msg.chat.kind),
        };
        if let Some(title) = msg.topic_title() {
            changed |= self.learn_topic(key, title);
        }
        if let Some(text) = msg.text.as_ref().or(msg.caption.as_ref()) {
            let author = match &msg.from {
                Some(u) => u.display_name(),
                // Channel posts are authored by the channel itself.
                None => self.name_of(key),
            };
            let head = if edited { "EDIT " } else { "" };
            self.append_log(
                key,
                &format!(
                    "[{}] {head}#{} {author}: {text}",
                    timestamp(msg.date),
                    msg.message_id
                ),
            );
        }
        if changed {
            self.save();
        }
    }

    /// Record a reaction. Reactions carry no topic (API limitation), so they
    /// are logged on the chat-level channel.
    pub fn observe_reaction(&mut self, r: &MessageReaction) {
        let key = (r.chat.id, None);
        let name = match r.chat.kind.as_str() {
            "private" => r.chat.display_name(),
            _ => r.chat.label(),
        };
        let changed = match self.channels.contains_key(&key) {
            true => false,
            false => self.insert(key, name, &r.chat.kind),
        };
        let actor = match &r.user {
            Some(u) => u.display_name(),
            None => r
                .actor_chat
                .as_ref()
                .map(|c| c.display_name())
                .unwrap_or_else(|| "someone".into()),
        };
        let line = if r.new_reaction.is_empty() {
            format!(
                "[{}] UNREACT #{} by {actor}: {}",
                timestamp(r.date),
                r.message_id,
                render_reactions(&r.old_reaction)
            )
        } else {
            format!(
                "[{}] REACT #{} by {actor}: {} (was: {})",
                timestamp(r.date),
                r.message_id,
                render_reactions(&r.new_reaction),
                render_reactions(&r.old_reaction)
            )
        };
        self.append_log(key, &line);
        if changed {
            self.save();
        }
    }

    /// Append the agent's own outgoing message, so the log holds the full
    /// scene text including the bot's lines.
    pub fn log_sent(&self, key: Key, message_id: i64, text: &str) {
        self.append_log(
            key,
            &format!("[{}] SENT #{message_id}: {text}", timestamp(now())),
        );
    }

    /// Append the replacement text of an edited outgoing message.
    pub fn log_edited(&self, key: Key, message_id: i64, text: &str) {
        self.append_log(
            key,
            &format!("[{}] EDIT #{message_id}: {text}", timestamp(now())),
        );
    }

    fn append_log(&self, key: Key, line: &str) {
        let Some(channel) = self.channels.get(&key) else {
            return;
        };
        let append = || -> std::io::Result<()> {
            std::fs::create_dir_all(&self.chat_dir)?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.chat_dir.join(&channel.log))?;
            writeln!(file, "{line}")
        };
        if let Err(e) = append() {
            tracing::warn!(error = %e, log = %channel.log, "could not append to channel log");
        }
    }

    /// Live resolution of a model-supplied channel name: exact match first,
    /// then a unique case-insensitive one.
    pub fn resolve(&self, name: &str) -> Option<Key> {
        if let Some((key, _)) = self.channels.iter().find(|(_, c)| c.name == name) {
            return Some(*key);
        }
        let lower = name.to_lowercase();
        let mut hits = self
            .channels
            .iter()
            .filter(|(_, c)| c.name.to_lowercase() == lower);
        let key = hits.next()?.0;
        hits.next().is_none().then_some(*key)
    }

    /// Name shown to the model for an event from `key`; mirrors the
    /// registration fallbacks for channels not observed yet.
    pub fn name_of(&self, key: Key) -> String {
        match self.channels.get(&key) {
            Some(c) => c.name.clone(),
            None => match key.1 {
                Some(t) => format!("topic {t}"),
                None => format!("chat {}", key.0),
            },
        }
    }

    /// All channel names, sorted for display.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.channels.values().map(|c| c.name.as_str()).collect();
        names.sort_by_key(|n| n.to_lowercase());
        names
    }

    /// The channel list seeded into the session head at startup and after
    /// every compaction.
    pub fn list(&self) -> String {
        if self.channels.is_empty() {
            return "No channels observed yet.".into();
        }
        let section = |label: &str, private: bool| {
            let mut names: Vec<&str> = self
                .channels
                .values()
                .filter(|c| (c.kind == "private") == private)
                .map(|c| c.name.as_str())
                .collect();
            if names.is_empty() {
                return String::new();
            }
            names.sort_by_key(|n| n.to_lowercase());
            format!("{label}:\n* {}\n", names.join("\n* "))
        };
        format!(
            "{}\n{}",
            section("DM channels", true),
            section("Group channels", false)
        )
        .trim_end()
        .to_string()
    }

    /// One-time conversion of the legacy `chats.json` (per-chat ring buffer
    /// without a topic dimension, capped at 50): each chat's history is
    /// dumped into `<name>.legacy.log` and the source file is renamed aside.
    fn migrate_legacy(&mut self, state_dir: &Path) {
        let legacy = state_dir.join("chats.json");
        let Ok(raw) = std::fs::read_to_string(&legacy) else {
            return;
        };
        let Ok(old) = serde_json::from_str::<Legacy>(&raw) else {
            tracing::warn!("legacy chats.json is unreadable; leaving it in place");
            return;
        };
        for (chat_id, chat) in &old.chats {
            let name = chat.title.clone().unwrap_or_else(|| format!("chat {chat_id}"));
            let key = (*chat_id, None);
            if !self.channels.contains_key(&key) {
                let log = unique_log(&self.channels, &format!("{}.legacy", sanitize(&name)));
                self.channels
                    .insert(key, Channel { name, kind: chat.kind.clone(), log });
            }
            if let Some(entries) = old.history.get(chat_id) {
                for e in entries {
                    self.append_log(
                        key,
                        &format!(
                            "[{}] #{} {}: {}",
                            timestamp(e.date),
                            e.message_id,
                            e.author,
                            e.text
                        ),
                    );
                }
            }
        }
        match std::fs::rename(&legacy, state_dir.join("chats.json.migrated")) {
            Ok(()) => {
                self.save();
                tracing::info!("migrated legacy chats.json into channel logs");
            }
            Err(e) => tracing::warn!(error = %e, "could not rename legacy chats.json"),
        }
    }
}

/// JSON row of the registry: serde_json cannot use tuple keys in objects,
/// so the (chat_id, thread) address is stored alongside the channel.
#[derive(Serialize, Deserialize)]
struct Row {
    chat_id: i64,
    #[serde(default)]
    thread: Option<i64>,
    name: String,
    kind: String,
    log: String,
}

fn rows_into_map(rows: Vec<Row>) -> BTreeMap<Key, Channel> {
    rows.into_iter()
        .map(|r| {
            (
                (r.chat_id, r.thread),
                Channel { name: r.name, kind: r.kind, log: r.log },
            )
        })
        .collect()
}

/// A thread's name is still an auto fallback if it was never learned.
fn is_fallback(thread: &Option<i64>, name: &str) -> bool {
    match *thread {
        Some(1) => name == "General" || name == "topic 1",
        Some(t) => name == format!("topic {t}"),
        None => false,
    }
}

fn unique_name(existing: &BTreeMap<Key, Channel>, name: &str) -> String {
    let taken: Vec<&str> = existing.values().map(|c| c.name.as_str()).collect();
    if !taken.contains(&name) {
        return name.to_string();
    }
    // A dash suffix survives `sanitize` unchanged, so the display name and
    // the log file stem stay in sync.
    (2..)
        .map(|n| format!("{name}-{n}"))
        .find(|n| !taken.contains(&n.as_str()))
        .expect("unbounded probe over free names")
}

fn unique_log(existing: &BTreeMap<Key, Channel>, stem: &str) -> String {
    let taken: Vec<&str> = existing.values().map(|c| c.log.as_str()).collect();
    (1..)
        .map(|n| if n == 1 { format!("{stem}.log") } else { format!("{stem}-{n}.log") })
        .find(|log| !taken.contains(&log.as_str()))
        .expect("unbounded probe over free names")
}

/// Filesystem-safe rendering of a channel name for log file names; display
/// names themselves are never sanitized.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c: char| matches!(c, ' ' | '.' | '_'));
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed.to_string()
    }
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Shape of the legacy `state/chats.json`, read only for migration.
#[derive(Deserialize)]
struct Legacy {
    #[serde(default)]
    chats: BTreeMap<i64, LegacyChat>,
    #[serde(default)]
    history: BTreeMap<i64, Vec<LegacyEntry>>,
}

#[derive(Deserialize)]
struct LegacyChat {
    kind: String,
    title: Option<String>,
}

#[derive(Deserialize)]
struct LegacyEntry {
    #[serde(default)]
    date: i64,
    message_id: i64,
    author: String,
    #[serde(default)]
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tg::TgChat;

    impl Channels {
        /// A registry without persisted state, for tests.
        pub(crate) fn empty(state_dir: &Path, workspace_dir: &Path) -> Self {
            Self {
                path: state_dir.join("channels.json"),
                chat_dir: workspace_dir.join("chat"),
                channels: BTreeMap::new(),
            }
        }
    }

    fn msg(id: i64, thread: Option<i64>, text: &str) -> TgMessage {
        TgMessage {
            message_id: id,
            from: Some(crate::tg::TgUser {
                id: 1,
                is_bot: false,
                first_name: "Alice".into(),
                last_name: String::new(),
                username: None,
            }),
            chat: TgChat {
                id: -100,
                kind: "supergroup".into(),
                title: Some("RP".into()),
                first_name: String::new(),
                last_name: String::new(),
                username: None,
            },
            message_thread_id: thread,
            date: 1_758_000_000,
            text: Some(text.into()),
            caption: None,
            photo: vec![],
            document: None,
            reply_to_message: None,
            forum_topic_created: None,
            forum_topic_edited: None,
        }
    }

    fn store(dir: &Path) -> Channels {
        Channels::empty(&dir.join("state"), dir)
    }

    #[test]
    fn threads_become_named_channels_with_logs() {
        let dir = std::env::temp_dir().join(format!("rp-ch1-{}", std::process::id()));
        let mut channels = store(&dir);
        // A message in a topic whose title rides in via reply_to_message.
        let mut m = msg(5, Some(42), "scene text");
        m.reply_to_message = Some(Box::new(TgMessage {
            forum_topic_created: Some(crate::tg::ForumTopicCreated { name: "IC".into() }),
            ..msg(2, None, "")
        }));
        channels.observe_message(&m, false);
        channels.observe_message(&msg(6, Some(42), "reply"), false);
        assert_eq!(channels.resolve("IC"), Some((-100, Some(42))));
        assert_eq!(channels.resolve("ic"), Some((-100, Some(42))));
        assert_eq!(channels.resolve("OOC"), None);
        let log = std::fs::read_to_string(dir.join("chat/IC.log")).unwrap();
        assert!(
            log.contains("[2025-09-16 05:20:00 UTC] #5 Alice: scene text"),
            "{log}"
        );
        assert!(log.contains("#6 Alice: reply"), "{log}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edits_are_new_lines_and_registry_persists() {
        let dir = std::env::temp_dir().join(format!("rp-ch2-{}", std::process::id()));
        let state = dir.join("state");
        std::fs::create_dir_all(&state).unwrap();
        {
            let mut channels = Channels::empty(&state, &dir);
            channels.observe_message(&msg(7, None, "hello"), false);
            channels.observe_message(&msg(7, None, "hello edited"), true);
        }
        let channels = Channels::load(&state, &dir);
        // A topic-less supergroup registers under its title.
        assert_eq!(channels.resolve("RP"), Some((-100, None)));
        let log = std::fs::read_to_string(dir.join("chat/RP.log")).unwrap();
        assert_eq!(log.lines().count(), 2, "{log}");
        assert!(log.contains("EDIT #7 Alice: hello edited"), "{log}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reactions_land_on_the_chat_level_channel() {
        let dir = std::env::temp_dir().join(format!("rp-ch3-{}", std::process::id()));
        let mut channels = store(&dir);
        channels.observe_reaction(&crate::tg::MessageReaction {
            chat: TgChat {
                id: -100,
                kind: "supergroup".into(),
                title: Some("RP".into()),
                first_name: String::new(),
                last_name: String::new(),
                username: None,
            },
            message_id: 11,
            user: Some(crate::tg::TgUser {
                id: 1,
                is_bot: false,
                first_name: "Alice".into(),
                last_name: String::new(),
                username: None,
            }),
            actor_chat: None,
            date: 1_758_000_002,
            old_reaction: vec![],
            new_reaction: vec![crate::tg::ReactionType {
                kind: "emoji".into(),
                emoji: Some("🔥".into()),
            }],
        });
        assert_eq!(channels.resolve("RP"), Some((-100, None)));
        let log = std::fs::read_to_string(dir.join("chat/RP.log")).unwrap();
        assert!(log.contains("REACT #11 by Alice: 🔥 (was: (none))"), "{log}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn own_sends_are_logged() {
        let dir = std::env::temp_dir().join(format!("rp-ch4-{}", std::process::id()));
        let mut channels = store(&dir);
        channels.observe_message(&msg(7, Some(42), "hi"), false);
        let key = (-100, Some(42));
        channels.log_sent(key, 555, "my turn");
        channels.log_edited(key, 555, "my turn, fixed");
        let log = std::fs::read_to_string(dir.join("chat/topic 42.log")).unwrap();
        assert!(log.contains("SENT #555: my turn"), "{log}");
        assert!(log.contains("EDIT #555: my turn, fixed"), "{log}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn colliding_names_get_distinct_files() {
        let dir = std::env::temp_dir().join(format!("rp-ch5-{}", std::process::id()));
        let mut channels = store(&dir);
        channels.observe_message(&msg(7, None, "group one"), false);
        let mut other = msg(8, None, "group two");
        other.chat.id = -200;
        channels.observe_message(&other, false);
        assert_eq!(channels.names(), vec!["RP", "RP-2"]);
        assert!(dir.join("chat/RP.log").exists());
        assert!(dir.join("chat/RP-2.log").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn learned_names_stick_and_later_renames_are_ignored() {
        let dir = std::env::temp_dir().join(format!("rp-ch6-{}", std::process::id()));
        let mut channels = store(&dir);
        // First sight without a topic title: fallback name, upgraded once
        // the real title is learned.
        channels.observe_message(&msg(7, Some(42), "hi"), false);
        assert_eq!(channels.name_of((-100, Some(42))), "topic 42");
        let mut creation = msg(8, Some(42), "still here");
        creation.forum_topic_edited = Some(crate::tg::ForumTopicEdited {
            name: Some("IC".into()),
        });
        channels.observe_message(&creation, false);
        assert_eq!(channels.name_of((-100, Some(42))), "IC");
        // A rename of a real-named channel changes nothing.
        channels.observe_message(&creation, false);
        assert_eq!(channels.name_of((-100, Some(42))), "IC");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_history_is_dumped_and_source_renamed() {
        let dir = std::env::temp_dir().join(format!("rp-ch7-{}", std::process::id()));
        let state = dir.join("state");
        std::fs::create_dir_all(&state).unwrap();
        let legacy = serde_json::json!({
            "chats": { "-100": { "kind": "supergroup", "title": "RP" } },
            "history": { "-100": [
                { "date": 1758000000, "message_id": 1, "author": "Alice", "text": "old one" },
                { "date": 1758000001, "message_id": 2, "author": "Alice", "text": "old two" }
            ]}
        });
        std::fs::write(state.join("chats.json"), legacy.to_string()).unwrap();
        let channels = Channels::load(&state, &dir);
        assert_eq!(channels.resolve("RP"), Some((-100, None)));
        let log = std::fs::read_to_string(dir.join("chat/RP.legacy.log")).unwrap();
        assert!(log.contains("#1 Alice: old one"), "{log}");
        assert!(!state.join("chats.json").exists());
        assert!(state.join("chats.json.migrated").exists());
        // Migration is one-shot; the registry itself persists.
        drop(channels);
        let channels = Channels::load(&state, &dir);
        assert_eq!(channels.names(), vec!["RP"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_sections_are_sorted() {
        let dir = std::env::temp_dir().join(format!("rp-ch8-{}", std::process::id()));
        let mut channels = store(&dir);
        channels.observe_message(&msg(7, Some(42), "a"), false);
        channels.observe_message(&msg(8, None, "b"), false);
        let mut dm = msg(9, None, "dm");
        dm.chat = TgChat {
            id: 3,
            kind: "private".into(),
            title: None,
            first_name: "Luna".into(),
            last_name: "Spirito".into(),
            username: None,
        };
        channels.observe_message(&dm, false);
        let list = channels.list();
        assert!(list.contains("DM channels:\n* Luna Spirito\n"), "{list}");
        let groups = list.split("Group channels:\n").nth(1).unwrap();
        assert!(groups.find("RP").unwrap() < groups.find("topic 42").unwrap(), "{list}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sanitize_keeps_names_readably_safe() {
        assert_eq!(sanitize("IC"), "IC");
        assert_eq!(sanitize("a/b:c?"), "a_b_c");
        assert_eq!(sanitize("///"), "unnamed");
    }
}
