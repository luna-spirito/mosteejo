//! Registry of chats the bot has seen and a per-chat ring buffer of recent
//! messages, persisted as one JSON file in the state dir. The Bot API can
//! neither enumerate chats nor fetch history, so both are accumulated from
//! updates that passed the ingress filter and served to the agent via the
//! `list_chats` and `read_chat` tools.

use crate::tg::{TgChat, TgMessage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Recent messages kept per chat.
const HISTORY_CAP: usize = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatInfo {
    pub kind: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub date: i64,
    pub message_id: i64,
    pub author: String,
    pub text: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Persisted {
    chats: BTreeMap<i64, ChatInfo>,
    history: BTreeMap<i64, Vec<HistoryEntry>>,
}

/// Owned by the agent (single writer); mutated in `collect_events` and read
/// by tools through a shared borrow.
pub struct Chats {
    path: PathBuf,
    data: Persisted,
}

impl Chats {
    pub fn load(state_dir: &Path) -> Self {
        let path = PathBuf::from(state_dir).join("chats.json");
        let data = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, data }
    }

    fn save(&self) {
        if let Ok(line) = serde_json::to_string(&self.data)
            && let Err(e) = std::fs::write(&self.path, line)
        {
            tracing::warn!(error = %e, "could not persist chat registry");
        }
    }

    fn record_chat(&mut self, chat: &TgChat) {
        self.data.chats.insert(
            chat.id,
            ChatInfo { kind: chat.kind.clone(), title: chat.title.clone() },
        );
    }

    pub fn observe_chat(&mut self, chat: &TgChat) {
        self.record_chat(chat);
        self.save();
    }

    /// Record a message in its chat's history; an edit replaces the stored
    /// text in place. Media without caption has nothing to re-read later.
    pub fn observe_message(&mut self, msg: &TgMessage) {
        self.record_chat(&msg.chat);
        let Some(text) = msg.text.clone().or_else(|| msg.caption.clone()) else {
            self.save();
            return;
        };
        let author = match &msg.from {
            Some(u) => u.display_name(),
            // Channel posts are authored by the channel itself.
            None => msg.chat.label(),
        };
        let entry = HistoryEntry { date: msg.date, message_id: msg.message_id, author, text };
        let history = self.data.history.entry(msg.chat.id).or_default();
        match history.iter().position(|e| e.message_id == msg.message_id) {
            Some(i) => history[i] = entry,
            None => {
                history.push(entry);
                let excess = history.len().saturating_sub(HISTORY_CAP);
                history.drain(..excess);
            }
        }
        self.save();
    }

    pub fn known(&self) -> &BTreeMap<i64, ChatInfo> {
        &self.data.chats
    }

    /// The `limit` newest entries, oldest first.
    pub fn recent(&self, chat_id: i64, limit: usize) -> Option<&[HistoryEntry]> {
        let history = self.data.history.get(&chat_id)?;
        let start = history.len().saturating_sub(limit);
        Some(&history[start..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(id: i64, text: &str) -> TgMessage {
        TgMessage {
            message_id: id,
            from: Some(crate::tg::TgUser {
                id: 1,
                is_bot: false,
                first_name: "Alice".into(),
                last_name: String::new(),
                username: None,
            }),
            chat: TgChat { id: -100, kind: "supergroup".into(), title: Some("RP".into()) },
            message_thread_id: None,
            date: 0,
            text: Some(text.into()),
            caption: None,
            photo: vec![],
            document: None,
            reply_to_message: None,
        }
    }

    fn store(dir: &Path) -> Chats {
        let state = dir.join("state");
        std::fs::create_dir_all(&state).unwrap();
        Chats::load(&state)
    }

    #[test]
    fn messages_record_and_edits_upsert() {
        let dir = std::env::temp_dir().join(format!("rp-chats-{}", std::process::id()));
        let mut chats = store(&dir);
        chats.observe_message(&msg(1, "hello"));
        chats.observe_message(&msg(2, "world"));
        chats.observe_message(&msg(1, "hello edited"));
        let recent = chats.recent(-100, 10).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].text, "hello edited");
        assert_eq!(recent[0].author, "Alice");
        // The chat registered itself under its title.
        assert_eq!(chats.known().get(&-100).unwrap().title.as_deref(), Some("RP"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn history_is_capped_and_persisted() {
        let dir = std::env::temp_dir().join(format!("rp-chats2-{}", std::process::id()));
        let state = dir.join("state");
        {
            let mut chats = store(&dir);
            for id in 0..(HISTORY_CAP as i64 + 10) {
                chats.observe_message(&msg(id, "m"));
            }
        }
        // A fresh load reads back what was persisted.
        let chats = Chats::load(&state);
        let recent = chats.recent(-100, usize::MAX).unwrap();
        assert_eq!(recent.len(), HISTORY_CAP);
        assert_eq!(recent[0].message_id, 10);
        std::fs::remove_dir_all(&dir).ok();
    }
}
