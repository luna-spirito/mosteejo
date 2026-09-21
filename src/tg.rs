//! Telegram Bot API client (long polling + sending) and the strict ingress
//! filter. Updates from users outside the allowlist never leave this module —
//! they are dropped before anything can observe them, which is the first
//! line of defense against prompt injection.

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use pulldown_cmark::{Event as MdEvent, Options, Parser, Tag, TagEnd};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

const LONG_POLL_SECS: u64 = 50;
const MAX_DOWNLOAD_BYTES: i64 = 20 * 1024 * 1024;

#[derive(Clone)]
pub struct Telegram {
    http: reqwest::Client,
    api: String,
    files: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Update {
    pub update_id: i64,
    #[serde(default)]
    pub message: Option<TgMessage>,
    #[serde(default)]
    pub edited_message: Option<TgMessage>,
    #[serde(default)]
    pub message_reaction: Option<MessageReaction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageReaction {
    pub chat: TgChat,
    pub message_id: i64,
    /// The reacting user; absent for anonymous (admin) reactions, where
    /// `actor_chat` identifies the chat acting instead.
    #[serde(default)]
    pub user: Option<TgUser>,
    #[serde(default)]
    pub actor_chat: Option<TgChat>,
    pub date: i64,
    #[serde(default)]
    pub old_reaction: Vec<ReactionType>,
    #[serde(default)]
    pub new_reaction: Vec<ReactionType>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReactionType {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub emoji: Option<String>,
}

impl Update {
    pub fn msg(&self) -> Option<&TgMessage> {
        self.message.as_ref().or(self.edited_message.as_ref())
    }

    pub fn edited(&self) -> bool {
        self.edited_message.is_some()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TgMessage {
    pub message_id: i64,
    #[serde(default)]
    pub from: Option<TgUser>,
    pub chat: TgChat,
    #[serde(default)]
    pub message_thread_id: Option<i64>,
    pub date: i64,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub photo: Vec<TgPhotoSize>,
    #[serde(default)]
    pub document: Option<TgDocument>,
    #[serde(default)]
    pub reply_to_message: Option<Box<TgMessage>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TgUser {
    pub id: i64,
    #[serde(default)]
    pub is_bot: bool,
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub last_name: String,
    #[serde(default)]
    pub username: Option<String>,
}

impl TgUser {
    pub fn display_name(&self) -> String {
        format!("{} {}", self.first_name, self.last_name)
            .trim_end()
            .to_string()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TgChat {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub title: Option<String>,
}

impl TgChat {
    /// Displayable chat name: title when known, else the bare id.
    pub fn label(&self) -> String {
        self.title
            .clone()
            .unwrap_or_else(|| format!("chat {}", self.id))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TgPhotoSize {
    pub file_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TgDocument {
    pub file_id: String,
    #[serde(default)]
    pub file_name: Option<String>,
}

/// Outcome of the ingress filter.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Never enters the system.
    Drop,
    /// Buffered for the next wake, whenever that is.
    Queue,
    /// Buffered and the agent wakes up right now.
    Wake,
}

#[derive(Debug, Clone)]
pub struct WatchList {
    pub allowed_users: Vec<i64>,
    pub subscribed_topics: Vec<(i64, i64)>,
    pub bot_id: i64,
    pub bot_username: String,
}

pub fn classify(update: &Update, watch: &WatchList) -> Verdict {
    if let Some(reaction) = &update.message_reaction {
        return classify_reaction(reaction, watch);
    }
    let Some(msg) = update.msg() else {
        return Verdict::Drop;
    };
    let Some(from) = &msg.from else {
        // Service messages (joins, pins) and channel posts carry no author:
        // pass without wake.
        return Verdict::Queue;
    };
    if from.is_bot || !watch.allowed_users.contains(&from.id) {
        return Verdict::Drop;
    }
    if msg.chat.kind == "private"
        || watch
            .subscribed_topics
            .contains(&(msg.chat.id, msg.message_thread_id.unwrap_or(0)))
    {
        return Verdict::Wake;
    }
    let text = msg.text.as_deref().unwrap_or_default();
    if text.starts_with('/')
        || text.contains(&format!("@{}", watch.bot_username))
        || msg
            .reply_to_message
            .as_ref()
            .and_then(|r| r.from.as_ref())
            .is_some_and(|u| u.id == watch.bot_id)
    {
        return Verdict::Wake;
    }
    Verdict::Queue
}

/// Reactions follow the message rules: private chats and subscribed chats
/// wake, everything else buffers. Actors outside the allowlist are dropped.
/// Reactions carry no topic, so a subscription matches on chat id alone.
fn classify_reaction(reaction: &MessageReaction, watch: &WatchList) -> Verdict {
    let allowed = match (&reaction.user, &reaction.actor_chat) {
        (Some(u), _) => !u.is_bot && watch.allowed_users.contains(&u.id),
        (None, Some(c)) => watch.allowed_users.contains(&c.id),
        (None, None) => false,
    };
    if !allowed {
        return Verdict::Drop;
    }
    if reaction.chat.kind == "private"
        || watch
            .subscribed_topics
            .iter()
            .any(|(c, _)| *c == reaction.chat.id)
    {
        return Verdict::Wake;
    }
    Verdict::Queue
}

/// A downloaded attachment referenced by the formatted wake message.
#[derive(Debug)]
pub struct Attachment {
    pub path: PathBuf,
    pub is_image: bool,
}

/// One filtered update prepared for the agent: either a (possibly edited)
/// message with attachments on disk, or a reaction notification.
#[derive(Debug)]
pub struct Event {
    pub kind: EventKind,
}

#[derive(Debug)]
pub enum EventKind {
    Message {
        msg: TgMessage,
        edited: bool,
        attachments: Vec<Attachment>,
    },
    Reaction(MessageReaction),
}

impl Event {
    pub fn chat(&self) -> &TgChat {
        match &self.kind {
            EventKind::Message { msg, .. } => &msg.chat,
            EventKind::Reaction(r) => &r.chat,
        }
    }
}

/// UTC rendering of a Telegram unix date, shared by the event formatter and
/// the chat-history tool.
pub fn timestamp(date: i64) -> String {
    chrono::DateTime::from_timestamp(date, 0)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_default()
}

#[derive(Deserialize)]
struct TgResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

impl Telegram {
    pub fn new(api_url: &str, token: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(LONG_POLL_SECS + 30))
            .build()?;
        Ok(Self {
            http,
            api: format!("{api_url}/bot{token}"),
            files: format!("{api_url}/file/bot{token}"),
        })
    }

    async fn call<T: serde::de::DeserializeOwned>(&self, method: &str, body: &Value) -> Result<T> {
        let resp = self
            .http
            .post(format!("{}/{}", self.api, method))
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        let parsed: TgResponse<T> = serde_json::from_str(&text)
            .with_context(|| format!("{method}: malformed response {text}"))?;
        match (parsed.ok, parsed.result) {
            (true, Some(result)) => Ok(result),
            _ => Err(anyhow!(
                "{method} failed ({}): {}",
                status,
                parsed.description.unwrap_or_default()
            )),
        }
    }

    pub async fn get_me(&self) -> Result<TgUser> {
        self.call("getMe", &json!({})).await
    }

    pub async fn get_updates(&self, offset: Option<i64>) -> Result<Vec<Update>> {
        let updates: Vec<Update> = self
            .call(
                "getUpdates",
                &json!({
                    "offset": offset,
                    "timeout": LONG_POLL_SECS,
                    "allowed_updates": ["message", "edited_message", "message_reaction"],
                }),
            )
            .await?;
        Ok(updates)
    }

    /// Returns the id of the sent message so the agent can edit it later.
    pub async fn send_message(
        &self,
        chat_id: i64,
        thread_id: Option<i64>,
        text: &str,
        parse_mode: Option<&str>,
        reply_to: Option<i64>,
    ) -> Result<i64> {
        let mut body = json!({ "chat_id": chat_id, "text": text });
        if let Some(t) = thread_id {
            body["message_thread_id"] = json!(t);
        }
        if let Some(p) = parse_mode {
            body["parse_mode"] = json!(p);
        }
        if let Some(r) = reply_to {
            body["reply_to_message_id"] = json!(r);
            body["allow_sending_without_reply"] = json!(true);
        }
        let sent: Value = self.call("sendMessage", &body).await?;
        Ok(sent.get("message_id").and_then(Value::as_i64).unwrap_or(0))
    }

    /// Replace the text of an existing message (only the bot's own).
    pub async fn edit_message_text(
        &self,
        chat_id: i64,
        message_id: i64,
        text: &str,
        parse_mode: Option<&str>,
    ) -> Result<()> {
        let mut body = json!({ "chat_id": chat_id, "message_id": message_id, "text": text });
        if let Some(p) = parse_mode {
            body["parse_mode"] = json!(p);
        }
        self.call("editMessageText", &body).await.map(|_: Value| ())
    }

    pub async fn send_typing(&self, chat_id: i64) -> Result<()> {
        self.call(
            "sendChatAction",
            &json!({ "chat_id": chat_id, "action": "typing" }),
        )
        .await
        .map(|_: Value| ())
    }

    async fn download(&self, file_id: &str) -> Result<Vec<u8>> {
        #[derive(Deserialize)]
        struct TgFile {
            file_path: Option<String>,
            #[serde(default)]
            file_size: Option<i64>,
        }
        let file: TgFile = self.call("getFile", &json!({ "file_id": file_id })).await?;
        let size = file.file_size.unwrap_or(0);
        if size > MAX_DOWNLOAD_BYTES {
            return Err(anyhow!("file too large: {size} bytes"));
        }
        let path = file.file_path.context("getFile returned no file_path")?;
        let bytes = self
            .http
            .get(format!("{}/{path}", self.files))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(bytes.to_vec())
    }

    /// Download photo/document attachments into `downloads_dir`; text-only
    /// messages produce no attachments.
    pub async fn fetch_attachments(
        &self,
        msg: &TgMessage,
        downloads_dir: &Path,
    ) -> Result<Vec<Attachment>> {
        let mut out = Vec::new();
        std::fs::create_dir_all(downloads_dir)?;
        if let Some(photo) = msg.photo.last() {
            let path = downloads_dir.join(format!("photo_{}.jpg", msg.message_id));
            std::fs::write(&path, self.download(&photo.file_id).await?)?;
            out.push(Attachment {
                path,
                is_image: true,
            });
        }
        if let Some(doc) = &msg.document {
            let name = doc
                .file_name
                .as_deref()
                .map(|n| n.rsplit('/').next().unwrap_or(n).to_string())
                .unwrap_or_else(|| format!("file_{}", msg.message_id));
            let path = downloads_dir.join(format!("{}_{}", msg.message_id, name));
            std::fs::write(&path, self.download(&doc.file_id).await?)?;
            out.push(Attachment {
                path,
                is_image: false,
            });
        }
        Ok(out)
    }

    pub fn image_data_url(bytes: &[u8], mime: &str) -> String {
        format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }
}

/// Render model-authored Markdown into the HTML dialect Telegram accepts
/// (`parse_mode: "HTML"`). The output is always balanced regardless of what
/// the model wrote, so sendMessage never fails on entity parsing; constructs
/// Telegram cannot display (tables, raw HTML, inline images) degrade to text.
/// Line spacing is preserved from the source: single breaks stay single and
/// paragraph breaks keep their blank line.
pub fn render_markdown(src: &str) -> String {
    let mut out = String::new();
    let mut lists: Vec<Option<u64>> = Vec::new();
    // Strikethrough on; tables stay off on purpose — Telegram cannot show them.
    let parser = Parser::new_ext(src, Options::ENABLE_STRIKETHROUGH).into_offset_iter();
    // End of the last content event (text, code, rule, …): container ranges
    // swallow the trailing newlines of a block, so the source gap before the
    // next block is measured from real content only.
    let mut content_end = 0;
    for (ev, range) in parser {
        let gap_lines = src[content_end.min(range.start)..range.start].matches('\n').count();
        let is_content = !matches!(ev, MdEvent::Start(_) | MdEvent::End(_));
        match ev {
            MdEvent::Start(tag) => match tag {
                Tag::Paragraph => block_gap(&mut out, gap_lines),
                Tag::Heading { .. } => {
                    block_gap(&mut out, gap_lines);
                    out.push_str("<b>");
                }
                Tag::Emphasis => out.push_str("<i>"),
                Tag::Strong => out.push_str("<b>"),
                Tag::Strikethrough => out.push_str("<s>"),
                Tag::BlockQuote(_) => {
                    block_gap(&mut out, gap_lines);
                    out.push_str("<blockquote>");
                }
                Tag::CodeBlock(_) => {
                    block_gap(&mut out, gap_lines);
                    out.push_str("<pre>");
                }
                Tag::List(start) => {
                    block_gap(&mut out, gap_lines);
                    lists.push(start);
                }
                Tag::Item => {
                    ensure_newline(&mut out);
                    match lists.last_mut() {
                        Some(Some(n)) => {
                            out.push_str(&format!("{n}. "));
                            *n += 1;
                        }
                        _ => out.push_str("• "),
                    }
                }
                Tag::Link { dest_url, .. } => {
                    out.push_str(&format!("<a href=\"{}\">", escape_attr(&dest_url)))
                }
                Tag::Image { dest_url, .. } => {
                    out.push_str(&format!("[<a href=\"{}\">", escape_attr(&dest_url)))
                }
                _ => {}
            },
            MdEvent::End(tag_end) => match tag_end {
                TagEnd::Paragraph => out.push('\n'),
                TagEnd::Heading(_) => out.push_str("</b>\n"),
                TagEnd::Emphasis => out.push_str("</i>"),
                TagEnd::Strong => out.push_str("</b>"),
                TagEnd::Strikethrough => out.push_str("</s>"),
                TagEnd::BlockQuote(_) => out.push_str("</blockquote>\n"),
                TagEnd::CodeBlock => out.push_str("</pre>\n"),
                TagEnd::Item => ensure_newline(&mut out),
                TagEnd::List(_) => {
                    lists.pop();
                }
                TagEnd::Link => out.push_str("</a>"),
                TagEnd::Image => out.push_str("</a>]"),
                _ => {}
            },
            MdEvent::Code(t) => out.push_str(&format!("<code>{}</code>", escape(&t))),
            MdEvent::Text(t) => out.push_str(&escape(&t)),
            MdEvent::SoftBreak | MdEvent::HardBreak => out.push('\n'),
            MdEvent::Rule => {
                block_gap(&mut out, gap_lines);
                out.push_str("—\n");
            }
            MdEvent::TaskListMarker(done) => out.push_str(if done { "[x] " } else { "[ ] " }),
            // Raw HTML is shown literally, never passed through: unknown tags
            // would make Telegram reject the whole message.
            MdEvent::Html(h) | MdEvent::InlineHtml(h) => out.push_str(&escape(&h)),
            _ => {}
        }
        if is_content {
            content_end = range.end;
        }
    }
    out.trim().to_string()
}

fn ensure_newline(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

/// Blank-line separation before a block element, taken from the source: the
/// previous block's end already emitted one newline, so at most one more is
/// added — markdown treats runs of newlines as a single break, and so do we.
/// Mid-line (right after an opening tag like `<blockquote>`) at least one
/// newline is still guaranteed.
fn block_gap(out: &mut String, gap_lines: usize) {
    if out.is_empty() {
        return;
    }
    let extra = if out.ends_with('\n') {
        gap_lines.saturating_sub(1).min(1)
    } else {
        gap_lines.clamp(1, 2)
    };
    out.push_str(&"\n".repeat(extra));
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(text: &str) -> String {
    escape(text).replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_renders_to_telegram_html() {
        assert_eq!(
            render_markdown("**b** *i* ~~s~~ `c`"),
            "<b>b</b> <i>i</i> <s>s</s> <code>c</code>"
        );
        assert_eq!(
            render_markdown("[x](https://e.x)"),
            "<a href=\"https://e.x\">x</a>"
        );
        assert_eq!(render_markdown("- a\n- b\n1. c"), "• a\n• b\n1. c");
        assert_eq!(render_markdown("# Head\ntext"), "<b>Head</b>\ntext");
        assert_eq!(
            render_markdown("> quoted"),
            "<blockquote>\nquoted\n</blockquote>"
        );
        assert_eq!(render_markdown("```\n<x>\n```"), "<pre>&lt;x&gt;\n</pre>");
        // Raw HTML and special chars never pass through unescaped.
        assert_eq!(render_markdown("a <b> & c"), "a &lt;b&gt; &amp; c");
        assert_eq!(render_markdown("<u>raw</u>"), "&lt;u&gt;raw&lt;/u&gt;");
        // Unbalanced markers stay literal instead of producing broken HTML.
        assert_eq!(render_markdown("2 * 3 ** 4"), "2 * 3 ** 4");
    }

    #[test]
    fn markdown_spacing() {
        assert_eq!(render_markdown("Hi\nHello"), "Hi\nHello");
        assert_eq!(render_markdown("Hi\n\nHello"), "Hi\n\nHello");
        // Runs of blank lines collapse to a single blank line, like markdown
        // itself treats them.
        assert_eq!(render_markdown("Hi\n\n\n\nHello"), "Hi\n\nHello");
        // Gaps come from the source, they are not forced between every block.
        assert_eq!(render_markdown("# Head\ntext"), "<b>Head</b>\ntext");
        assert_eq!(render_markdown("# Head\n\ntext"), "<b>Head</b>\n\ntext");
        assert_eq!(render_markdown("a\n- x"), "a\n• x");
        assert_eq!(render_markdown("a\n\n- x"), "a\n\n• x");
        assert_eq!(
            render_markdown("> quoted\n\ntail"),
            "<blockquote>\nquoted\n</blockquote>\n\ntail"
        );
    }

    fn update(
        from_id: Option<i64>,
        chat_kind: &str,
        chat_id: i64,
        thread: Option<i64>,
        text: &str,
    ) -> Update {
        Update {
            update_id: 1,
            message: Some(TgMessage {
                message_id: 10,
                from: from_id.map(|id| TgUser {
                    id,
                    is_bot: false,
                    first_name: "Alice".into(),
                    last_name: String::new(),
                    username: Some("alice".into()),
                }),
                chat: TgChat {
                    id: chat_id,
                    kind: chat_kind.into(),
                    title: None,
                },
                message_thread_id: thread,
                date: 0,
                text: Some(text.into()),
                caption: None,
                photo: vec![],
                document: None,
                reply_to_message: None,
            }),
            edited_message: None,
            message_reaction: None,
        }
    }

    fn reaction(from_id: Option<i64>, chat_kind: &str, chat_id: i64) -> Update {
        Update {
            update_id: 1,
            message: None,
            edited_message: None,
            message_reaction: Some(MessageReaction {
                chat: TgChat {
                    id: chat_id,
                    kind: chat_kind.into(),
                    title: None,
                },
                message_id: 10,
                user: from_id.map(|id| TgUser {
                    id,
                    is_bot: false,
                    first_name: "Alice".into(),
                    last_name: String::new(),
                    username: Some("alice".into()),
                }),
                actor_chat: None,
                date: 0,
                old_reaction: vec![],
                new_reaction: vec![ReactionType {
                    kind: "emoji".into(),
                    emoji: Some("🔥".into()),
                }],
            }),
        }
    }

    fn watch() -> WatchList {
        WatchList {
            allowed_users: vec![1],
            subscribed_topics: vec![(-100, 42)],
            bot_id: 7,
            bot_username: "rp_bot".into(),
        }
    }

    #[test]
    fn disallowed_users_never_pass() {
        let w = watch();
        assert_eq!(
            classify(
                &update(Some(2), "supergroup", -100, Some(42), "injected"),
                &w
            ),
            Verdict::Drop
        );
        assert_eq!(
            classify(&update(Some(2), "private", 2, None, "hi"), &w),
            Verdict::Drop
        );
    }

    #[test]
    fn priority_channels_wake() {
        let w = watch();
        assert_eq!(
            classify(&update(Some(1), "private", 1, None, "hi"), &w),
            Verdict::Wake
        );
        assert_eq!(
            classify(&update(Some(1), "supergroup", -100, Some(42), "scene"), &w),
            Verdict::Wake
        );
        assert_eq!(
            classify(&update(Some(1), "supergroup", -100, Some(9), "/ping"), &w),
            Verdict::Wake
        );
        assert_eq!(
            classify(
                &update(Some(1), "supergroup", -100, Some(9), "@rp_bot yo"),
                &w
            ),
            Verdict::Wake
        );
        let mut reply = update(Some(1), "supergroup", -100, Some(9), "answer me");
        reply.message.as_mut().unwrap().reply_to_message = Some(Box::new(TgMessage {
            message_id: 1,
            from: Some(TgUser {
                id: 7,
                is_bot: true,
                first_name: "Bot".into(),
                last_name: String::new(),
                username: None,
            }),
            chat: TgChat {
                id: -100,
                kind: "supergroup".into(),
                title: None,
            },
            message_thread_id: Some(9),
            date: 0,
            text: Some("bot said".into()),
            caption: None,
            photo: vec![],
            document: None,
            reply_to_message: None,
        }));
        assert_eq!(classify(&reply, &w), Verdict::Wake);
    }

    #[test]
    fn side_channels_queue_without_wake() {
        let w = watch();
        assert_eq!(
            classify(
                &update(Some(1), "supergroup", -100, Some(9), "idle chat"),
                &w
            ),
            Verdict::Queue
        );
    }

    #[test]
    fn reactions_follow_chat_rules() {
        let w = watch();
        // Allowed user: subscribed chat wakes, others queue, private wakes.
        assert_eq!(
            classify(&reaction(Some(1), "supergroup", -100), &w),
            Verdict::Wake
        );
        assert_eq!(
            classify(&reaction(Some(1), "supergroup", -999), &w),
            Verdict::Queue
        );
        assert_eq!(
            classify(&reaction(Some(1), "private", 1), &w),
            Verdict::Wake
        );
        // Strangers and bots never pass.
        assert_eq!(
            classify(&reaction(Some(2), "private", 2), &w),
            Verdict::Drop
        );
    }

    #[test]
    fn reaction_payload_parses() {
        let raw = r#"{"update_id": 5, "message_reaction": {
            "chat": {"id": -100, "type": "supergroup", "title": "RP"},
            "message_id": 11,
            "user": {"id": 1, "is_bot": false, "first_name": "Alice"},
            "date": 1758000002,
            "old_reaction": [{"type": "emoji", "emoji": "👍"}],
            "new_reaction": [{"type": "emoji", "emoji": "🔥"},
                             {"type": "custom_emoji", "custom_emoji_id": "42"}]
        }}"#;
        let update: Update = serde_json::from_str(raw).unwrap();
        let r = update.message_reaction.as_ref().unwrap();
        assert_eq!(r.chat.label(), "RP");
        assert_eq!(r.new_reaction.len(), 2);
        assert_eq!(r.new_reaction[0].emoji.as_deref(), Some("🔥"));
        assert_eq!(r.old_reaction[0].emoji.as_deref(), Some("👍"));
    }

    #[test]
    fn service_messages_queue() {
        let w = watch();
        assert_eq!(
            classify(&update(None, "supergroup", -100, Some(42), ""), &w),
            Verdict::Queue
        );
    }
}
