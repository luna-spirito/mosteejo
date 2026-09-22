//! The agent runtime: wake-ups, turns, tool dispatch and compaction.
//!
//! Wake model: priority updates (DMs, subscribed topics, pings there,
//! reactions in them) signal an immediate wake; everything else waits for
//! the idle tick. On every wake the whole buffered queue is replayed into
//! the conversation as one user message, then a normal turn runs
//! (LLM <-> tools) until the model stops calling tools.

use crate::channels::Channels;
use crate::config::Config;
use crate::llm::{ChatError, ChatOptions, Llm, Reply};
use crate::session::{Message, Session, ToolCall, message_tokens, shadow_count};
use crate::tg::{
    Attachment, Event, EventKind, MessageReaction, Telegram, WatchList, render_reactions, timestamp,
};
use crate::tools::Tools;
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, mpsc};

/// Effort levels accepted by Z.ai `reasoning_effort`. GLM-5.3* only honors
/// `low`/`high`/`max`; the rest exist for older models — we pass through and
/// let the server validate.
const EFFORTS: [&str; 7] = ["max", "xhigh", "high", "medium", "low", "minimal", "none"];

pub const COMPACTION_INSTRUCTION: &str = "\
Compaction/summarization triggered: you must condense the conversation above into a structured checkpoint that lets you resume your work/the story with no essential context.

You still should have all the tools available so that you can record important information into your working directory. At the end of your turn, you must provide a single message that entails all the information that needs to be passed down to the new agent (you with no memory besides the system prompt), so that it can pick up from where you left. New agent will be provided with the same system prompt, the same environment (e. g. filesystem) and the message, all the other information will get lost.

Don't send any Telegram messages unless necessary.
Be as detailed as possible, proactively link files that might be needed for the new agent.";

const HEARTBEAT: &str = "\
[tick] Scheduled wake, no new Telegram messages. You may take autonomous actions (advance a scene, act for NPCs, tidy up your notes) or do nothing at all. You probably shouldn't overthink and just do nothing at all.";

/// Model + reasoning effort currently driving the main loop. Overridable at
/// runtime via `/model`, persisted in the state dir across restarts.
#[derive(Debug, Clone)]
pub struct ModelChoice {
    pub model: String,
    pub effort: Option<String>,
}

impl ModelChoice {
    fn load(state_dir: &Path, cfg_model: &str, cfg_effort: Option<&str>) -> Self {
        let path = state_dir.join("model.json");
        let persisted: Option<PersistedChoice> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        persisted
            .map(|p| Self {
                model: p.model,
                effort: p.effort,
            })
            .unwrap_or_else(|| Self {
                model: cfg_model.to_string(),
                effort: cfg_effort.map(str::to_string),
            })
    }

    fn save(&self, state_dir: &Path) {
        let path = state_dir.join("model.json");
        if let Ok(line) = serde_json::to_string(&PersistedChoice {
            model: self.model.clone(),
            effort: self.effort.clone(),
        }) && let Err(e) = std::fs::write(&path, line)
        {
            tracing::warn!(error = %e, "could not persist model choice");
        }
    }

    fn describe(&self) -> String {
        format!(
            "Модель: {}, effort: {}",
            self.model,
            self.effort
                .as_deref()
                .unwrap_or("(по умолчанию провайдера)")
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedChoice {
    model: String,
    effort: Option<String>,
}

/// `/model` command outcome, parsed from an update's text.
#[derive(Debug, PartialEq, Eq)]
enum ModelCommand {
    Status,
    Set {
        model: Option<String>,
        effort: Option<String>,
    },
}

fn parse_model_command(text: &str) -> Result<ModelCommand, String> {
    let mut parts = text.split_whitespace();
    let cmd = parts.next().unwrap_or_default();
    // Accept the /model@bot_name mention form Telegram uses in groups.
    if cmd.split('@').next() != Some("/model") {
        return Err("не команда /model".into());
    }
    let usage =
        "Использование: /model — показать; /model <модель>[:low|high|max] — сменить.".to_string();
    match parts.next() {
        None => Ok(ModelCommand::Status),
        Some(spec) => {
            if parts.next().is_some() {
                return Err(usage);
            }
            match spec.split_once(':') {
                Some((model, effort)) => {
                    if !model.is_empty() && !valid_model_name(model) {
                        return Err(format!("Неверное имя модели «{model}»."));
                    }
                    if !EFFORTS.contains(&effort) {
                        return Err(format!(
                            "Неизвестный effort «{effort}». Допустимо: {}.",
                            EFFORTS.join(", ")
                        ));
                    }
                    Ok(ModelCommand::Set {
                        model: (!model.is_empty()).then(|| model.to_string()),
                        effort: Some(effort.to_string()),
                    })
                }
                None => {
                    if !valid_model_name(spec) {
                        return Err(format!("Неверное имя модели «{spec}»."));
                    }
                    Ok(ModelCommand::Set {
                        model: Some(spec.to_string()),
                        effort: None,
                    })
                }
            }
        }
    }
}

fn valid_model_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

fn is_model_command(event: &Event) -> bool {
    let EventKind::Message { msg, .. } = &event.kind else {
        return false;
    };
    msg.text
        .as_deref()
        .map(|t| t.split_whitespace().next().map(|c| c.split('@').next()) == Some(Some("/model")))
        .unwrap_or(false)
}

pub struct Inbox {
    pub rx: mpsc::UnboundedReceiver<crate::tg::Update>,
    pub notify: Arc<Notify>,
}

pub struct Agent {
    cfg: Config,
    llm: Llm,
    tg: Telegram,
    session: Session,
    inbox: Inbox,
    watch: WatchList,
    channels: Channels,
    schemas: Value,
    choice: ModelChoice,
    last_wake: Instant,
    /// Compaction metric anchor: the provider's real `prompt_tokens` for the
    /// surface as of length `usize`; the metric adds an estimate of the tail
    /// appended since. `None` before the first response (and right after a
    /// compaction) falls back to the pure estimate.
    prompt_anchor: Option<(usize, u64)>,
}

impl Agent {
    pub fn new(
        cfg: Config,
        llm: Llm,
        tg: Telegram,
        session: Session,
        inbox: Inbox,
        watch: WatchList,
        channels: Channels,
    ) -> Self {
        let schemas = crate::tools::schemas();
        let choice = ModelChoice::load(
            &cfg.agent.state_dir,
            &cfg.llm.model,
            cfg.llm.reasoning_effort.as_deref(),
        );
        tracing::info!(model = %choice.model, effort = ?choice.effort, "model choice");
        let mut agent = Self {
            cfg,
            llm,
            tg,
            session,
            inbox,
            watch,
            channels,
            schemas,
            choice,
            last_wake: Instant::now(),
            prompt_anchor: None,
        };
        agent.seed_channels();
        agent
    }

    /// Freeze the channel list into the session head: byte-stable within a
    /// session generation (prefix-cache friendly), re-scanned at startup and
    /// after every compaction. Resolution in tools always uses the live
    /// registry, so channels discovered mid-session are addressable at once.
    fn seed_channels(&mut self) {
        self.session.set_head(Message::user(format!(
            "[Telegram channels] Address send_message/edit_message by these exact names. \
             Every channel's history is appended to workspace/chat/<name>.log — grep it for context.\n\n{}",
            self.channels.list()
        )));
    }

    pub async fn run(mut self) -> Result<()> {
        loop {
            self.wait_for_wake().await;
            let asleep = self.last_wake.elapsed();
            self.last_wake = Instant::now();
            let events = self.collect_events().await;

            // Control-plane commands never reach the model's context.
            let mut roleplay = Vec::new();
            for event in events {
                if is_model_command(&event) {
                    self.handle_model_command(event).await;
                } else {
                    roleplay.push(event);
                }
            }

            let (text, images) = format_events(&roleplay, asleep, &self.channels);
            let wake = match text {
                Some(text) => Message::user_with_images(text, images),
                None if self.cfg.agent.heartbeat => Message::user(HEARTBEAT),
                None => continue,
            };
            self.session.append(wake)?;
            if let Err(e) = self.run_turn().await {
                tracing::error!(error = %e, "turn failed; will retry on next wake");
            }
        }
    }

    async fn handle_model_command(&mut self, event: Event) {
        let EventKind::Message { msg, .. } = &event.kind else {
            return;
        };
        let reply = match parse_model_command(msg.text.as_deref().unwrap_or_default()) {
            Ok(ModelCommand::Status) => self.choice.describe(),
            Ok(ModelCommand::Set { model, effort }) => {
                if let Some(model) = model {
                    self.choice.model = model;
                }
                if let Some(effort) = effort {
                    self.choice.effort = Some(effort);
                }
                self.choice.save(&self.cfg.agent.state_dir);
                tracing::info!(model = %self.choice.model, effort = ?self.choice.effort, "model switched");
                self.choice.describe()
            }
            Err(usage) => usage,
        };
        match self
            .tg
            .send_message(
                msg.chat.id,
                msg.message_thread_id,
                &reply,
                None,
                Some(msg.message_id),
            )
            .await
        {
            Ok(id) => {
                self.channels
                    .log_sent((msg.chat.id, msg.message_thread_id), id, &reply);
            }
            Err(e) => tracing::warn!(error = %e, "model command reply failed"),
        }
    }

    async fn wait_for_wake(&mut self) {
        loop {
            if !self.inbox.rx.is_empty() {
                return;
            }
            let notified = self.inbox.notify.notified();
            let idle = Duration::from_secs(self.cfg.agent.idle_wake_secs);
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(idle) => return,
            }
        }
    }

    /// Drain the buffered updates; already filtered at ingress, re-checked
    /// here as defense in depth. Every passing update feeds the channel
    /// registry and its log.
    async fn collect_events(&mut self) -> Vec<Event> {
        let downloads = PathBuf::from(&self.cfg.agent.workspace_dir).join("downloads");
        let mut events = Vec::new();
        while let Ok(update) = self.inbox.rx.try_recv() {
            if let Some(reaction) = &update.message_reaction {
                let allowed = match (&reaction.user, &reaction.actor_chat) {
                    (Some(u), _) => !u.is_bot && self.watch.allowed_users.contains(&u.id),
                    (None, Some(c)) => self.watch.allowed_users.contains(&c.id),
                    (None, None) => false,
                };
                if !allowed {
                    tracing::debug!("dropped disallowed reaction");
                    continue;
                }
                self.channels.observe_reaction(reaction);
                events.push(Event {
                    kind: EventKind::Reaction(reaction.clone()),
                });
                continue;
            }
            let Some(msg) = update.msg() else { continue };
            let allowed = msg
                .from
                .as_ref()
                .map(|u| !u.is_bot && self.watch.allowed_users.contains(&u.id))
                .unwrap_or(true);
            if !allowed {
                tracing::debug!(user = ?msg.from.as_ref().map(|u| u.id), "dropped disallowed update");
                continue;
            }
            let attachments = match self.tg.fetch_attachments(msg, &downloads).await {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!(error = %e, "attachment download failed");
                    vec![]
                }
            };
            self.channels.observe_message(msg, update.edited());
            events.push(Event {
                kind: EventKind::Message {
                    msg: msg.clone(),
                    edited: update.edited(),
                    attachments,
                },
            });
        }
        events
    }

    fn chat_options(&self) -> ChatOptions<'_> {
        ChatOptions {
            model: &self.choice.model,
            max_tokens: self.cfg.llm.max_output_tokens,
            temperature: self.cfg.llm.temperature,
            effort: self.choice.effort.as_deref(),
            preserve_thinking: self.cfg.llm.preserve_thinking,
        }
    }

    fn tools(&self) -> Tools<'_> {
        Tools {
            tg: Some(&self.tg),
            channels: &self.channels,
            workspace: &self.cfg.agent.workspace_dir,
        }
    }

    async fn run_turn(&mut self) -> Result<()> {
        loop {
            let reply = match self.request().await {
                Ok(reply) => reply,
                Err(ChatError::ContextOverflow(e)) => {
                    tracing::warn!(error = %e, "context overflow; compacting aggressively and retrying once");
                    self.maybe_compact(true).await?;
                    self.request()
                        .await
                        .map_err(|e| anyhow!("llm request failed: {e}"))?
                }
                Err(e) => return Err(anyhow!("llm request failed: {e}")),
            };
            if reply.finish_reason.as_deref() == Some("length") {
                tracing::warn!("reply hit max_tokens; consider raising llm.max_output_tokens");
            }
            let mut assistant =
                Message::assistant_with_calls(reply.text.clone(), reply.tool_calls.clone());
            if self.cfg.llm.preserve_thinking {
                assistant = assistant.with_reasoning(reply.reasoning);
            }
            self.session.append(assistant)?;
            if reply.tool_calls.is_empty() {
                // Turn over and nobody is waiting: compact here, in the
                // pause, so the next user message never pays for it. The
                // real prompt_tokens of the last request are already
                // anchored.
                self.maybe_compact(false).await?;
                return Ok(());
            }
            self.dispatch(&reply.tool_calls).await?;
        }
    }

    /// The full wire surface: the system prompt from config (never stored in
    /// the session, so a restart applies a new prompt immediately), the
    /// frozen channel list, then the session content.
    fn composed(&self) -> Vec<Message> {
        let mut msgs = Vec::with_capacity(2 + self.session.surface().len());
        msgs.push(self.system_message());
        msgs.extend(self.session.head().cloned());
        msgs.extend_from_slice(self.session.surface());
        msgs
    }

    fn system_message(&self) -> Message {
        Message::system(self.cfg.agent.system_prompt.as_deref().unwrap_or_default())
    }

    /// One model request. On success anchors the compaction metric to the
    /// provider's own token count for this exact surface state.
    async fn request(&mut self) -> Result<Reply, ChatError> {
        let mark = self.session.surface().len();
        let reply = self
            .llm
            .chat(&self.composed(), &self.schemas, &self.chat_options())
            .await?;
        if let Some(p) = reply.prompt_tokens {
            self.prompt_anchor = Some((mark, p));
        }
        Ok(reply)
    }

    async fn dispatch(&mut self, calls: &[ToolCall]) -> Result<()> {
        for call in calls {
            let started = Instant::now();
            let out = match crate::tools::parse_arguments(&call.arguments) {
                Ok(args) => self.tools().execute(&call.name, args).await,
                Err(e) => format!("Error: {e}"),
            };
            tracing::info!(tool = %call.name, ms = started.elapsed().as_millis() as u64, "tool finished");
            self.session
                .append(Message::tool_result(call.id.clone(), out))?;
        }
        Ok(())
    }

    /// Compact when the token total crosses the threshold. The total is the
    /// provider's real `prompt_tokens` from the last request plus an estimate
    /// of the tail appended since; before any response (and right after a
    /// compaction) it is a pure estimate. With `force` (context-overflow
    /// recovery) the retained tail shrinks and the threshold is bypassed.
    /// Compaction failure is logged, never fatal: the turn continues with the
    /// uncompacted surface.
    async fn maybe_compact(&mut self, force: bool) -> Result<()> {
        let total = match self.prompt_anchor {
            Some((mark, prompt)) => prompt + self.session.tail_tokens(mark),
            None => {
                self.session.tail_tokens(0)
                    + message_tokens(&self.system_message())
                    + self.session.head().map(message_tokens).unwrap_or(0)
            }
        };
        let threshold = self.cfg.llm.compaction_threshold_tokens;
        if !force && total < threshold {
            return Ok(());
        }
        let retain = if force {
            self.cfg.llm.compaction_retain_tokens.min(2_000)
        } else {
            self.cfg.llm.compaction_retain_tokens
        };
        let Some(shadowed) = shadow_count(self.session.surface(), retain).filter(|n| *n > 0) else {
            return Ok(());
        };
        let shadowed_tokens: u64 = self.session.surface()[1..1 + shadowed]
            .iter()
            .map(message_tokens)
            .sum();
        tracing::info!(total, threshold, shadowed, "compacting");

        let summary = match self.summarize().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "summarization failed; skipping compaction");
                return Ok(());
            }
        };
        let summary_tokens = message_tokens(&Message::user(summary.clone()));
        if summary_tokens >= shadowed_tokens {
            tracing::warn!(
                summary_tokens,
                shadowed_tokens,
                "summary is not smaller; skipping compaction"
            );
            return Ok(());
        }
        self.session.compact(shadowed, summary)?;
        // A fresh session generation: re-scan the channel list, and the
        // anchor described the old surface anyway.
        self.seed_channels();
        self.prompt_anchor = None;
        tracing::info!(
            shadowed,
            shadowed_tokens,
            summary_tokens,
            "compaction committed"
        );
        Ok(())
    }

    /// Run the summarizer as a normal agentic loop over a scratch copy of
    /// the surface. All tools stay available, including Telegram's. The
    /// request replays the surface verbatim, so the provider prefix cache
    /// stays warm and only the instruction is novel.
    async fn summarize(&self) -> Result<String> {
        let mut scratch = self.composed();
        scratch.push(Message::user(COMPACTION_INSTRUCTION));
        loop {
            let reply = self
                .llm
                .chat(&scratch, &self.schemas, &self.chat_options())
                .await
                .map_err(|e| anyhow!("summarizer request failed: {e}"))?;
            let calls = reply.tool_calls.clone();
            let text = reply.text.clone();
            let mut assistant = Message::assistant_with_calls(reply.text, calls.clone());
            if self.cfg.llm.preserve_thinking {
                assistant = assistant.with_reasoning(reply.reasoning);
            }
            scratch.push(assistant);
            if calls.is_empty() {
                return if text.trim().is_empty() {
                    Err(anyhow!("summarizer produced an empty checkpoint"))
                } else {
                    Ok(text)
                };
            }
            for call in calls {
                let out = match crate::tools::parse_arguments(&call.arguments) {
                    Ok(args) => self.tools().execute(&call.name, args).await,
                    Err(e) => format!("Error: {e}"),
                };
                scratch.push(Message::tool_result(call.id, out));
            }
        }
    }
}

fn format_user(u: &crate::tg::TgUser) -> String {
    match &u.username {
        Some(handle) => format!("{} (@{handle})", u.display_name()),
        None => u.display_name(),
    }
}

fn describe_event(e: &Event, channels: &Channels) -> String {
    match &e.kind {
        EventKind::Message {
            msg,
            edited,
            attachments,
        } => describe_message(msg, *edited, attachments, channels),
        EventKind::Reaction(r) => describe_reaction(r, channels),
    }
}

fn describe_message(
    m: &crate::tg::TgMessage,
    edited: bool,
    attachments: &[Attachment],
    channels: &Channels,
) -> String {
    let time = timestamp(m.date);
    let author = match &m.from {
        Some(u) => format_user(u),
        None => "service".to_string(),
    };
    let channel = channels.name_of((m.chat.id, m.message_thread_id));
    let reply_note = match m.reply_to_message.as_ref().map(|r| r.message_id) {
        Some(r) => format!(", reply to #{r}"),
        None => String::new(),
    };
    let edited_note = if edited { " (edited)" } else { "" };
    let mut out = format!(
        "--- [{time}] {author} in {channel}, message #{}{edited_note}{reply_note}\n",
        m.message_id
    );
    let body = m
        .text
        .clone()
        .or_else(|| m.caption.clone())
        .unwrap_or_else(|| "(no text)".into());
    out.push_str(&body);
    out.push('\n');
    for a in attachments {
        let kind = if a.is_image { "image" } else { "file" };
        let size = std::fs::metadata(&a.path).map(|md| md.len()).unwrap_or(0);
        out.push_str(&format!(
            "[attachment: {kind} saved to {} ({size} bytes)]\n",
            a.path.display()
        ));
    }
    out
}

fn describe_reaction(r: &MessageReaction, channels: &Channels) -> String {
    let actor = match &r.user {
        Some(u) => format_user(u),
        None => r
            .actor_chat
            .as_ref()
            .map(|c| c.display_name())
            .unwrap_or_else(|| "someone".into()),
    };
    let action = if r.new_reaction.is_empty() {
        "removed reaction on"
    } else {
        "reacted to"
    };
    format!(
        "--- [{}] {actor} {action} message #{} in {}: {} (was: {})\n",
        timestamp(r.date),
        r.message_id,
        channels.name_of((r.chat.id, None)),
        render_reactions(&r.new_reaction),
        render_reactions(&r.old_reaction),
    )
}

/// Render drained events as one user message: a single text block plus the
/// downloaded images as vision blocks. `None` when nothing arrived. The
/// header names the source channels so the agent sees at a glance which
/// channel each burst came from.
fn format_events(
    events: &[Event],
    asleep: Duration,
    channels: &Channels,
) -> (Option<String>, Vec<String>) {
    if events.is_empty() {
        return (None, vec![]);
    }
    let mut sources: Vec<String> = events
        .iter()
        .map(|e| channels.name_of(e.channel_key()))
        .collect();
    sources.sort();
    sources.dedup();
    let mut text = format!(
        "[Telegram: {} new update(s) from {}; you were asleep for ~{}s]\n\n",
        events.len(),
        sources.join(", "),
        asleep.as_secs()
    );
    let mut images = Vec::new();
    for e in events {
        text.push_str(&describe_event(e, channels));
        text.push('\n');
        let EventKind::Message { attachments, .. } = &e.kind else {
            continue;
        };
        for attachment in attachments {
            if image_mime(&attachment.path).is_none() {
                continue;
            }
            match image_data_url(attachment) {
                Some(url) => images.push(url),
                None => text.push_str(&format!(
                    "[could not load image {}]\n",
                    attachment.path.display()
                )),
            }
        }
    }
    (Some(text), images)
}

fn image_mime(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

/// Embed an attachment as a vision data URL; `None` for non-image files
/// (they stay file-tool-only) or unreadable ones.
fn image_data_url(attachment: &Attachment) -> Option<String> {
    let mime = image_mime(&attachment.path)?;
    let bytes = std::fs::read(&attachment.path).ok()?;
    Some(Telegram::image_data_url(&bytes, mime))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(text: &str) -> Result<ModelCommand, String> {
        parse_model_command(text)
    }

    fn event_with_text(text: &str) -> Event {
        Event {
            kind: EventKind::Message {
                msg: crate::tg::TgMessage {
                    message_id: 1,
                    from: None,
                    chat: crate::tg::TgChat {
                        id: -100,
                        kind: "supergroup".into(),
                        title: None,
                        first_name: String::new(),
                        last_name: String::new(),
                        username: None,
                    },
                    message_thread_id: Some(42),
                    date: 0,
                    text: Some(text.into()),
                    caption: None,
                    photo: vec![],
                    document: None,
                    reply_to_message: None,
                    forum_topic_created: None,
                    forum_topic_edited: None,
                },
                edited: false,
                attachments: vec![],
            },
        }
    }

    fn reaction_event(chat_title: Option<&str>, emoji: &str) -> Event {
        Event {
            kind: EventKind::Reaction(MessageReaction {
                chat: crate::tg::TgChat {
                    id: -100,
                    kind: "supergroup".into(),
                    title: chat_title.map(str::to_string),
                    first_name: String::new(),
                    last_name: String::new(),
                    username: None,
                },
                message_id: 7,
                user: Some(crate::tg::TgUser {
                    id: 1,
                    is_bot: false,
                    first_name: "Alice".into(),
                    last_name: String::new(),
                    username: Some("alice".into()),
                }),
                actor_chat: None,
                date: 0,
                old_reaction: vec![],
                new_reaction: vec![crate::tg::ReactionType {
                    kind: "emoji".into(),
                    emoji: Some(emoji.into()),
                }],
            }),
        }
    }

    fn channels() -> Channels {
        let dir = std::env::temp_dir().join("rp-agent-channel-tests");
        std::fs::create_dir_all(&dir).unwrap();
        Channels::empty(&dir.join("state"), &dir)
    }

    #[test]
    fn model_command_parsing() {
        assert_eq!(command("/model"), Ok(ModelCommand::Status));
        assert_eq!(command("/model@rp_bot"), Ok(ModelCommand::Status));
        assert_eq!(
            command("/model glm-5.3"),
            Ok(ModelCommand::Set {
                model: Some("glm-5.3".into()),
                effort: None
            })
        );
        assert_eq!(
            command("/model glm-5.3-flash:high"),
            Ok(ModelCommand::Set {
                model: Some("glm-5.3-flash".into()),
                effort: Some("high".into())
            })
        );
        assert_eq!(
            command("/model glm-5.3:max"),
            Ok(ModelCommand::Set {
                model: Some("glm-5.3".into()),
                effort: Some("max".into())
            })
        );
        // Bare effort change keeps the model.
        assert_eq!(
            command("/model :low"),
            Ok(ModelCommand::Set {
                model: None,
                effort: Some("low".into())
            })
        );
        assert!(command("/model glm 5.3").is_err());
        assert!(command("/model glm-5.3:ultra").is_err());
        assert!(command("/model ../etc:low").is_err());
    }

    #[test]
    fn model_command_detection() {
        assert!(is_model_command(&event_with_text("/model glm-5.3")));
        assert!(is_model_command(&event_with_text("/model@rp_bot :low")));
        assert!(!is_model_command(&event_with_text("/modelx low")));
        assert!(!is_model_command(&event_with_text("story text /model")));
        assert!(!is_model_command(&event_with_text("/ping")));
    }

    #[test]
    fn image_embedding_is_extension_gated() {
        let dir = std::env::temp_dir().join("rp-agent-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("pic.PNG");
        std::fs::write(&png, b"not really png but bytes").unwrap();
        let txt = dir.join("note.txt");
        std::fs::write(&txt, b"plain text").unwrap();

        let att = |p: &std::path::Path| Attachment {
            path: p.to_path_buf(),
            is_image: false,
        };
        assert!(
            image_data_url(&att(&png))
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(image_data_url(&att(&txt)), None);
    }

    #[test]
    fn header_names_source_channels() {
        let mut channels = channels();
        let a = event_with_text("one");
        let EventKind::Message { msg, .. } = &a.kind else {
            unreachable!()
        };
        channels.observe_message(msg, false);
        let b = reaction_event(Some("News"), "🔥");
        let EventKind::Reaction(r) = &b.kind else {
            unreachable!()
        };
        channels.observe_reaction(r);
        let (text, _) = format_events(&[a, b], Duration::from_secs(30), &channels);
        let text = text.unwrap();
        assert!(
            text.starts_with("[Telegram: 2 new update(s) from News, topic 42;"),
            "{text}"
        );
    }

    #[test]
    fn events_are_described_with_channel_names() {
        let dir = std::env::temp_dir().join("rp-agent-channel-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let mut channels = Channels::empty(&dir.join("state"), &dir);
        let text = describe_event(&event_with_text("scene"), &channels);
        assert!(text.contains("in topic 42, message #1"), "{text}");
        assert!(!text.contains("chat id"), "{text}");
        assert!(!text.contains("user id"), "{text}");
        // A DM resolves to the person's name, not a bare id.
        let mut dm = event_with_text("hi");
        let EventKind::Message { msg, .. } = &mut dm.kind else {
            unreachable!()
        };
        msg.chat = crate::tg::TgChat {
            id: 5,
            kind: "private".into(),
            title: None,
            first_name: "Luna".into(),
            last_name: "Spirito".into(),
            username: None,
        };
        msg.message_thread_id = None;
        channels.observe_message(msg, false);
        let text = describe_event(&dm, &channels);
        assert!(text.contains("in Luna Spirito, message #1"), "{text}");
    }

    #[test]
    fn reactions_are_described_for_the_model() {
        let mut channels = channels();
        let reaction = reaction_event(Some("News"), "🔥");
        let EventKind::Reaction(r) = &reaction.kind else {
            unreachable!()
        };
        channels.observe_reaction(r);
        let text = describe_event(&reaction, &channels);
        assert!(
            text.contains("Alice (@alice) reacted to message #7 in News"),
            "{text}"
        );
        assert!(text.contains("🔥 (was: (none))"), "{text}");
        // Empty new_reaction reads as removal.
        let removed = Event {
            kind: EventKind::Reaction(MessageReaction {
                chat: crate::tg::TgChat {
                    id: 1,
                    kind: "private".into(),
                    title: None,
                    first_name: String::new(),
                    last_name: String::new(),
                    username: None,
                },
                message_id: 7,
                user: None,
                actor_chat: None,
                date: 0,
                old_reaction: vec![crate::tg::ReactionType {
                    kind: "emoji".into(),
                    emoji: Some("👍".into()),
                }],
                new_reaction: vec![],
            }),
        };
        let text = describe_event(&removed, &channels);
        assert!(text.contains("removed reaction on message #7"), "{text}");
        assert!(text.contains("(was: 👍)"), "{text}");
    }
}
