//! Entry point: load config, verify Telegram identity, seed the system
//! prompt, start the long-poll poller task and hand control to the agent.

mod agent;
mod config;
mod llm;
mod session;
mod tg;
mod tools;

use crate::agent::{Agent, Inbox};
use crate::config::Config;
use crate::llm::Llm;
use crate::session::{Message, Session};
use crate::tg::{Telegram, Update, Verdict, WatchList};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cfg_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rp.toml".to_string());
    let cfg = Config::load(std::path::Path::new(&cfg_path))?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cfg))
}

async fn run(cfg: Config) -> Result<()> {
    let tg = Telegram::new(&cfg.telegram.api_url, cfg.telegram.token.as_deref().expect("resolved by config"))?;
    let me = tg.get_me().await.context("getMe")?;
    let bot_username = me
        .username
        .clone()
        .or_else(|| cfg.telegram.bot_username.clone())
        .context("bot username unknown: set `telegram.bot_username`")?;
    tracing::info!(id = me.id, username = %bot_username, "telegram ready");

    std::fs::create_dir_all(&cfg.agent.workspace_dir).context("create workspace dir")?;
    std::fs::create_dir_all(&cfg.agent.state_dir).context("create state dir")?;
    let offset_path = PathBuf::from(&cfg.agent.state_dir).join("tg_offset");

    let mut session = Session::load(&PathBuf::from(&cfg.agent.state_dir).join("session.jsonl"))?;
    if session.surface().is_empty() {
        session.append(Message::system(cfg.agent.system_prompt.clone().expect("resolved by config")))?;
    }

    let watch = WatchList {
        allowed_users: cfg.telegram.allowed_users.iter().map(|u| u.id).collect(),
        subscribed_topics: cfg
            .telegram
            .subscribed_topics
            .iter()
            .map(|t| (t.chat_id, t.thread_id))
            .collect(),
        bot_id: me.id,
        bot_username: bot_username.clone(),
    };
    let allowed = watch
        .allowed_users
        .iter()
        .map(|id| {
            cfg.telegram
                .allowed_users
                .iter()
                .find(|u| u.id == *id)
                .and_then(|u| u.name.clone())
                .map(|n| format!("{n} (#{id})"))
                .unwrap_or_else(|| format!("#{id}"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!(users = %allowed, "ingress allowlist");

    let (tx, rx) = mpsc::unbounded_channel();
    let notify = Arc::new(Notify::new());
    tokio::spawn(poller(
        tg.clone(),
        watch.clone(),
        tx,
        notify.clone(),
        offset_path,
    ));

    let llm = Llm::new(&cfg.llm.base_url, cfg.llm.api_key.clone().expect("resolved by config"))?;
    Agent::new(cfg, llm, tg, session, Inbox { rx, notify }, watch)
        .run()
        .await
}

/// Long-poll loop: applies the strict ingress filter and feeds the agent's
/// inbox. Priority updates signal an immediate wake via `notify`.
async fn poller(
    tg: Telegram,
    watch: WatchList,
    tx: mpsc::UnboundedSender<Update>,
    notify: Arc<Notify>,
    offset_path: PathBuf,
) {
    let mut offset: i64 = std::fs::read_to_string(&offset_path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    tracing::info!(offset, "poller started");
    loop {
        match tg.get_updates((offset > 0).then_some(offset + 1)).await {
            Ok(updates) => {
                for update in updates {
                    offset = offset.max(update.update_id);
                    match tg::classify(&update, &watch) {
                        Verdict::Drop => {
                            tracing::debug!(update = update.update_id, "dropped by ingress filter")
                        }
                        verdict => {
                            let wake = verdict == Verdict::Wake;
                            if tx.send(update).is_err() {
                                return; // agent is gone; shutting down
                            }
                            if wake {
                                notify.notify_one();
                            }
                        }
                    }
                }
                if let Err(e) = std::fs::write(&offset_path, offset.to_string()) {
                    tracing::warn!(error = %e, "could not persist telegram offset");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "getUpdates failed; retrying in 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}
