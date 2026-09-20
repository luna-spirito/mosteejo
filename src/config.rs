//! TOML configuration. Secrets (tokens) may live in the environment instead
//! of the file; the system prompt may be inline or in a file.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn default_api_url() -> String {
    "https://api.telegram.org".into()
}

fn default_base_url() -> String {
    "https://api.z.ai/api/paas/v4".into()
}

fn default_model() -> String {
    "glm-5.3-flash".into()
}

fn default_max_output() -> u32 {
    8192
}

fn default_threshold() -> u64 {
    48_000
}

fn default_retain() -> u64 {
    8_000
}

fn default_workspace() -> PathBuf {
    "workspace".into()
}

fn default_state() -> PathBuf {
    "state".into()
}

fn default_idle() -> u64 {
    600
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub telegram: TelegramCfg,
    pub llm: LlmCfg,
    pub agent: AgentCfg,
}

#[derive(Deserialize, Clone)]
pub struct TelegramCfg {
    pub token: Option<String>,
    /// Override to talk to a local Bot API server or a test stub.
    #[serde(default = "default_api_url")]
    pub api_url: String,
    pub bot_username: Option<String>,
    #[serde(default)]
    pub allowed_users: Vec<AllowedUser>,
    #[serde(default)]
    pub subscribed_topics: Vec<TopicCfg>,
}

#[derive(Deserialize, Clone)]
pub struct AllowedUser {
    pub id: i64,
    pub name: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct TopicCfg {
    pub chat_id: i64,
    pub thread_id: i64,
}

#[derive(Deserialize, Clone)]
pub struct LlmCfg {
    pub api_key: Option<String>,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_output")]
    pub max_output_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// `reasoning_effort` for Z.ai: low | high | max on GLM-5.3* (default of
    /// the provider is `max`). `None` sends nothing and keeps API defaults.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Preserved thinking: store `reasoning_content` of assistant messages
    /// and replay it verbatim (`clear_thinking: false`). Recommended for
    /// tool-using agents; raises token usage between compactions.
    #[serde(default = "default_true")]
    pub preserve_thinking: bool,
    #[serde(default = "default_threshold")]
    pub compaction_threshold_tokens: u64,
    #[serde(default = "default_retain")]
    pub compaction_retain_tokens: u64,
}

#[derive(Deserialize, Clone)]
pub struct AgentCfg {
    pub system_prompt: Option<String>,
    pub system_prompt_file: Option<PathBuf>,
    #[serde(default = "default_workspace")]
    pub workspace_dir: PathBuf,
    #[serde(default = "default_state")]
    pub state_dir: PathBuf,
    #[serde(default = "default_idle")]
    pub idle_wake_secs: u64,
    #[serde(default = "default_true")]
    pub heartbeat: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&raw).context("parse config")?;

        if cfg.telegram.token.is_none() {
            cfg.telegram.token = std::env::var("TELEGRAM_BOT_TOKEN").ok();
        }
        anyhow::ensure!(
            cfg.telegram.token.is_some(),
            "telegram token missing: set `telegram.token` or $TELEGRAM_BOT_TOKEN"
        );
        if cfg.llm.api_key.is_none() {
            cfg.llm.api_key = std::env::var("ZAI_API_KEY").ok();
        }
        anyhow::ensure!(
            cfg.llm.api_key.is_some(),
            "llm api key missing: set `llm.api_key` or $ZAI_API_KEY"
        );

        if let Some(file) = &cfg.agent.system_prompt_file {
            let prompt = std::fs::read_to_string(file)
                .with_context(|| format!("read system prompt {}", file.display()))?;
            cfg.agent.system_prompt = Some(prompt);
        }
        anyhow::ensure!(
            cfg.agent.system_prompt.is_some(),
            "system prompt missing: set `agent.system_prompt` or `agent.system_prompt_file`"
        );
        Ok(cfg)
    }
}
