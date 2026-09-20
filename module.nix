# NixOS module: rp-bot as a hardened systemd service.
#
# Configuration is rendered into rp.toml from module options — it holds no
# secrets and lands in the world-readable store. Secrets are passed as
# systemd credentials (`services.rp-bot.credentials`, feed it
# `config.age.secrets.*.path`, `config.sops.secrets.*.path` or plain files);
# the unit mounts them via LoadCredential and the bot reads them from
# $CREDENTIALS_DIRECTORY (names: `telegram_bot_token`, `zai_api_key`).
{ config, lib, pkgs, ... }:
let
  cfg = config.services.rp-bot;
  tomlFormat = pkgs.formats.toml { };

  dropNull = lib.filterAttrs (_: v: v != null);
  settings = {
    telegram = dropNull {
      api_url = cfg.telegram.apiUrl;
      bot_username = cfg.telegram.botUsername;
      allowed_users = map (u: dropNull { id = u.id; name = u.name; }) cfg.telegram.allowedUsers;
      subscribed_topics = map (t: {
        chat_id = t.chatId;
        thread_id = t.threadId;
      }) cfg.telegram.subscribedTopics;
    };
    llm = dropNull {
      base_url = cfg.llm.baseUrl;
      model = cfg.llm.model;
      max_output_tokens = cfg.llm.maxOutputTokens;
      temperature = cfg.llm.temperature;
      reasoning_effort = cfg.llm.reasoningEffort;
      preserve_thinking = cfg.llm.preserveThinking;
      compaction_threshold_tokens = cfg.llm.compactionThresholdTokens;
      compaction_retain_tokens = cfg.llm.compactionRetainTokens;
    };
    agent = dropNull {
      system_prompt = cfg.agent.systemPrompt;
      system_prompt_file = if cfg.agent.promptFile == null then null else toString cfg.agent.promptFile;
      workspace_dir = cfg.agent.workspaceDir;
      state_dir = cfg.agent.stateDir;
      idle_wake_secs = cfg.agent.idleWakeSecs;
      heartbeat = cfg.agent.heartbeat;
    };
  };
  tomlFile = tomlFormat.generate "rp.toml" settings;
in
{
  options.services.rp-bot = {
    enable = lib.mkEnableOption "rp-bot, a Telegram group roleplay agent";

    package = lib.mkOption {
      type = lib.types.package;
      example = lib.literalExpression "rp-bot.packages.\${pkgs.system}.default";
      description = "The rp-bot package (built by the same flake via crane).";
    };

    configFile = lib.mkOption {
      type = lib.types.path;
      readOnly = true;
      description = "The rp.toml generated from the module options (secret-free).";
    };

    credentials = lib.mkOption {
      type = lib.types.attrsOf lib.types.path;
      default = { };
      example = {
        telegram_bot_token = "/run/secrets/telegram_bot_token";
        zai_api_key = "/run/secrets/zai_api_key";
      };
      description = ''
        Credential files exposed to the service via LoadCredential. Expected
        attribute names: `telegram_bot_token`, `zai_api_key`. Use
        `systemd-creds encrypt` / agenix / sops outputs here.
      '';
    };

    telegram = {
      apiUrl = lib.mkOption {
        type = lib.types.str;
        default = "https://api.telegram.org";
        description = "Bot API base URL (override for a local Bot API server).";
      };
      botUsername = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Bot username for @mention wake-ups; auto-detected from getMe when null.";
      };
      allowedUsers = lib.mkOption {
        type = lib.types.listOf (lib.types.submodule {
          options = {
            id = lib.mkOption { type = lib.types.int; description = "Telegram user id."; };
            name = lib.mkOption {
              type = lib.types.nullOr lib.types.str;
              default = null;
              description = "Display name, used in logs only.";
            };
          };
        });
        default = [ ];
        description = "Strict ingress allowlist: messages from anyone else are dropped unseen.";
      };
      subscribedTopics = lib.mkOption {
        type = lib.types.listOf (lib.types.submodule {
          options = {
            chatId = lib.mkOption { type = lib.types.int; description = "Supergroup id."; };
            threadId = lib.mkOption { type = lib.types.int; description = "Forum topic (message_thread_id)."; };
          };
        });
        default = [ ];
        description = "Topics the bot listens to and wakes on.";
      };
    };

    llm = {
      baseUrl = lib.mkOption {
        type = lib.types.str;
        default = "https://api.z.ai/api/paas/v4";
      };
      model = lib.mkOption {
        type = lib.types.str;
        default = "glm-5.3-flash";
        description = "Initial model; switchable at runtime via /model.";
      };
      maxOutputTokens = lib.mkOption {
        type = lib.types.int;
        default = 8192;
      };
      temperature = lib.mkOption {
        type = lib.types.nullOr lib.types.float;
        default = null;
      };
      reasoningEffort = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum [
          "max"
          "xhigh"
          "high"
          "medium"
          "low"
          "minimal"
          "none"
        ]);
        default = null;
        description = "Z.ai reasoning effort; null keeps the provider default.";
      };
      preserveThinking = lib.mkOption {
        type = lib.types.bool;
        default = true;
      };
      compactionThresholdTokens = lib.mkOption {
        type = lib.types.int;
        default = 48000;
      };
      compactionRetainTokens = lib.mkOption {
        type = lib.types.int;
        default = 8000;
      };
    };

    agent = {
      systemPrompt = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Roleplay system prompt (inline). Required unless promptFile is set.";
      };
      promptFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "File with the system prompt, read by the bot at startup.";
      };
      workspaceDir = lib.mkOption {
        type = lib.types.str;
        default = "/var/lib/rp-bot/workspace";
      };
      stateDir = lib.mkOption {
        type = lib.types.str;
        default = "/var/lib/rp-bot/state";
      };
      idleWakeSecs = lib.mkOption {
        type = lib.types.int;
        default = 600;
      };
      heartbeat = lib.mkOption {
        type = lib.types.bool;
        default = true;
      };
    };
  };

  config = lib.mkIf cfg.enable {
    services.rp-bot.configFile = tomlFile;

    systemd.services.rp-bot = {
      description = "Telegram RP roleplay bot";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment.RUST_LOG = lib.mkDefault "info";

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/rp-bot ${tomlFile}";
        WorkingDirectory = "/var/lib/rp-bot";
        StateDirectory = "rp-bot";
        LoadCredential = lib.mapAttrsToList (name: path: "${name}:${path}") cfg.credentials;
        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 5;
        UMask = "0077";
        # Hardening: outbound HTTPS and the state dir is all the bot needs;
        # the agent's own bash tool then runs inside the same sandbox.
        NoNewPrivileges = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        LockPersonality = true;
        RestrictAddressFamilies = [
          "AF_UNIX"
          "AF_INET"
          "AF_INET6"
        ];
        SystemCallFilter = "@system-service";
        CapabilityBoundingSet = "";
      };
    };
  };
}
