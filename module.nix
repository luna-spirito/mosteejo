# NixOS module: rp-bot instances as hardened systemd services.
#
# Multiple bots share one machine: each `services.rp-bots.instances.<name>`
# becomes its own `rp-bot-<name>` unit with its own rp.toml, credentials and
# state directories. The toml is rendered from module options — it holds no
# secrets and lands in the world-readable store. Secrets are passed as systemd
# credentials (`instances.<name>.credentials`, feed it
# `config.age.secrets.*.path`, `config.sops.secrets.*.path` or plain files);
# the unit mounts them via LoadCredential and the bot reads them from
# $CREDENTIALS_DIRECTORY (names: `telegram_bot_token`, `zai_api_key`).
{ config, lib, pkgs, ... }:
let
  cfg = config.services.rp-bots;
  tomlFormat = pkgs.formats.toml { };

  dropNull = lib.filterAttrs (_: v: v != null);

  # Parent directory of the workspace as a systemd StateDirectory value:
  # systemd creates/chowns /var/lib/<this> before the unit starts. null when
  # the workspace lives outside /var/lib — then no StateDirectory is set and
  # the directories must be provisioned some other way.
  stateDirectoryOf = workspaceDir:
    let parent = dirOf workspaceDir; in
    if lib.hasPrefix "/var/lib/" parent
    then lib.removePrefix "/var/lib/" parent
    else null;

  instanceOptions = { name, config, ... }: {
    options = {
      configFile = lib.mkOption {
        type = lib.types.path;
        readOnly = true;
        description = "The rp.toml generated from this instance's options (secret-free).";
      };

      credentials = lib.mkOption {
        type = lib.types.attrsOf lib.types.path;
        default = { };
        example = {
          telegram_bot_token = "/run/secrets/telegram_bot_token";
          zai_api_key = "/run/secrets/zai_api_key";
        };
        description = ''
          Credential files exposed to this instance's service via
          LoadCredential. Expected attribute names: `telegram_bot_token`,
          `zai_api_key`. Use `systemd-creds encrypt` / agenix / sops outputs
          here. Distinct bots need distinct token files.
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
          default = "/var/lib/rp-bot/${name}/workspace";
          description = "Agent workspace directory (notes, chat logs, downloads).";
        };
        stateDir = lib.mkOption {
          type = lib.types.str;
          default = "/var/lib/rp-bot/${name}/state";
          description = "Agent state directory (sessions, channels.json, tg_offset).";
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

    config = {
      configFile = tomlFormat.generate "rp-bot-${name}.toml" {
        telegram = dropNull {
          api_url = config.telegram.apiUrl;
          bot_username = config.telegram.botUsername;
          allowed_users = map (u: dropNull { id = u.id; name = u.name; }) config.telegram.allowedUsers;
          subscribed_topics = map (t: {
            chat_id = t.chatId;
            thread_id = t.threadId;
          }) config.telegram.subscribedTopics;
        };
        llm = dropNull {
          base_url = config.llm.baseUrl;
          model = config.llm.model;
          max_output_tokens = config.llm.maxOutputTokens;
          temperature = config.llm.temperature;
          reasoning_effort = config.llm.reasoningEffort;
          preserve_thinking = config.llm.preserveThinking;
          compaction_threshold_tokens = config.llm.compactionThresholdTokens;
          compaction_retain_tokens = config.llm.compactionRetainTokens;
        };
        agent = dropNull {
          system_prompt = config.agent.systemPrompt;
          system_prompt_file = if config.agent.promptFile == null then null else toString config.agent.promptFile;
          workspace_dir = config.agent.workspaceDir;
          state_dir = config.agent.stateDir;
          idle_wake_secs = config.agent.idleWakeSecs;
          heartbeat = config.agent.heartbeat;
        };
      };
    };
  };
in
{
  options.services.rp-bots = {
    package = lib.mkOption {
      type = lib.types.package;
      example = lib.literalExpression "rp-bot.packages.\${pkgs.system}.default";
      description = "The rp-bot package shared by all instances (built by the same flake via crane).";
    };

    instances = lib.mkOption {
      type = lib.types.attrsOf (lib.types.submodule instanceOptions);
      default = { };
      description = "rp-bot instances: each attribute spawns a `rp-bot-<name>` systemd service with its own config, credentials and state.";
    };
  };

  config =
    let
      mkService = name: icfg:
        let stateDirectory = stateDirectoryOf icfg.agent.workspaceDir; in
        {
          description = "Telegram RP roleplay bot (${name})";
          after = [ "network-online.target" ];
          wants = [ "network-online.target" ];
          wantedBy = [ "multi-user.target" ];

          environment.RUST_LOG = lib.mkDefault "info";

          serviceConfig = {
            ExecStart = "${cfg.package}/bin/rp-bot ${icfg.configFile}";
            WorkingDirectory = dirOf icfg.agent.workspaceDir;
            LoadCredential = lib.mapAttrsToList (cname: path: "${cname}:${path}") icfg.credentials;
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
          } // lib.optionalAttrs (stateDirectory != null) {
            StateDirectory = stateDirectory;
          };
        };
    in
    {
      systemd.services = lib.mapAttrs'
        (name: icfg: lib.nameValuePair "rp-bot-${name}" (mkService name icfg))
        cfg.instances;
    };
}
