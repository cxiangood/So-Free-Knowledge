# sofree-knowledge

Standalone CLI for SoFree Knowledge message collection and per-chat knowledge scope.

This package does not depend on `feishu_sync` or Feishu skills. It calls Feishu OpenAPI directly.

## Install

From this repository:

```bash
cd sofree-knowledge-cli
python -m pip install -e .
```

Or run without installing:

```bash
python -m sofree_knowledge.cli --help
```

## Configure

Set app credentials through environment variables:

```bash
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
```

Equivalent names are also supported:

```bash
SOFREE_FEISHU_APP_ID / SOFREE_FEISHU_APP_SECRET
LARKSUITE_CLI_APP_ID / LARKSUITE_CLI_APP_SECRET
```

For reading chats that require user membership, provide a user access token:

```bash
export FEISHU_ACCESS_TOKEN="u-xxx"
```

If `FEISHU_ACCESS_TOKEN` is not set, the CLI tries to read `~/.feishu/token.json`.

You can also pass an env file:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env collect-messages
```

## Commands

Collect messages from chats visible to the bot:

```bash
sofree-knowledge --output-dir . collect-messages
```

Collect a specific chat only:

```bash
sofree-knowledge --output-dir . collect-messages --chat-ids oc_xxx --no-include-visible-chats
```

Collect with a time range:

```bash
sofree-knowledge --output-dir . collect-messages --start-time 2026-04-01 --end-time 2026-04-25
```

Set a chat's knowledge scope:

```bash
sofree-knowledge --output-dir . set-knowledge-scope oc_xxx chat_only
sofree-knowledge --output-dir . set-knowledge-scope oc_xxx global_review
```

Read a chat's knowledge scope:

```bash
sofree-knowledge --output-dir . get-knowledge-scope oc_xxx
```

## Output

`collect-messages` prints a JSON summary and writes:

```text
message_archive/<run_id>/
  manifest.json
  chats.jsonl
  messages.jsonl
```

`set-knowledge-scope` writes:

```text
knowledge_policy.json
```

All CLI commands print JSON to stdout, so OpenClaw or other automation can parse the result.

## Permissions

For bot-visible chat collection:

- `im:chat:read`
- `im:message:readonly`
- bot must be in the chat

For user-token chat reading:

- `im:message.group_msg:get_as_user`
- `im:message.p2p_msg:get_as_user`
- the user must be a member of the chat
