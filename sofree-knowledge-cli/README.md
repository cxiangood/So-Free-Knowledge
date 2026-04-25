# sofree-knowledge

Standalone CLI for SoFree Knowledge message collection, Feishu OAuth, and per-chat knowledge scope.

It does not depend on `feishu_sync` or Feishu skills.

## Install

```bash
cd sofree-knowledge-cli
python -m pip install -e .
```

## Configure App Credentials

Use environment variables or an env file:

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
```

Equivalent names are also supported:

```bash
SOFREE_FEISHU_APP_ID / SOFREE_FEISHU_APP_SECRET
LARKSUITE_CLI_APP_ID / LARKSUITE_CLI_APP_SECRET
APP_ID / APP_SECRET
```

Pass an env file when running:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env auth-status
```

If `--env-file` is omitted, CLI will auto-discover `.env` in this order:

1. `<output-dir>/.env`
2. `./.env`
3. `../.env`
4. `<output-dir>/So-Free-Knowledge/.env` (legacy fallback)

## OAuth

Print an authorization URL:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env auth-url
```

Start OAuth flow. Without `--enable-autofill`, this opens the browser and prints the next step:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env init-token
```

Capture the local callback automatically:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env init-token --enable-autofill
```

Exchange a copied code or redirect URL:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env exchange-code "http://localhost:8000/callback?code=..."
```

Check token status:

```bash
sofree-knowledge auth-status
```

Tokens are saved to `~/.feishu/token.json` by default. CLI output redacts token values.

## Message Collection

Collect messages from bot-visible chats:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env --output-dir . collect-messages
```

Collect a specific chat only:

```bash
sofree-knowledge --output-dir . collect-messages --chat-ids oc_xxx --no-include-visible-chats
```

Collect with a time range:

```bash
sofree-knowledge --output-dir . collect-messages --start-time 2026-04-01 --end-time 2026-04-25
```

Output:

```text
message_archive/<run_id>/
  manifest.json
  chats.jsonl
  messages.jsonl
```

## Knowledge Scope

```bash
sofree-knowledge --output-dir . set-knowledge-scope oc_xxx chat_only
sofree-knowledge --output-dir . set-knowledge-scope oc_xxx global_review
sofree-knowledge --output-dir . get-knowledge-scope oc_xxx
```

Policy is saved to:

```text
knowledge_policy.json
```

## Permissions

Bot-visible collection:

- `im:chat:read`
- `im:message:readonly`
- bot must be in the chat

User-token chat reading:

- `im:message.group_msg:get_as_user`
- `im:message.p2p_msg:get_as_user`
- the user must be a member of the chat
