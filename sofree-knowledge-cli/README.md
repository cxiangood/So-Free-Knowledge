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

### Token Lifecycle

- `user_access_token` is short-lived, typically around 2 hours.
- If a valid `refresh_token` is present, the CLI refreshes `user_access_token` automatically.
- The current client refreshes proactively before expiry and also retries once on `401`.
- Re-authentication is only needed when refresh itself fails, for example:
  - `refresh_token` is missing or expired
  - the user revoked authorization
  - required scopes changed after an upgrade

In other words: expired access token should normally be handled by refresh, not by forcing the user through OAuth again.

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

## Confused Detection

Rule-trigger + LLM-judge workflow for lightweight "confused" detection:

```bash
sofree-knowledge confused detect-candidates --messages-file ./messages.json
```

Build prompt for one candidate:

```bash
sofree-knowledge confused build-judge-prompt --candidate-file ./candidate.json
```

Parse LLM result and get inline insert text:

```bash
sofree-knowledge confused parse-judgement --judgement-file ./judgement.json
```

Returned `inline_insert_text` is intentionally short, suitable for subtle inline insertion
instead of a bot card or direct bot reply.

## Lingo CRUD

Write to Feishu Lingo directly (remote by default), while mirroring to local store:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env lingo upsert --keyword "北极星指标" --type black --value "团队核心牵引指标"
```

Delete remote Lingo entry by `entity_id`:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env lingo delete --entity-id enterprise_xxx
```

Local-only mode (no remote API call):

```bash
sofree-knowledge --output-dir . lingo upsert --no-remote --keyword "北极星指标" --type black --value "团队核心牵引指标"
sofree-knowledge --output-dir . lingo delete --no-remote --keyword "北极星指标"
```

Weekly auto-mining pipeline:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env --output-dir . lingo auto-sync --recent-days 7 --min-run-interval-days 7
```

This command stitches the full flow together:

1. Collect recent chat messages.
2. Run `token_classify` keyword mining with optional analyzer/classifier stages.
3. Extract keyword contexts from messages.
4. Ask the built-in LLM/OpenClaw-style judge which candidates should become new lingo entries.
5. Batch sync publishable entries into local store and Feishu Lingo.

## Personal Assistant

Recommended final usage for OpenClaw scheduling:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env --output-dir . assistant recommend --output-format card
```

This single command will:

1. Pull recent online documents/messages/knowledge data.
2. Append weak-supervision samples to `assistant_dual_tower_samples.jsonl`.
3. If accumulated samples are not enough, automatically disable dual tower and fall back to OpenClaw signals.
4. Once accumulated samples are enough, automatically train/update `assistant_dual_tower_model.json`.
5. Use the trained dual tower model for recommendation automatically.

Tune the switch threshold if needed:

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env --output-dir . assistant recommend --dual-tower-min-samples 20 --output-format json
```

Build personal aggregation from documents + access records + chat messages + knowledge items, then score urgency/recommendation:

```bash
sofree-knowledge assistant build-personal-brief --documents-file ./documents.json --access-records-file ./access.json --messages-file ./messages.json --knowledge-file ./knowledge.json --target-user-id ou_xxx
```

Output can be selected:

```bash
sofree-knowledge assistant build-personal-brief --documents-file ./documents.json --output-format doc
sofree-knowledge assistant build-personal-brief --documents-file ./documents.json --output-format card
```

`all` mode returns structured report + `doc_markdown` + Feishu card JSON.

One-command online mode (auto pull Feishu chats + recent drive docs, auto resolve current user id from token):

```bash
sofree-knowledge --env-file ../So-Free-Knowledge/.env assistant build-personal-brief --online --output-format all
```

Limit source scope:

```bash
sofree-knowledge assistant build-personal-brief --online --chat-ids oc_xxx,oc_yyy --no-include-visible-chats --max-chats 10 --max-messages-per-chat 100
```

Push card after build (default target is personal session via open_id):

```bash
sofree-knowledge assistant build-personal-brief --online --push --output-format card
```

Only when `--receive-chat-id` is explicitly provided will push go to group chat:

```bash
sofree-knowledge assistant build-personal-brief --online --push --receive-chat-id oc_xxx --output-format card
```

### Assistant Output And Push Behavior

`assistant build-personal-brief` and `assistant recommend` always return JSON envelopes at the CLI layer. The difference is in which fields are included:

- `--output-format json`
  - returns structured metadata and report fields
  - intentionally omits rendered card payloads
- `--output-format card`
  - returns `card` and `interest_card` payloads in JSON
- `--output-format all`
  - returns report plus card payloads
- `--output-format doc`
  - returns markdown only and is deprecated

This means "I got JSON instead of a pushed card" is not necessarily a bug. Common cases are:

1. `--push` was not enabled.
2. `--output-format json` was used, so the command returned report JSON instead of card payloads.
3. profile setup is still required; in that case recommendation push is deferred and the command returns or pushes `profile_setup_card` first.
4. push was deduplicated because the same card content was already sent recently; check `meta.push.skipped`.
5. push failed but the command still returned JSON successfully; check `meta.push.errors`.
6. no explicit `--push-summary-card` or `--push-interest-card` was set; current default push behavior sends the interest card path, not every card type.

When debugging recommendation runs, check these fields first:

- `meta.push.enabled`
- `meta.push.summary_enabled`
- `meta.push.interest_enabled`
- `meta.push.profile_setup_prompted`
- `meta.push.recommendation_deferred_until_profile_confirmed`
- `meta.push.skipped`
- `meta.push.errors`
- `report.retrieval_plan.strategy`

## Permissions

Bot-visible collection:

- `im:chat:read`
- `im:message:readonly`
- bot must be in the chat

User-token chat reading:

- `im:message.group_msg:get_as_user`
- `im:message.p2p_msg:get_as_user`
- the user must be a member of the chat
