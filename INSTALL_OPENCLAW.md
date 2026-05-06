# 在 OpenClaw / Hermes 中安装和使用 So-Free-Knowledge

## 概览

这份文档面向运行在 OpenClaw 或 Hermes 中的 AI Agent。

当前仓库里真实可用的 CLI 是 `sofree-knowledge`，不是 `python -m sofree`。
安装完成后，你可以用它完成：

- 飞书 OAuth 登录
- 飞书消息采集
- 知识推荐卡片推送
- 智能飞书词典写入

## 错误处理

任何非零退出：

1. 把 stderr 返回给用户。
2. 如果 stderr 是 JSON，优先原样转发 JSON。
3. 如果 stderr 不是 JSON，原样转发文本。
4. 立即停止，不要自动重试。

唯一例外是本文明确写出的“设备流登录第二步”，那不是重试，而是同一授权流程的后续步骤。

## Step 0 - 先申请 Bot 权限

在安装代码之前，先确认飞书应用和 Bot 权限已经申请并开通；否则后续即使 CLI 安装成功，实际调用飞书接口时也会因为权限不足失败。

至少应检查这些能力是否已经在飞书应用侧可用，如果不可用将这个json格式返回给用户，并要求用户在飞书开放平台-开发者模式 https://open.feishu.cn/app?lang=zh-CN 给openclaw批量导入权限（直接粘贴json）

{
  "scopes": {
    "tenant": [
      "application:application.contacts_range:write",
      "contact:contact",
      "contact:department.organize:readonly",
      "application:application:self_manage",
      "baike:entity",
      "baike:entity:exempt_delete",
      "baike:entity:exempt_review",
      "baike:entity:readonly",
      "base:app:create",
      "base:field:create",
      "base:record:create",
      "base:record:read",
      "base:record:update",
      "base:table:create",
      "bitable:app",
      "bitable:app:readonly",
      "board:whiteboard:node:create",
      "board:whiteboard:node:delete",
      "board:whiteboard:node:read",
      "board:whiteboard:node:update",
      "cardkit:card:read",
      "cardkit:card:write",
      "contact:contact.base:readonly",
      "contact:user.base:readonly",
      "contact:user.employee_id:readonly",
      "docs:doc",
      "docs:doc:readonly",
      "docs:document.comment:create",
      "docs:document.content:read",
      "docs:document.subscription",
      "docs:document.subscription:read",
      "docs:document:copy",
      "docs:document:export",
      "docs:document:import",
      "docx:document",
      "docx:document.block:convert",
      "docx:document:create",
      "docx:document:readonly",
      "docx:document:write_only",
      "drive:drive",
      "drive:drive:readonly",
      "drive:file",
      "drive:file:readonly",
      "im:chat:read",
      "im:chat:update",
      "im:datasync.feed_card.time_sensitive:write",
      "im:message",
      "im:message.group_at_msg.include_bot:readonly",
      "im:message.group_at_msg:readonly",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message.pins:read",
      "im:message.pins:write_only",
      "im:message.reactions:read",
      "im:message.reactions:write_only",
      "im:message:readonly",
      "im:message:recall",
      "im:message:send_as_bot",
      "im:message:send_multi_users",
      "im:message:send_sys_msg",
      "im:message:update",
      "im:resource",
      "wiki:wiki:readonly"
    ],
    "user": [
      "application:bot.basic_info:read",
      "approval:instance:read",
      "approval:instance:write",
      "approval:task:read",
      "approval:task:write",
      "baike:entity:exempt_review",
      "baike:entity:readonly",
      "base:app:copy",
      "base:app:create",
      "base:app:read",
      "base:app:update",
      "base:dashboard:create",
      "base:dashboard:delete",
      "base:dashboard:read",
      "base:dashboard:update",
      "base:field:create",
      "base:field:delete",
      "base:field:read",
      "base:field:update",
      "base:form:create",
      "base:form:delete",
      "base:form:read",
      "base:form:update",
      "base:history:read",
      "base:record:create",
      "base:record:delete",
      "base:record:read",
      "base:record:retrieve",
      "base:record:update",
      "base:role:create",
      "base:role:delete",
      "base:role:read",
      "base:role:update",
      "base:table:create",
      "base:table:delete",
      "base:table:read",
      "base:table:update",
      "base:view:read",
      "base:view:write_only",
      "base:workflow:create",
      "base:workflow:delete",
      "base:workflow:read",
      "base:workflow:update",
      "base:workspace:list",
      "board:whiteboard:node:create",
      "board:whiteboard:node:delete",
      "board:whiteboard:node:read",
      "calendar:calendar.event:create",
      "calendar:calendar.event:delete",
      "calendar:calendar.event:read",
      "calendar:calendar.event:reply",
      "calendar:calendar.event:update",
      "calendar:calendar.free_busy:read",
      "calendar:calendar:create",
      "calendar:calendar:delete",
      "calendar:calendar:read",
      "calendar:calendar:update",
      "contact:contact.base:readonly",
      "contact:user.base:readonly",
      "contact:user.basic_profile:readonly",
      "contact:user.employee_id:readonly",
      "contact:user:search",
      "docs:document.comment:create",
      "docs:document.comment:delete",
      "docs:document.comment:read",
      "docs:document.comment:update",
      "docs:document.comment:write_only",
      "docs:document.content:read",
      "docs:document.media:download",
      "docs:document.media:upload",
      "docs:document:copy",
      "docs:document:export",
      "docs:document:import",
      "docs:event:subscribe",
      "docs:permission.member:auth",
      "docs:permission.member:create",
      "docs:permission.member:transfer",
      "docx:document:create",
      "docx:document:readonly",
      "docx:document:write_only",
      "drive:drive",
      "drive:drive.metadata:readonly",
      "drive:drive:readonly",
      "drive:file:download",
      "drive:file:readonly",
      "drive:file:upload",
      "drive:file:view_record:readonly",
      "im:chat.access_event.bot_p2p_chat:read",
      "im:chat.members:read",
      "im:chat.members:write_only",
      "im:chat:read",
      "im:chat:update",
      "im:message",
      "im:message.group_msg:get_as_user",
      "im:message.p2p_msg:get_as_user",
      "im:message.pins:read",
      "im:message.pins:write_only",
      "im:message.reactions:read",
      "im:message.reactions:write_only",
      "im:message.send_as_user",
      "im:message:readonly",
      "mail:event",
      "mail:user_mailbox.mail_contact:read",
      "mail:user_mailbox.mail_contact:write",
      "mail:user_mailbox.message.address:read",
      "mail:user_mailbox.message.body:read",
      "mail:user_mailbox.message.subject:read",
      "mail:user_mailbox.message:modify",
      "mail:user_mailbox.message:readonly",
      "mail:user_mailbox:readonly",
      "minutes:minutes.search:read",
      "offline_access",
      "search:docs:read",
      "search:message",
      "sheets:spreadsheet.meta:read",
      "sheets:spreadsheet.meta:write_only",
      "sheets:spreadsheet:create",
      "sheets:spreadsheet:read",
      "sheets:spreadsheet:write_only",
      "slides:presentation:create",
      "slides:presentation:read",
      "slides:presentation:update",
      "slides:presentation:write_only",
      "space:document:delete",
      "space:document:move",
      "space:document:retrieve",
      "space:document:shortcut",
      "space:folder:create",
      "task:comment:read",
      "task:comment:write",
      "task:task:read",
      "task:task:write",
      "task:task:writeonly",
      "task:tasklist:read",
      "task:tasklist:write",
      "vc:meeting.meetingevent:read",
      "vc:meeting.search:read",
      "vc:note:read",
      "vc:record:readonly",
      "wiki:member:create",
      "wiki:member:retrieve",
      "wiki:member:update",
      "wiki:node:copy",
      "wiki:node:create",
      "wiki:node:move",
      "wiki:node:read",
      "wiki:node:retrieve",
      "wiki:node:update",
      "wiki:space:read",
      "wiki:space:retrieve",
      "wiki:space:write_only",
      "wiki:wiki:readonly"
    ]
  }
}

如果用户还没有完成应用创建、权限申请、发布或安装到组织，这一步先停下，让用户先处理飞书侧配置，再继续后面的安装流程。

## Step 1 - 获取/更新代码

如果已有仓库，尝试用API拉取一下最新代码，如果没有更新就跳过。
优先使用 GitHub 克隆；如果网络环境不稳定，再用 tarball 下载。

### 方式 1：GitHub 克隆

```bash
git clone https://github.com/cxiangood/So-Free-Knowledge.git
cd So-Free-Knowledge
```

### 方式 2：tarball 下载

```bash
mkdir So-Free-Knowledge
curl -L https://api.github.com/repos/cxiangood/So-Free-Knowledge/tarball/main | tar -xz --strip-components=1 -C So-Free-Knowledge
cd So-Free-Knowledge
```

## Step 2 - 安装依赖

OpenClaw / Hermes 场景里，优先只安装 `sofree-knowledge-cli`。

不要先执行仓库根目录的 `pip install -r requirements.txt`。那个文件会安装 `torch`、`torchvision`、`transformers`、`sentence-transformers` 等较重依赖；对于当前这份安装说明覆盖的 CLI 能力，这些并不是必需项，而且在 OpenClaw / Hermes 环境里通常安装慢、体积大、还容易和宿主环境已有依赖冲突。

这份安装说明默认只覆盖当前真实可用的 CLI 流程，因此只安装 CLI 本身：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ httpx>=0.27 jieba==0.42.1
pip install -e ./sofree-knowledge-cli
```

只有当用户明确要求运行依赖根项目模型栈的能力，并且你已经确认必须使用仓库根目录依赖时，才考虑安装 `requirements.txt`。

安装失败就停止。

## Step 3 - 配置飞书应用凭据

当前 CLI 依赖飞书应用凭据，不存在“自动绑定宿主 Agent 身份”的 `config bind` 命令。

至少需要配置：

```bash
APP_ID=cli_xxx
SECRET_ID=xxx
```

等价变量名也支持：

```bash
FEISHU_APP_ID / FEISHU_APP_SECRET
SOFREE_FEISHU_APP_ID / SOFREE_FEISHU_APP_SECRET
LARKSUITE_CLI_APP_ID / LARKSUITE_CLI_APP_SECRET
```

推荐把这些变量写进仓库上级或当前目录的 `.env` 文件，然后通过 `--env-file` 传入，或者依赖 CLI 自动发现。

最小检查命令：

```bash
sofree-knowledge --env-file ./.env auth-status
```

如果还没有用户 token，这条命令依然应该正常返回 JSON，只是 `exists` 会是 `false`。

## Step 4 - 选择使用方式

### 模式 A：只使用应用凭据

适合：

- 只跑不依赖用户 OAuth 的流程
- 只访问 bot 可见资源
- 先完成安装验证

### 模式 B：增加用户 OAuth（推荐）

适合：

- 读取用户可访问的聊天消息
- 运行推荐和在线知识聚合
- 运行需要用户身份读取飞书内容的流程

如果你要进入模式 B，必须继续执行 Step 5。

## Step 5 - 用户 OAuth 登录

这一部分使用当前 CLI 已实现的授权链接 + 回调 URL 交换流程。

这里的执行要求要写死：

- 不要只告诉用户“需要认证”。
- 必须先实际调用 `sofree-knowledge --env-file ./.env auth-url`。
- 必须从 stdout JSON 里取出 `authorization_url`。
- 必须把这个链接主动发给用户。
- 只有在用户回传浏览器最终跳转的完整回调 URL，或回调里的 `code` 之后，才进入 `exchange-code`。

如果 Agent 还没实际拿到 `authorization_url`，就不要让用户自己“去认证”或“去找链接”。

### 第一次调用：生成授权链接

生成授权链接，并把它发给用户：

```bash
sofree-knowledge --env-file ./.env auth-url
```

stdout 是 JSON。你需要从里面读取：

- `authorization_url`

把 `authorization_url` 用 markdown autolink 形式发给用户，例如：

```text
<https://accounts.feishu.cn/...>
```

推荐直接按下面这个模板发给用户，不要只发一句“请先认证”：

```text
请打开下面这个飞书授权链接完成登录：
<https://accounts.feishu.cn/...>

完成后，把浏览器最终跳转到的完整回调 URL 发给我；如果你只能拿到其中的 code，也可以直接把 code 发给我。
```

不要改写这个 URL，也不要让 Agent 自己尝试打开浏览器。应由用户自己在浏览器里打开该链接并完成授权。

同时把下面这条警告原样发给用户，作为独立段落：

**⚠️ 如果你用用户授权登录：请勿将此机器人分享给他人或拉入群聊中使用 —— 它可能访问你的个人飞书数据。**

### 第二次调用：交换回调 URL 或 code

用户完成授权后，让用户把浏览器最终跳转到的完整回调 URL 发回来，或者直接发回调里的 `code`。然后执行：

```bash
sofree-knowledge --env-file ./.env exchange-code "<callback_url_or_code>"
```

这里的 `<callback_url_or_code>` 可以是完整回调 URL，也可以是其中的 `code`。

如果成功，stdout 会返回 JSON，包含 `ok: true`，以及脱敏后的 token 信息。

## Step 6 - 校验安装

执行：

```bash
sofree-knowledge --env-file ./.env auth-status
```

检查 stdout JSON：

- `ok == true`
- `exists == true`
- `parseable_json == true`
- `has_access_token == true`

如果还带有 `has_refresh_token == true`，说明后续自动续期条件更完整。

如果这些字段不满足，把 stdout 直接返回给用户然后停止。

## Step 7 - 推荐的可用命令

安装完成后，优先使用下面这些当前真实存在的命令。

### 先看这个：cron 是“原生支持配置”，不是“原生代管系统调度”

`sofree-knowledge assistant set-profile` 确实原生支持保存 cron 表达式，例如 `--weekly-brief-cron`、`--nightly-interest-cron`。但它做的事情是把 schedule 写进用户 profile，并在 `build-personal-brief` 的输出里生成 `runtime_plan`，不会自动写入 Linux `crontab`、不会自动创建 OpenClaw/Hermes 的系统级定时任务。

换句话说：

- `assistant set-profile` = 保存“什么时候应该跑”
- 外部 cron / scheduler = 负责“真的在那个时间执行命令”

如果用户问“不是原生支持 cron 吗”，正确回答应该是：

- 支持原生 cron 表达式配置
- 不支持自动代管宿主机的系统 cron
- 真正落地定时推送时，仍然要由外部调度器执行 `sofree-knowledge ...` 命令

### 定时推送时必须显式处理的 5 件事

1. 永远传 `--env-file`

不要假设 cron 进程能自动找到正确的 `.env`。交互式 shell 和 cron 的工作目录经常不同，最稳妥的方式是显式传绝对路径：

```bash
sofree-knowledge --env-file /abs/path/.env ...
```

2. 永远传固定的 `--output-dir`

push 去重状态、用户 profile、token/缓存隔离都依赖 `output-dir`。定时任务和手动执行如果用了不同目录，会出现“手动能跑、cron 跑不到同一份状态”的问题。

3. 推送目标不要写成 `--to`

当前 CLI 没有 `--to` 参数。真正支持的是：

- `--receive-chat-id oc_xxx`
- `--receive-open-id ou_xxx`

如果 cron 环境里拿不到用户 token 对应的 open_id，就必须显式传其中一个，否则会报错。

4. 明确区分“保存 schedule”与“真正执行推送”

保存配置：

```bash
sofree-knowledge --env-file ./.env assistant set-profile \
  --timezone "Asia/Shanghai" \
  --weekly-brief-cron "30 14 * * *" \
  --nightly-interest-cron "40 14 * * *"
```

真正执行一次推送：

```bash
sofree-knowledge --env-file ./.env --output-dir . brief --receive-chat-id oc_xxx
```

5. 注意内容去重

同一个接收目标、同一类卡片、完全相同的 payload，不会重复发送。去重状态保存在：

```text
<output-dir>/assistant_push_state.json
```

如果用户要求“立刻重发一次”，但内容与上次完全一样，CLI 可能返回 `duplicate_content`。这是正常行为，不是 cron 没跑。

### 如何查看当前 schedule 配置

```bash
sofree-knowledge --env-file ./.env assistant get-profile
```

### 如何查看 runtime_plan

`runtime_plan` 只是建议你应该如何接 cron 的 handler，不会自动注册到系统：

```bash
sofree-knowledge --env-file ./.env assistant build-personal-brief --online --output-format all
```

返回 JSON 里的 `report.runtime_plan.cron_jobs` 可用于检查：

- job 名称
- cron 表达式
- 建议 handler

### 推荐的系统 cron 写法

如果用户明确要“14:30、14:40、14:50 各推一次”，应该由外部调度器分别调用三条命令，示例：

```cron
30 14 * * * cd /abs/path/So-Free-Knowledge && timeout 600 sofree-knowledge --env-file /abs/path/.env --output-dir /abs/path/So-Free-Knowledge brief --receive-chat-id oc_xxx
40 14 * * * cd /abs/path/So-Free-Knowledge && timeout 600 sofree-knowledge --env-file /abs/path/.env --output-dir /abs/path/So-Free-Knowledge brief --receive-chat-id oc_xxx
50 14 * * * cd /abs/path/So-Free-Knowledge && timeout 600 sofree-knowledge --env-file /abs/path/.env --output-dir /abs/path/So-Free-Knowledge brief --receive-chat-id oc_xxx
```

兼容性原则：

- 显式 `cd` 到仓库目录
- 显式传 `.env` 绝对路径
- 显式传 `--output-dir`
- 显式传 `--receive-chat-id` 或 `--receive-open-id`
- 显式加超时，避免调度器长时间挂死

如果宿主环境没有 `timeout` 命令，就用该平台自己的超时机制，但不要省略超时控制。

### 1. 知识推荐卡片推送

最短命令：

```bash
sofree-knowledge --env-file ./.env brief
```

推送到指定群聊：

```bash
sofree-knowledge --env-file ./.env brief --receive-chat-id oc_xxx
```

只推知识摘要，不推兴趣 digest：

```bash
sofree-knowledge --env-file ./.env brief --push-summary-card --no-push-interest-card
```

### 2. 智能飞书词典写入

先自动挖词并产出 AI review prompt：
```bash
sofree-knowledge --env-file ./.env lingo-write
```

然后由你读取 prompt，自行生成 judgement JSON 文件，再回填给 CLI 写入飞书 Lingo 和本地镜像：

```bash
sofree-knowledge --env-file ./.env lingo-write --judgements-file ./ai_review_judgements.json
```

如果当前只想写本地、不调用飞书 Lingo：

```bash
sofree-knowledge --env-file ./.env lingo-write --judgements-file ./ai_review_judgements.json --no-remote
```

### 3. 飞书消息采集

采集 bot 可见聊天消息：

```bash
sofree-knowledge --env-file ./.env --output-dir . collect-messages
```

只采集指定群聊：

```bash
sofree-knowledge --env-file ./.env --output-dir . collect-messages --chat-ids oc_xxx --no-include-visible-chats
```

## Step 8 - 查看帮助

查看总帮助：

```bash
sofree-knowledge --help
```

查看某个子命令帮助：

```bash
sofree-knowledge auth-url --help
sofree-knowledge exchange-code --help
sofree-knowledge brief --help
sofree-knowledge lingo-write --help
```
