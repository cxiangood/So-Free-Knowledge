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

## Step 1 - 获取代码

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

先安装项目依赖，再安装 CLI。

```bash
pip install -r requirements.txt
pip install -e ./sofree-knowledge-cli
```

安装失败就停止。

## Step 3 - 配置飞书应用凭据

当前 CLI 依赖飞书应用凭据，不存在“自动绑定宿主 Agent 身份”的 `config bind` 命令。

至少需要配置：

```bash
APP_ID=cli_xxx
APP_SECRET=xxx
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

这一部分使用当前 CLI 已实现的设备流登录。

### 第一次调用：获取授权链接

直接启动设备流并进入授权：

```bash
sofree-knowledge --env-file ./.env auth login
```

这个命令会直接发起设备流登录。默认会尝试打开浏览器，并在用户完成授权后返回结果。

同时把下面这条警告原样发给用户，作为独立段落：

**⚠️ 如果你用用户授权登录：请勿将此机器人分享给他人或拉入群聊中使用 —— 它可能访问你的个人飞书数据。**

### 第二次调用：完成设备流登录

用户确认授权完成后，再执行：

```bash
sofree-knowledge --env-file ./.env auth login --device-code <device_code>
```

这里的 `<device_code>` 必须使用第一次调用返回的 `request.device_code`。

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
sofree-knowledge auth login --help
sofree-knowledge brief --help
sofree-knowledge lingo-write --help
```
