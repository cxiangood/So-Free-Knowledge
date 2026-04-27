# local_pipeline

本地闭环模拟流水线（不连接飞书）：

- `ingest`：消息接入与标准化
- `signal_detector`：弱信号检测与规则评分
- `semantic_lifter`：模板升维 + 可选 LLM 润色
- `router`：路由到 `knowledge/task/observe`
- `stores`：JSON 状态持久化
- `simulator`：本地推送事件模拟
- `feedback_loop`：反馈驱动二次灵感

## CLI

```bash
python -m local_pipeline run --messages-file message_archive/20260427T043305Z/messages.jsonl --enable-llm false
```

启用 task 路由即时飞书卡片推送（OpenAPI）：

```bash
python -m local_pipeline run --messages-file message_archive/20260427T043305Z/messages.jsonl --task-push-enabled true
```

若发送失败，会写入待重试队列：

```text
outputs/local_pipeline/state/pending_task_push.jsonl
```

等价子命令：

```bash
python -m local_pipeline score --messages-file ...
python -m local_pipeline route --messages-file ... --enable-llm true
python -m local_pipeline simulate --state-dir outputs/local_pipeline/state --task-updates-file task_updates.jsonl
python -m local_pipeline report --run-dir outputs/local_pipeline/<run_id>
```

监听飞书消息（OpenAPI WebSocket）：

```bash
python -m local_pipeline listen-messages --env-file .env --compact true --print-events true
```
