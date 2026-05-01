# local_pipeline 消息路径案例集（基于当前规则 + LLM逻辑）

默认阈值：

- `candidate_threshold=0.45`
- `task_threshold=0.50`
- `knowledge_threshold=0.60`

## 1) 去重跳过

路径：

`message_cache→deduplicate(skipped)`

触发条件：

- 同一个 `message_id` 重复回调。

消息示例：

- `大家今天下午3点同步一下进展`

备注：

- 内容无所谓，关键是 `message_id` 重复。

## 2) 无候选（弱信号/噪声）

路径：

`message_cache→deduplicate→context_extract→signal_detect→done`

触发条件：

- 噪声或低分，未过 `candidate_threshold`。

消息示例：

- `ok`
- `收到`
- `https://example.com`

## 3) task（RAG 未命中）

路径：

`...→signal_detect→semantic_lift→route→task_store→task_push→done`

触发条件：

- 问句 + 至少 1 个 mention（提高 action/impact 分）。
- 向量库为空或检索分数低于阈值。

消息示例：

- `@Tom 这个接口今天能修好吗？`

## 4) task（RAG 命中增强）

路径：

`...→route→rag_retrieve_task→task_store→task_push→done`

触发条件：

- 与向量库已有知识主题相近，检索命中。

消息示例：

- `@Tom 订单超时重试策略要怎么落地？今天能给方案吗？`

## 5) task（推送失败）

路径：

`...→route→rag_retrieve_task→task_store→task_push(failed)`

触发条件：

- task 可路由，但推送配置缺失或飞书发送失败。

消息示例：

- `@Tom 请今天内完成告警降噪处理并回复结果？`

## 6) observe（非问题）

路径：

`...→signal_detect→semantic_lift→route→observe_store→done`

触发条件：

- 路由到 observe，且规则不判定为问题。

消息示例：

- `本周接口响应时间波动较大，需要持续观察。`

## 7) observe 问题可答并回聊

路径：

`...→route→rag_retrieve_observe→observe_reply→done`

触发条件：

- 判定为问题 + RAG 命中 + 文本发送成功。

消息示例：

- `请问我们之前对“登录超时”的处理结论是什么？`

## 8) observe 问题不可答

路径：

`...→route→rag_retrieve_observe→observe_store→done`

触发条件：

- 判定为问题，但检索不到足够知识。

消息示例：

- `为什么昨天某客户环境突然全量失败？有历史结论吗？`

## 9) observe 可答但回聊失败

路径：

`...→route→rag_retrieve_observe→observe_reply→observe_store→done`

触发条件：

- RAG 能答，但飞书文本发送失败。

消息示例：

- `请问“消息重放保护”之前有没有规范？`

## 10) knowledge（LLM 主导）

路径：

`...→semantic_lift(LLM)→route→knowledge_store→done`

触发条件：

- LLM-first mode (no enable flag)
- LLM 将 `suggested_target=knowledge`
- `confidence>=0.60`

消息示例：

- `我们把这次发布复盘沉淀成规范：灰度比例、回滚阈值、监控看板统一模板。`

## LLM 特定案例

### A. LLM 降级路径

现象：

- 缺少 LLM 配置时，自动回退模板升维。
- `warnings` 中会出现缺失配置告警。

### B. LLM 改写路由

现象：

- 原本规则可能是 `observe/task`，LLM 把 `suggested_target` 改成 `knowledge` 并提高 `confidence`，触发 `knowledge_store`。

消息示例：

- `这次故障处理步骤已经稳定，建议整理为标准SOP供全员复用。`

## 使用说明与注意事项

- 在线模式下 `@人` 需要对应飞书 `mentions`，不是只看纯文本 `@` 字符。
- 当前词库存在乱码常量，中文关键词命中能力偏弱；问号与 mention 是更稳定触发手段。
- 当前实现里“observe 问题不可答”分支的 `observe_fallback_count` 可能重复计数，后续需要修正统计口径。
