# LangGraph 状态设计说明

本文档描述本项目中 LangGraph 工作流的状态字段定义、作用及流转逻辑。

## 状态定义总览

状态类型为 `EngineGraphState`，是继承自 `TypedDict` 的字典类型，定义于 `local_pipeline/flow/engine.py:100-110`，包含以下字段：

```python
class EngineGraphState(TypedDict, total=False):
    message: MessageEvent
    result: EngineResult
    simple_messages: list[str]
    current_candidates: list[Any]
    cards: list[LiftedCard]
    decisions: list[Any]
    decision_index: int
    failed_attempts: list[TaskPushAttempt]
    trace_status: str
    trace_started: bool
```

## 各字段详细说明

### 1. 核心输入与结果状态

| 状态字段 | 类型 | 作用模块 | 用途说明 |
|---------|------|----------|----------|
| `message` | `MessageEvent` | 所有节点 | 输入的飞书消息事件，包含消息ID、聊天ID、内容等核心信息，是整个工作流的处理对象，在工作流入口时初始化 |
| `result` | `EngineResult` | 所有节点 | 全局结果统计对象，记录整个处理流程的各类指标（候选数、路由结果、推送统计、错误信息等），最终返回给调用方，在工作流入口时初始化，各节点更新对应统计项 |

### 2. 上下文与中间处理状态

| 状态字段 | 类型 | 作用模块 | 用途说明 |
|---------|------|----------|----------|
| `simple_messages` | `list[str]` | `context_extract` → `signal_detect` → `semantic_lift` | 简化格式的历史上下文消息列表，`context_extract` 节点从聊天缓存中拉取指定窗口大小的历史消息转换得到，用于信号检测和语义提升时提供对话上下文 |
| `current_candidates` | `list[Any]` | `signal_detect` → `semantic_lift` | `signal_detect` 阶段识别出的潜在有价值内容候选，作为语义提升的输入，无候选时直接进入结束流程 |
| `cards` | `list[LiftedCard]` | `semantic_lift` → `route` → 各处理节点 | `semantic_lift` 节点对候选进行语义提升后生成的结构化卡片，包含摘要、问题、建议、证据等完整信息，是路由和后续处理的核心对象 |

### 3. 路由与决策状态

| 状态字段 | 类型 | 作用模块 | 用途说明 |
|---------|------|----------|----------|
| `decisions` | `list[Any]` | `route` → `select_decision` | `route` 节点对每个卡片做出的分类决策（目标池：knowledge/task/observe），指导后续处理流向 |
| `decision_index` | `int` | `select_decision` → 各处理节点 | 当前正在处理的决策索引，初始为0，每次处理完一个决策后自增1，用于循环遍历所有决策，实现多卡片的依次处理，当索引大于等于决策总数时进入结束流程 |

### 4. 辅助状态

| 状态字段 | 类型 | 作用模块 | 用途说明 |
|---------|------|----------|----------|
| `failed_attempts` | `list[TaskPushAttempt]` | `task_store` → `finalize` | 记录任务推送失败的尝试，在 `task_store` 节点推送失败时添加，在 `finalize` 节点统一加入重试队列 |
| `trace_status` | `str` | 各节点 → `finalize` | 跟踪整个流程的执行状态（默认"ok"，出现错误时设为"failed"），用于链路追踪和错误上报 |
| `trace_started` | `bool` | `deduplicate` → `finalize` | 标记是否已经启动链路跟踪，在去重节点未跳过处理时设置为True，确保跟踪流程的完整性 |

## 状态流转逻辑

1. **初始化**：工作流入口时初始化状态，包含输入消息、空的结果对象和其他默认值
2. **节点处理**：每个节点读取所需状态字段，执行业务逻辑后返回需要更新的状态字段
3. **自动合并**：LangGraph 自动合并节点返回的状态更新，传递给下一个节点
4. **循环处理**：多决策处理通过 `decision_index` 实现循环流转，`select_decision` 节点根据当前索引判断下一个处理节点，直到所有决策处理完毕
5. **结束流程**：所有决策处理完成后进入 `finalize` 节点，完成收尾工作，最终状态中的 `result` 字段包含完整的执行统计信息返回给调用方

## 设计特点

- 每个节点只需关注自身的输入输出，模块间完全解耦
- 状态更新采用增量方式，节点只返回需要修改的字段，避免全量覆盖
- 多卡片处理通过索引循环实现，无需复杂的分支设计
- 全流程可追溯，统计信息集中在 `result` 对象中，便于调试和监控
