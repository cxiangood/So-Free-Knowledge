# local_pipeline

Local silent knowledge Agent modules:

- `agent`: silent knowledge push Agent runtime, memory, tools, and reports
- `flow`: LangGraph decision kernel + online/offline orchestration

- `comm`: listen/send/feishu communication layer
- `core`: detect/lift/route/kb/obs/task main business logic
- `msg`: message types/cache/parse layer
- `store`: state/io/report layer
- `shared`: shared models/utils

The key product shape is no longer a command-triggered bot. `SilentKnowledgeAgent`
observes Feishu messages without requiring `@bot` or slash commands, routes weak
signals through the LangGraph engine, and only pushes knowledge/tasks when the
configured thresholds or observe-fermentation rules are met.

## Python API

Agent mode:

```python
from local_pipeline.agent import SilentKnowledgeAgent, SilentKnowledgeAgentConfig
from local_pipeline.flow.engine import EngineConfig

agent = SilentKnowledgeAgent(
    SilentKnowledgeAgentConfig(
        engine=EngineConfig(
            env_file=".env",
            output_dir="outputs/local_pipeline",
            state_dir="outputs/local_pipeline/state",
            chat_history_path="outputs/local_pipeline/state/chat_message_store.json",
            task_push_enabled=True,
            task_push_chat_id="oc_xxx",
        )
    )
)

# Called by the Feishu listener. No user command is required.
report = agent.handle_message(event)
print(report.to_dict())
```

Agent tool definitions are explicit and inspectable:

```python
for tool in agent.tool_definitions():
    print(tool["name"], tool["parameters"])
```

The next-step policy lives in `local_pipeline.agent.planner.SilentKnowledgePlanner`:

- `plan_message_observation(...)` decides which tool to call when a silent Feishu message arrives.
- `decide_after_observation(...)` converts the engine result into Agent decisions such as `store`, `push`, `wait`, or `retry`.
- `plan_direct_question(...)` handles explicit question paths through memory retrieval.

Offline mode (fixed archive input by default):

```python
from local_pipeline.flow.offline import OfflineConfig, run

result = run(
    OfflineConfig(
        messages_file="message_archive/20260427T043305Z/messages.jsonl",
        output_dir="outputs/local_pipeline",
        state_dir="outputs/local_pipeline/state",
        chat_history_path="outputs/local_pipeline/state/chat_message_store.json",
        chat_history_limit=100,
        context_window_size=20,
        candidate_threshold=0.45,
        knowledge_threshold=0.60,
        task_threshold=0.50,
        task_push_enabled=True,
        task_push_chat_id="oc_xxx",
        env_file=".env",
        step_trace_enabled=True,  # print message path
        rag_enabled=True,
        rag_top_k=5,
        rag_min_score=0.35,
        rag_embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        observe_auto_reply_enabled=True,
    )
)
print(result)
```

Online mode:

```python
from local_pipeline.flow.online import OnlineConfig, start

start(
    OnlineConfig(
        env_file=".env",
        event_types="im.message.receive_v1",
        compact=False,
        output_dir="outputs/local_pipeline",
        state_dir="outputs/local_pipeline/state",
        chat_history_path="outputs/local_pipeline/state/chat_message_store.json",
        chat_history_limit=100,
        context_window_size=20,
        candidate_threshold=0.45,
        knowledge_threshold=0.60,
        task_threshold=0.50,
        task_push_enabled=True,
        task_push_chat_id="oc_xxx",
        step_trace_enabled=True,  # print message path
        rag_enabled=True,
        rag_top_k=5,
        rag_min_score=0.35,
        rag_embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        observe_auto_reply_enabled=True,
    )
)
```

Unified downstream flow (online/offline): `cache -> detect -> lift -> route -> knowledge/observe/task -> task push`.

Agent boundary: `silent message trigger -> Agent runtime -> Agent tools/memory -> LangGraph engine -> knowledge/task/observe push`.

Message payload format inside `local_pipeline` is:

```json
{
  "sender": { "...": "..." },
  "message": { "...": "..." }
}
```

No outer `event` wrapper is used for internal message serialization or chat history storage.

## Phase-2 RAG

- Knowledge route now writes to local vector store: `outputs/local_pipeline/state/vector_kb/`
  - `index.faiss`
  - `meta.json`
- Task route uses RAG retrieval to enrich task card content before push/store.
- Observe route supports LLM-first question detection with rule fallback:
  - question + retrievable knowledge => auto reply to source `chat_id` (`text` message)
  - otherwise => fallback to `observe_store`.

Key runtime flags in `OnlineConfig` / `OfflineConfig`:

- `rag_enabled=True`
- `rag_top_k=5`
- `rag_min_score=0.35`
- `rag_embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
- `observe_auto_reply_enabled=True`

## Path Cases

- Full message-path cases and trigger examples: [PATH_CASES.md](./PATH_CASES.md)
