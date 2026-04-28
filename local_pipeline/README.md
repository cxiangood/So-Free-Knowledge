# local_pipeline

Local closed-loop simulation modules (no CLI entrypoints in this package):

- `comm`: listen/send/feishu communication layer
- `core`: detect/lift/route/kb/obs/task main business logic
- `msg`: message types/cache/parse layer
- `flow`: unified engine + online/offline orchestration
- `store`: state/io/report layer
- `shared`: shared models/utils

Root legacy scripts have been removed. Implementation now lives only in the layered folders above.

## Python API

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
        enable_llm=False,
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
        enable_llm=False,
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
- Observe route supports rule-based question detection:
  - question + retrievable knowledge => auto reply to source `chat_id` (`text` message)
  - otherwise => fallback to `observe_store`.

Key runtime flags in `OnlineConfig` / `OfflineConfig`:

- `rag_enabled=True`
- `rag_top_k=5`
- `rag_min_score=0.35`
- `rag_embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
- `observe_auto_reply_enabled=True`
