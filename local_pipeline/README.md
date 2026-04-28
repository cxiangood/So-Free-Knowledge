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
        task_push_enabled=True,
        task_push_chat_id="oc_xxx",
        env_file=".env",
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
        print_events=True,
        output_dir="outputs/local_pipeline",
        state_dir="outputs/local_pipeline/state",
        chat_history_path="outputs/local_pipeline/state/chat_message_store.json",
        chat_history_limit=100,
        context_window_size=20,
        enable_llm=False,
        task_push_enabled=True,
        task_push_chat_id="oc_xxx",
        step_trace_enabled=True,  # print each step after completion
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
