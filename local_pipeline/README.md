# local_pipeline

Local closed-loop simulation modules (no CLI entrypoints in this package):

- `ingest`: normalize chat messages
- `signal_detector`: weak-signal detection and scoring
- `semantic_lifter`: structured lifting with optional LLM refinement
- `router`: route to `knowledge/task/observe`
- `stores`: JSON persistence
- `simulator`: local push-event simulation
- `feedback_loop`: secondary inspiration generation
- `openapi_message_listener`: Feishu OpenAPI WebSocket listener
- `message_event_bus`: in-process pub/sub bus
- `chat_message_store`: per-chat message ring-buffer persistence
- `realtime_processor`: message-driven realtime weak-signal closed loop

## Python API

Run pipeline:

```python
from local_pipeline.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(
    messages_file="message_archive/20260427T043305Z/messages.jsonl",
    config=PipelineConfig(
        task_push_enabled=True,
        task_push_chat_id="oc_xxx",
        env_file=".env",
    ),
)
print(result)
```

Start listener service:

```python
from local_pipeline.listener_service import ListenerService, ListenerServiceConfig

service = ListenerService(
    ListenerServiceConfig(
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
service.start()
```

Listener default behavior is now:
`message received -> cache by chat_id -> weak-signal detect -> semantic lift -> route -> local store -> task push`.
