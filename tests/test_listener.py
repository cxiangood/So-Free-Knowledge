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
        context_window_size=2,
        enable_llm=False,
        task_push_enabled=True,
        task_push_chat_id="oc_9663f97db577d40181f3ccc9a4ef4b03",
        step_trace_enabled=True,  # print each step after completion
    )
)