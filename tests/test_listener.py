from local_pipeline.listener_service import ListenerService, ListenerServiceConfig

service = ListenerService(
    ListenerServiceConfig(
        env_file=".env",
        event_types="im.message.receive_v1",
        compact=False,
        print_events=True,
        chat_history_limit=100,
        context_window_size=1,
        enable_llm=False,
        task_push_enabled=True,
        step_trace_enabled=False,
        task_push_chat_id="oc_9663f97db577d40181f3ccc9a4ef4b03",
    )
)
service.start()