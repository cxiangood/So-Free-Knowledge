from local_pipeline.listener_service import ListenerService, ListenerServiceConfig

service = ListenerService(
    ListenerServiceConfig(
        env_file=".env",
        event_types="im.message.receive_v1",
        compact=True,
        print_events=True,
    )
)
service.start()