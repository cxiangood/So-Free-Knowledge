from __future__ import annotations

def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline as _run_pipeline

        return _run_pipeline
    if name == "ChatMessageStore":
        from .chat_message_store import ChatMessageStore as _store

        return _store
    if name == "MessageEventBus":
        from .message_event_bus import MessageEventBus as _bus

        return _bus
    if name in {"MessageEvent", "OpenAPIMessageListener"}:
        from .openapi_message_listener import MessageEvent, OpenAPIMessageListener

        return {"MessageEvent": MessageEvent, "OpenAPIMessageListener": OpenAPIMessageListener}[name]
    raise AttributeError(name)


__all__ = ["run_pipeline", "ChatMessageStore", "MessageEventBus", "MessageEvent", "OpenAPIMessageListener"]
