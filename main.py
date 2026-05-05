from insight.flow.online import OnlineConfig, start
from utils import (
    configure_logging,
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_str,
    getenv,
    load_env_file,
)


env_file = get_config_str("insight.env_file", ".env")
load_env_file(env_file, override=True)

configure_logging(
    app_name=get_config_str("insight.logging.app_name", "INSIGHT"),
    quiet=get_config_bool("insight.logging.quiet", True),
)

start(
    OnlineConfig(
        env_file=env_file,
        event_types=get_config_str("insight.event_types", "im.message.receive_v1"),
        compact=get_config_bool("insight.compact", False),
        output_dir=get_config_str("insight.output_dir", "outputs/local_pipeline"),
        state_dir=get_config_str("insight.state_dir", "outputs/local_pipeline/state"),
        chat_history_path=get_config_str("insight.chat_history_path", "outputs/local_pipeline/state/chat_message_store.json"),
        chat_history_limit=get_config_int("insight.chat_history_limit", 100),
        context_window_size=get_config_int("insight.context_window_size", 20),
        task_push_enabled=get_config_bool("insight.task_push_enabled", False),
        detect_threshold=get_config_float("insight.detect_threshold", 40.0),
        task_push_chat_id=str(getenv("TASK_PUSH_CHAT_ID", "") or "").strip(),
        step_trace_enabled=get_config_bool("insight.step_trace_enabled", True),
        rag_enabled=get_config_bool("insight.rag_enabled", True),
        rag_top_k=get_config_int("insight.rag_top_k", 5),
        rag_min_score=get_config_float("insight.rag_min_score", 0.35),
        rag_embed_model=get_config_str("insight.rag_embed_model", "BAAI/bge-large-zh"),
        observe_auto_reply_enabled=get_config_bool("insight.observe_auto_reply_enabled", True),
        observe_ferment_threshold=get_config_float("insight.observe_ferment_threshold", 4.0),
        observe_logic1_base=get_config_float("insight.observe_logic1_base", 1.0),
        observe_logic2_base=get_config_float("insight.observe_logic2_base", 1.5),
        observe_logic3_base=get_config_float("insight.observe_logic3_base", 2.0),
        observe_force_non_observe_on_pop=get_config_bool("insight.observe_force_non_observe_on_pop", True),
        max_workers=get_config_int("insight.online.max_workers", 8),
    )
)
