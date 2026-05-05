from insight.flow.online import OnlineConfig, start
from utils import (
    configure_logging,
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_str,
    load_env_file,
)


env_file = get_config_str("insight.env_file")
load_env_file(env_file, override=True)

configure_logging(
    app_name=get_config_str("insight.logging.app_name"),
    quiet=get_config_bool("insight.logging.quiet"),
)

start(
    OnlineConfig(
        env_file=env_file,
        event_types=get_config_str("insight.event_types"),
        compact=get_config_bool("insight.compact"),
        output_dir=get_config_str("insight.output_dir"),
        state_dir=get_config_str("insight.state_dir"),
        chat_history_path=get_config_str("insight.chat_history_path"),
        chat_history_limit=get_config_int("insight.chat_history_limit"),
        context_window_size=get_config_int("insight.context_window_size"),
        task_push_enabled=get_config_bool("insight.task_push_enabled"),
        detect_threshold=get_config_float("insight.detect_threshold"),
        step_trace_enabled=get_config_bool("insight.step_trace_enabled"),
        rag_enabled=get_config_bool("insight.rag_enabled"),
        rag_top_k=get_config_int("insight.rag_top_k"),
        rag_min_score=get_config_float("insight.rag_min_score"),
        rag_embed_model=get_config_str("insight.rag_embed_model"),
        observe_auto_reply_enabled=get_config_bool("insight.observe_auto_reply_enabled"),
        observe_ferment_threshold=get_config_float("insight.observe_ferment_threshold"),
        observe_logic1_base=get_config_float("insight.observe_logic1_base"),
        observe_logic2_base=get_config_float("insight.observe_logic2_base"),
        observe_logic3_base=get_config_float("insight.observe_logic3_base"),
        observe_force_non_observe_on_pop=get_config_bool("insight.observe_force_non_observe_on_pop"),
        max_workers=get_config_int("insight.online.max_workers"),
    )
)
