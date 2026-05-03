from __future__ import annotations

import os
import re
import sys
import time
import atexit
import signal
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
import logging

import lark_oapi as lark

from utils import getenv, load_env_file

from feishu.apis import (
    BotProfile,
    FeishuBotGateway,
    FeishuEventListener,
    IncomingMessage,
    IncomingReaction,
    MessageGateway,
)
from llm.client import LLMClient, LLMConfig

SOFREE_CLI_SRC = Path(__file__).resolve().parents[1] / "sofree-knowledge-cli" / "src"
if SOFREE_CLI_SRC.exists():
    sys.path.insert(0, str(SOFREE_CLI_SRC))

from sofree_knowledge.confused_detector import (  # noqa: E402
    build_confused_judge_prompt,
    detect_confused_candidates,
    format_inline_explanation,
    parse_confused_judgement,
)

DEFAULT_CONFUSED_REACTION_KEYS = {
    "question",
    "question_mark",
    "what",
    "dizzy",
    "doubt",
    "confused",
    "疑问",
    "什么",
    "?",
    "？",
}


def resolve_env_file() -> str:
    configured = getenv("SOFREE_ENV_FILE", "").strip()
    if configured:
        return str(Path(configured).expanduser())
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / ".env",
        root.parent / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


class Bot:
    """A simple bot that listens for messages and replies in configurable ways."""

    def __init__(
        self,
        bot_id: str,
        display_name: str,
        role_name: str,
        role_prompt: str,
        app_id: str,
        app_secret: str,
        llm_api_key: str = "",
        llm_model_id: str = "",
        llm_base_url: str = "",
        reply_type: str = "echo",
        confused_feature_enabled: bool = True,
        confused_enabled_chat_ids: set[str] | None = None,
        confused_reaction_keys: set[str] | None = None,
        reply_on_normal_message: bool = False,
        confused_output_mode: str = "silent",
        gateway: MessageGateway | None = None,
        log_level: lark.LogLevel = lark.LogLevel.INFO,
    ) -> None:
        normalized_reply_type = reply_type.strip().lower()
        if normalized_reply_type not in {"echo", "llm"}:
            raise ValueError(
                "reply_type must be one of {'echo', 'llm'}, "
                f"got: {reply_type!r}"
            )

        self.profile = BotProfile(
            bot_id=bot_id,
            display_name=display_name,
            role_name=role_name,
            role_prompt=role_prompt,
            app_id=app_id,
            app_secret=app_secret,
            llm_api_key=llm_api_key,
            llm_model_id=llm_model_id,
            llm_base_url=llm_base_url,
        )
        self.reply_type = normalized_reply_type
        self.llm_client = LLMClient(
            LLMConfig.from_env(
                api_key=llm_api_key,
                model_id=llm_model_id,
                base_url=llm_base_url,
            )
        )
        self.gateway = gateway or FeishuBotGateway([self.profile])
        self.log_level = log_level
        self.listener: FeishuEventListener | None = None
        self.confused_feature_enabled = bool(confused_feature_enabled)
        self.confused_enabled_chat_ids = set(confused_enabled_chat_ids or set())
        self.confused_reaction_keys = set(
            confused_reaction_keys or DEFAULT_CONFUSED_REACTION_KEYS
        )
        self.reply_on_normal_message = bool(reply_on_normal_message)
        self.confused_output_mode = (
            "bot_reply" if str(confused_output_mode).strip().lower() == "bot_reply" else "silent"
        )
        self._chat_histories: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=80)
        )
        self._message_id_to_chat_id: dict[str, str] = {}
        self._seen_message_ids: set[str] = set()
        self._seen_message_order: deque[str] = deque(maxlen=4000)
        self._seen_reaction_keys: set[str] = set()
        self._seen_reaction_order: deque[str] = deque(maxlen=4000)
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        if self.listener is None:
            self.listener = FeishuEventListener(
                self.profile,
                self.handle_incoming_message,
                on_reaction_created=self.handle_reaction_created,
                log_level=self.log_level,
            )
        self.listener.start()

    def handle_incoming_message(self, message: IncomingMessage) -> None:
        self.logger.info(
            "message received: chat_id=%s message_id=%s sender_type=%s text=%s",
            message.chat_id,
            message.message_id,
            message.sender_type,
            message.text[:120],
        )
        if message.sender_type != "user":
            return
        if not message.text.strip():
            return
        if self._is_duplicate_message(message.message_id):
            self.logger.info("skip duplicate message event: message_id=%s", message.message_id)
            return

        command_reply = self._handle_command(message)
        if command_reply:
            self._send_text_with_log(message.chat_id, command_reply, source="command")
            return

        self._append_chat_history(message)
        inline_explain = self._try_confused_explain(message)
        if inline_explain:
            self._emit_confused_explain(chat_id=message.chat_id, explain_text=inline_explain, source="reply")
            return

        if not self.reply_on_normal_message:
            return

        reply = self.build_reply(message)
        self._send_text_with_log(message.chat_id, reply, source="normal_reply")

    def handle_reaction_created(self, reaction: IncomingReaction) -> None:
        self.logger.info(
            "reaction created: message_id=%s key=%s user=%s",
            reaction.message_id,
            reaction.reaction_key,
            reaction.user_open_id,
        )
        reaction_event_key = self._build_reaction_event_key(reaction)
        if self._is_duplicate_reaction(reaction_event_key):
            self.logger.info("skip duplicate reaction event: key=%s", reaction_event_key)
            return
        chat_id = self._message_id_to_chat_id.get(reaction.message_id, "")
        if not chat_id:
            self.logger.warning(
                "reaction ignored: unknown message_id in cache message_id=%s key=%s",
                reaction.message_id,
                reaction.reaction_key,
            )
            return
        if not self.confused_feature_enabled or chat_id not in self.confused_enabled_chat_ids:
            return
        history = list(self._chat_histories[chat_id])
        if not history:
            return
        tuned_keys = set(self.confused_reaction_keys or set())
        reaction_key = str(reaction.reaction_key or "").strip().lower()
        if _looks_confused_reaction_key(reaction_key):
            tuned_keys.add(reaction_key)
        candidates = detect_confused_candidates(
            messages=history,
            target_message_id=reaction.message_id,
            reactions=[
                {
                    "message_id": reaction.message_id,
                    "reaction_key": reaction.reaction_key,
                    "user_id": reaction.user_open_id,
                }
            ],
            confused_reaction_keys=tuned_keys or None,
            max_candidates=1,
        )
        if not candidates:
            return

        prompt = build_confused_judge_prompt(candidates[0])
        self.logger.info("reaction confused prompt: %s", prompt)
        raw = self.llm_client.build_reply("Output JSON object only.", prompt)
        self.logger.info("reaction confused llm raw: %s", raw)
        try:
            judgement = parse_confused_judgement(raw)
        except Exception:
            return
        self.logger.info("reaction confused judgement: %s", judgement)
        if not judgement.get("is_confused", False):
            return
        inline_text = format_inline_explanation(judgement.get("micro_explain", ""))
        if inline_text:
            self._emit_confused_explain(chat_id=chat_id, explain_text=inline_text, source="reaction")

    def _handle_command(self, message: IncomingMessage) -> str:
        text = message.text.strip()
        text = re.sub(r"@[\w\u4e00-\u9fff-]+", "", text).strip()

        commands = {
            "/开启混淆检测": "enable_chat",
            "/关闭混淆检测": "disable_chat",
            "/全局开启混淆检测": "enable_global",
            "/全局关闭混淆检测": "disable_global",
            "/混淆检测状态": "status",
            "/帮助": "help",
        }
        action = commands.get(text, "")
        if not action:
            return ""

        if action == "enable_chat":
            self.set_confused_enabled_for_chat(message.chat_id, True)
            return "已开启本群混淆检测。"
        if action == "disable_chat":
            self.set_confused_enabled_for_chat(message.chat_id, False)
            return "已关闭本群混淆检测。"
        if action == "enable_global":
            self.set_confused_feature_enabled(True)
            return "已全局开启混淆检测。"
        if action == "disable_global":
            self.set_confused_feature_enabled(False)
            return "已全局关闭混淆检测。"
        if action == "status":
            global_status = "开启" if self.confused_feature_enabled else "关闭"
            chat_status = "开启" if message.chat_id in self.confused_enabled_chat_ids else "关闭"
            enabled_chats = len(self.confused_enabled_chat_ids)
            return (
                f"混淆检测状态\n"
                f"- 全局：{global_status}\n"
                f"- 本群：{chat_status}\n"
                f"- 已开启群数：{enabled_chats}"
            )
        return (
            "可用命令:\n"
            "/开启混淆检测\n"
            "/关闭混淆检测\n"
            "/全局开启混淆检测\n"
            "/全局关闭混淆检测\n"
            "/混淆检测状态"
        )

    def build_reply(self, message: IncomingMessage) -> str:
        text = message.text.strip()
        if self.reply_type == "llm":
            self.logger.info("normal reply prompt(system): %s", self.profile.role_prompt)
            self.logger.info("normal reply prompt(user): %s", text)
            raw = self.llm_client.build_reply(self.profile.role_prompt, text)
            self.logger.info("normal reply llm raw: %s", raw)
            return raw
        return f"{self.profile.display_name} 收到消息：{text}"

    def _append_chat_history(self, message: IncomingMessage) -> None:
        self._chat_histories[message.chat_id].append(
            {
                "message_id": message.message_id,
                "sender": {"id": message.sender_open_id},
                "content": message.text,
            }
        )
        self._message_id_to_chat_id[message.message_id] = message.chat_id

    def _try_confused_explain(self, message: IncomingMessage) -> str:
        if not self.confused_feature_enabled:
            return ""
        if message.chat_id not in self.confused_enabled_chat_ids:
            return ""
        history = list(self._chat_histories[message.chat_id])
        if len(history) < 2:
            return ""

        candidates = detect_confused_candidates(
            messages=history,
            confused_reaction_keys=self.confused_reaction_keys or None,
            max_candidates=1,
        )
        if not candidates:
            return ""

        candidate = candidates[0]
        has_current_evidence = any(
            str(evidence.get("message_id", "")) == message.message_id
            for evidence in candidate.get("evidence", [])
            if isinstance(evidence, dict)
        )
        if not has_current_evidence:
            return ""

        prompt = build_confused_judge_prompt(candidate)
        self.logger.info("reply confused prompt: %s", prompt)
        raw = self.llm_client.build_reply("Output JSON object only.", prompt)
        self.logger.info("reply confused llm raw: %s", raw)
        try:
            judgement = parse_confused_judgement(raw)
        except Exception:
            return ""
        self.logger.info("reply confused judgement: %s", judgement)
        if not judgement.get("is_confused", False):
            return ""
        return format_inline_explanation(judgement.get("micro_explain", ""))

    def _emit_confused_explain(self, chat_id: str, explain_text: str, source: str) -> None:
        text = str(explain_text or "").strip()
        if not text:
            return
        if self.confused_output_mode == "bot_reply":
            self._send_text_with_log(chat_id, text, source=f"confused_{source}")
            self.logger.info(
                "confused explain sent: chat_id=%s source=%s text=%s",
                chat_id,
                source,
                text,
            )
            return
        self.logger.info(
            "confused explain generated (silent mode): chat_id=%s source=%s text=%s",
            chat_id,
            source,
            text,
        )

    def _send_text_with_log(self, chat_id: str, text: str, source: str) -> None:
        self.logger.info("send text: source=%s chat_id=%s text=%s", source, chat_id, text)
        self.gateway.send_text(self.profile.bot_id, chat_id, text)

    def _is_duplicate_message(self, message_id: str) -> bool:
        normalized = str(message_id or "").strip()
        if not normalized:
            return False
        if normalized in self._seen_message_ids:
            return True
        if len(self._seen_message_order) >= self._seen_message_order.maxlen:
            expired = self._seen_message_order.popleft()
            self._seen_message_ids.discard(expired)
        self._seen_message_ids.add(normalized)
        self._seen_message_order.append(normalized)
        return False

    def _build_reaction_event_key(self, reaction: IncomingReaction) -> str:
        message_id = str(reaction.message_id or "").strip()
        reaction_key = str(reaction.reaction_key or "").strip().lower()
        user_open_id = str(reaction.user_open_id or "").strip()
        action_time = str(reaction.action_time or "").strip()
        return f"{message_id}|{reaction_key}|{user_open_id}|{action_time}"

    def _is_duplicate_reaction(self, reaction_event_key: str) -> bool:
        normalized = str(reaction_event_key or "").strip()
        if not normalized:
            return False
        if normalized in self._seen_reaction_keys:
            return True
        if len(self._seen_reaction_order) >= self._seen_reaction_order.maxlen:
            expired = self._seen_reaction_order.popleft()
            self._seen_reaction_keys.discard(expired)
        self._seen_reaction_keys.add(normalized)
        self._seen_reaction_order.append(normalized)
        return False

    def set_confused_enabled_for_chat(self, chat_id: str, enabled: bool) -> set[str]:
        normalized_chat_id = str(chat_id or "").strip()
        if not normalized_chat_id:
            raise ValueError("chat_id is required")
        if enabled:
            self.confused_enabled_chat_ids.add(normalized_chat_id)
        else:
            self.confused_enabled_chat_ids.discard(normalized_chat_id)
        os.environ["SOFREE_CONFUSED_CHAT_IDS"] = format_chat_id_set(self.confused_enabled_chat_ids)
        return set(self.confused_enabled_chat_ids)

    def set_confused_feature_enabled(self, enabled: bool) -> None:
        self.confused_feature_enabled = bool(enabled)
        os.environ["SOFREE_CONFUSED_ENABLED"] = "1" if self.confused_feature_enabled else "0"


def parse_chat_id_set(raw: str) -> set[str]:
    values = [value.strip() for value in str(raw or "").split(",")]
    return {value for value in values if value}


def format_chat_id_set(values: set[str]) -> str:
    return ",".join(sorted({value.strip() for value in values if value and value.strip()}))


def parse_env_bool(raw: str, default: bool = True) -> bool:
    value = str(raw or "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _pid_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".sofree_bot.pid"


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(20):
        if not _is_process_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def ensure_single_instance(logger: logging.Logger) -> None:
    pid_file = _pid_file_path()
    current_pid = os.getpid()
    previous_pid = 0
    if pid_file.exists():
        try:
            previous_pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            previous_pid = 0
    if previous_pid and previous_pid != current_pid and _is_process_alive(previous_pid):
        logger.info("single-instance: terminating previous bot pid=%s", previous_pid)
        _terminate_process(previous_pid)
    pid_file.write_text(str(current_pid), encoding="utf-8")

    def _cleanup_pid_file() -> None:
        try:
            if pid_file.exists():
                content = pid_file.read_text(encoding="utf-8").strip()
                if content == str(current_pid):
                    pid_file.unlink(missing_ok=True)
        except OSError:
            return

    atexit.register(_cleanup_pid_file)


def _looks_confused_reaction_key(key: str) -> bool:
    if not key:
        return False
    return any(token in key for token in ("?", "？", "问", "什么", "what", "doubt", "confus", "dizzy"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[BOT] [%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    load_env_file(resolve_env_file())
    if parse_env_bool(getenv("SOFREE_SINGLE_INSTANCE", "1"), default=True):
        ensure_single_instance(logging.getLogger(__name__))

    role_prompt = "你是群聊助手。"

    listener_bot_config = {
        "bot_id": getenv("SOFREE_BOT_ID", "listener_host"),
        "display_name": getenv("SOFREE_BOT_DISPLAY_NAME", "listener_host"),
        "role_name": "listener",
        "role_prompt": role_prompt,
        "app_id": getenv("SOFREE_FEISHU_APP_ID") or getenv("FEISHU_APP_ID", ""),
        "app_secret": getenv("SOFREE_FEISHU_APP_SECRET") or getenv("FEISHU_APP_SECRET", ""),
        "llm_api_key": getenv("LLM_API_KEY", "") or getenv("ANTHROPIC_API_KEY", ""),
        "llm_model_id": getenv("LLM_MODEL_ID", "") or getenv("ANTHROPIC_ENDPOINT", ""),
        "llm_base_url": getenv("LLM_BASE_URL", "") or getenv("ANTHROPIC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "reply_type": getenv("SOFREE_BOT_REPLY_TYPE", "llm"),
        "reply_on_normal_message": parse_env_bool(
            getenv("SOFREE_REPLY_ON_NORMAL_MESSAGE", "0"),
            default=False,
        ),
        "confused_output_mode": getenv("SOFREE_CONFUSED_OUTPUT_MODE", "silent"),
        "confused_feature_enabled": parse_env_bool(
            getenv("SOFREE_CONFUSED_ENABLED", "1"),
            default=True,
        ),
        "confused_enabled_chat_ids": parse_chat_id_set(
            getenv("SOFREE_CONFUSED_CHAT_IDS", "oc_f482e00e55461a4d343f21334c9a96d7")
        ),
        "confused_reaction_keys": DEFAULT_CONFUSED_REACTION_KEYS,
        "gateway": None,
        "log_level": lark.LogLevel.INFO,
    }

    bot = Bot(**listener_bot_config)
    bot.start()
