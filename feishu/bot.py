from __future__ import annotations

import lark_oapi as lark

from feishu.apis import FeishuBotGateway, FeishuEventListener, MessageGateway,BotProfile,IncomingMessage
from llm.client import LLMClient, LLMConfig


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
            LLMConfig(
                api_key=llm_api_key,
                model_id=llm_model_id,
                base_url=llm_base_url,
            )
        )
        self.gateway = gateway or FeishuBotGateway([self.profile])
        self.log_level = log_level
        self.listener: FeishuEventListener | None = None

    def start(self) -> None:
        if self.listener is None:
            self.listener = FeishuEventListener(
                self.profile,
                self.handle_incoming_message,
                log_level=self.log_level,
            )
        self.listener.start()

    def handle_incoming_message(self, message: IncomingMessage) -> None:
        if message.sender_type != "user":
            return
        if not message.text.strip():
            return

        reply = self.build_reply(message)
        self.gateway.send_text(self.profile.bot_id, message.chat_id, reply)

    def build_reply(self, message: IncomingMessage) -> str:
        text = message.text.strip()
        if self.reply_type == "llm":
            return self.llm_client.build_reply(self.profile.role_prompt, text)
        return f"{self.profile.display_name} 收到消息：{text}"


if __name__ == "__main__":

    role_prompt = \
    """
    You are the top-tier coordinator & supervisor in the multi-agent collaboration system.
    Core Responsibilities
    Decompose overall objectives into clear, actionable subtasks.
    Assign tasks rationally to subordinate agents based on their capabilities and boundaries.
    Arrange execution sequence: serial/parallel workflow, control overall progress.
    Mediate conflicts, unify logic and standardize output when agent results diverge.
    Review content quality, correct errors, reject unqualified deliverables for revision.
    Integrate, summarize and polish all sub-agent outputs into a complete, coherent final result.
    Only undertake management, scheduling and review; do not replace professional agents to execute detailed work.
    Behavior Rules
    Issue precise, unambiguous instructions with clear deliverance requirements.
    Keep logic rigorous, communication efficient, no redundant dialogue.
    Proactively collect key information if missing; ensure full-task closed-loop delivery.
    """

    listener_bot_config = {
        "bot_id": "listener_host",
        "display_name": "listener_host",
        "role_name": "listener",
        "role_prompt": role_prompt,
        "app_id": "",
        "app_secret": "",
        "llm_api_key": "",
        "llm_model_id": "",
        "llm_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "reply_type": "llm",
        "gateway": None,
        "log_level": lark.LogLevel.INFO,
    }

    bot = Bot(**listener_bot_config)
    bot.start()
