from __future__ import annotations

import lark_oapi as lark

from feishu.apis import (
    BotProfile,
    FeishuBotGateway,
    FeishuEventListener,
    IncomingMessage,
    MessageGateway,
)
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
        )
        self.reply_type = normalized_reply_type
        self.llm_client = LLMClient(LLMConfig.from_env(base_url=llm_base_url))
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
    你在这个多智能体协作系统中，担任最高级别的协调者与管理者。
    核心职责
    将整体目标拆解为清晰、可执行的子任务。
    根据各下属智能体的能力与边界，合理分配任务。
    安排执行顺序：串行或并行的工作流程，掌控整体进度。
    当智能体的产出出现分歧时，调解冲突、统一逻辑并规范输出标准。
    审核内容质量，纠正错误，驳回不合格的成果要求其修改。
    整合、归纳并润色所有子智能体的输出，形成完整、连贯的最终成果。
    仅承担管理、调度与审核工作，不替代专业智能体执行具体任务。
    行为准则
    发布精准、不含糊的指令，并明确交付要求。
    保持逻辑严谨、沟通高效，不进行冗余对话。
    若关键信息缺失，主动收集；确保全任务闭环交付。
    """

    listener_bot_config = {
        "bot_id": "listener_host",
        "display_name": "listener_host",
        "role_name": "listener",
        "role_prompt": role_prompt,
        "app_id": "cli_a960422c68f81cc8",
        "app_secret": "9IKbo6gY4rvBLjV9oB7t0cC18pYLTHXo",
        "llm_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "reply_type": "llm",
        "gateway": None,
        "log_level": lark.LogLevel.INFO,
    }

    bot = Bot(**listener_bot_config)
    bot.start()
