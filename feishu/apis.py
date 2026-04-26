from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BotProfile:
    bot_id: str
    display_name: str
    role_name: str
    role_prompt: str
    app_id: str
    app_secret: str
    is_listener: bool = False
    speak_order: int = 0
    llm_api_key: str = ""
    llm_model_id: str = ""
    llm_base_url: str = ""


@dataclass(slots=True)
class IncomingMessage:
    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    sender_type: str
    text: str
    mentioned_open_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IncomingReaction:
    message_id: str
    reaction_key: str
    operator_type: str
    user_open_id: str
    action_time: str


class MessageGateway(Protocol):
    def send_text(self, bot_id: str, receive_id: str, text: str, receive_id_type: str = "chat_id") -> str:
        raise NotImplementedError


class FeishuAPIError(RuntimeError):
    """Raised when Feishu OpenAPI returns an error."""


class FeishuBotGateway:
    def __init__(self, bots: list[BotProfile]) -> None:
        self._clients = {
            bot.bot_id: lark.Client.builder().app_id(bot.app_id).app_secret(bot.app_secret).build()
            for bot in bots
        }

    def send_text(self, bot_id: str, receive_id: str, text: str, receive_id_type: str = "chat_id") -> str:
        if bot_id not in self._clients:
            raise FeishuAPIError(f"Unknown bot_id: {bot_id}")

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            )
            .build()
        )

        response = self._clients[bot_id].im.v1.message.create(request)
        if not response.success():
            raise FeishuAPIError(
                f"Failed to send message with {bot_id}: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )
        if response.data is None:
            return ""
        return getattr(response.data, "message_id", "") or ""


class FeishuEventListener:
    def __init__(
        self,
        listener_bot: BotProfile,
        on_message: Callable[[IncomingMessage], None],
        on_reaction_created: Callable[[IncomingReaction], None] | None = None,
        log_level: lark.LogLevel = lark.LogLevel.INFO,
    ) -> None:
        self._on_message = on_message
        self._on_reaction_created = on_reaction_created
        self._event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_receive)
            .register_p2_im_message_reaction_created_v1(self._handle_reaction_created)
            .register_p2_im_message_reaction_deleted_v1(self._handle_reaction_deleted)
            .build()
        )
        self._client = lark.ws.Client(
            listener_bot.app_id,
            listener_bot.app_secret,
            event_handler=self._event_handler,
            log_level=log_level,
        )

    def start(self) -> None:
        self._client.start()

    def _handle_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        try:
            LOGGER.info("event received: event_type=im.message.receive_v1")
            event = data.event
            incoming = IncomingMessage(
                message_id=event.message.message_id,
                chat_id=event.message.chat_id,
                chat_type=event.message.chat_type,
                sender_open_id=getattr(event.sender.sender_id, "open_id", "") or "",
                sender_type=event.sender.sender_type,
                text=parse_text_content(event.message.content),
                mentioned_open_ids=[
                    getattr(mention.id, "open_id", "") or ""
                    for mention in (event.message.mentions or [])
                ],
            )
            self._on_message(incoming)
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to process Feishu incoming event.")

    def _handle_reaction_created(self, data) -> None:  # type: ignore[no-untyped-def]
        LOGGER.info("event received: event_type=im.message.reaction.created_v1")
        if self._on_reaction_created is None:
            return
        try:
            event = data.event
            reaction_type = getattr(event, "reaction_type", None)
            reaction_key = (
                getattr(reaction_type, "emoji_type", "")
                or getattr(reaction_type, "key", "")
                or ""
            )
            user_id = getattr(event, "user_id", None)
            incoming = IncomingReaction(
                message_id=getattr(event, "message_id", "") or "",
                reaction_key=str(reaction_key or ""),
                operator_type=getattr(event, "operator_type", "") or "",
                user_open_id=getattr(user_id, "open_id", "") or "",
                action_time=getattr(event, "action_time", "") or "",
            )
            self._on_reaction_created(incoming)
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to process Feishu reaction created event.")

    def _handle_reaction_deleted(self, data) -> None:  # type: ignore[no-untyped-def]
        try:
            event = data.event
            message_id = getattr(event, "message_id", "") or ""
            reaction_type = getattr(event, "reaction_type", None)
            reaction_key = (
                getattr(reaction_type, "emoji_type", "")
                or getattr(reaction_type, "key", "")
                or ""
            )
            LOGGER.info(
                "event received: event_type=im.message.reaction.deleted_v1 message_id=%s key=%s",
                message_id,
                reaction_key,
            )
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to process Feishu reaction deleted event.")


def parse_text_content(raw_content: str) -> str:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content
    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])
    return raw_content
