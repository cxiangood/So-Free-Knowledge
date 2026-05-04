import json
from pathlib import Path

from sofree_knowledge.archive import collect_messages


class FakeClient:
    def list_visible_chats(self, page_size=100, page_token=""):
        return {
            "items": [{"chat_id": "oc_group", "name": "Group", "chat_mode": "group"}],
            "has_more": False,
            "page_token": "",
        }

    def list_chat_messages(
        self,
        chat_id,
        start_time="",
        end_time="",
        page_size=50,
        page_token="",
        sort="asc",
    ):
        return {
            "items": [
                {
                    "message_id": f"{chat_id}_m1",
                    "chat_id": chat_id,
                    "msg_type": "text",
                    "create_time": "1710000000",
                    "sender": {"id": "ou_user", "sender_type": "user"},
                    "body": {"content": '{"text":"你好"}'},
                }
            ],
            "has_more": False,
            "page_token": "",
        }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_collect_messages_writes_jsonl(tmp_path):
    manifest = collect_messages(client=FakeClient(), output_dir=tmp_path, output_subdir="run")

    messages = read_jsonl(tmp_path / "message_archive" / "run" / "messages.jsonl")
    assert manifest["message_count"] == 1
    assert messages[0]["content"] == "你好"


def test_collect_messages_can_only_collect_explicit_chat(tmp_path):
    manifest = collect_messages(
        client=FakeClient(),
        output_dir=tmp_path,
        output_subdir="run",
        chat_ids="oc_explicit",
        include_visible_chats=False,
    )

    messages = read_jsonl(tmp_path / "message_archive" / "run" / "messages.jsonl")
    assert manifest["chat_count"] == 1
    assert {message["chat_id"] for message in messages} == {"oc_explicit"}


def test_collect_messages_filters_raw_card_content(tmp_path):
    class CardClient(FakeClient):
        def list_chat_messages(
            self,
            chat_id,
            start_time="",
            end_time="",
            page_size=50,
            page_token="",
            sort="asc",
        ):
            return {
                "items": [
                    {
                        "message_id": f"{chat_id}_card",
                        "chat_id": chat_id,
                        "msg_type": "post",
                        "create_time": "1710000001",
                        "card_msg_content_type": "raw_card_content",
                        "sender": {"id": "ou_user", "sender_type": "user"},
                        "body": {"content": '{"title":"card"}'},
                    },
                    {
                        "message_id": f"{chat_id}_m1",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000002",
                        "sender": {"id": "ou_user", "sender_type": "user"},
                        "body": {"content": '{"text":"保留文本"}'},
                    },
                ],
                "has_more": False,
                "page_token": "",
            }

    manifest = collect_messages(client=CardClient(), output_dir=tmp_path, output_subdir="run")
    messages = read_jsonl(tmp_path / "message_archive" / "run" / "messages.jsonl")

    assert manifest["message_count"] == 1
    assert [message["message_id"] for message in messages] == ["oc_group_m1"]
