from sofree_knowledge.assistant_online import collect_online_personal_inputs


class FakeClient:
    def list_visible_chats(self, page_size=100, page_token=""):
        return {
            "items": [{"chat_id": "oc_1", "name": "Group"}],
            "has_more": False,
            "page_token": "",
        }

    def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
        return {
            "items": [
                {
                    "message_id": "m1",
                    "chat_id": chat_id,
                    "msg_type": "text",
                    "create_time": "1710000000",
                    "sender": {"sender_id": {"open_id": "ou_target"}},
                    "body": {"content": '{"text":"请看文档 https://foo.feishu.cn/docx/abc123 今天截止"}'},
                }
            ],
            "has_more": False,
            "page_token": "",
        }

    def list_drive_files(self, page_size=50, page_token=""):
        return {
            "items": [
                {
                    "token": "abc123",
                    "name": "发布流程",
                    "url": "https://foo.feishu.cn/docx/abc123",
                    "modified_time": "1710000000",
                }
            ],
            "has_more": False,
            "page_token": "",
        }


def test_collect_online_personal_inputs_builds_documents_and_access_records(monkeypatch):
    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})

    out = collect_online_personal_inputs(
        client=FakeClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
    )

    assert out["resolved_target_user_id"] == "ou_target"
    assert out["meta"]["message_count"] == 1
    assert any(item.get("doc_id") == "abc123" for item in out["documents"])
    assert out["access_records"][0]["doc_id"] == "abc123"
    assert out["knowledge_items"]
