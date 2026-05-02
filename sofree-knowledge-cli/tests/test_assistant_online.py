from sofree_knowledge.assistant_online import collect_online_personal_inputs


from sofree_knowledge.feishu_client import FeishuAPIError


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

    def list_file_view_records(self, file_token, file_type="docx", page_size=50, page_token=""):
        return {
            "items": [
                {"viewer_id": "ou_target", "view_time": "1710000000"},
                {"viewer_id": "ou_other", "view_time": "1710000001"},
            ],
            "has_more": False,
            "page_token": "",
        }

    def list_file_comments(self, file_token, file_type="docx", page_size=50, page_token="", is_solved=None):
        return {
            "items": [
                {
                    "reply_list": {
                        "replies": [
                            {"create_user_id": "ou_target"},
                            {"create_user_id": "ou_other"},
                        ]
                    }
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
        recent_days=3650,
    )

    assert out["resolved_target_user_id"] == "ou_target"
    assert out["meta"]["message_count"] == 0
    assert out["messages"] == []
    assert any(item.get("doc_id") == "abc123" for item in out["documents"])
    actions = {(item["doc_id"], item["action"], item.get("count", 0)) for item in out["access_records"]}
    assert ("abc123", "share", 1) in actions
    assert ("abc123", "view", 1) in actions
    assert ("abc123", "comment", 1) in actions
    assert out["knowledge_items"] == []


def test_collect_online_personal_inputs_filters_noise_messages(monkeypatch):
    class NoiseClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "n1",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_target"}},
                        "body": {"content": '{"text":"LLM 调用失败: 429 Too Many Requests"}'},
                    },
                    {
                        "message_id": "n2",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000001",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"客户需求变更，今晚上线风险较高"}'},
                    },
                    {
                        "message_id": "n3",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000002",
                        "sender": {"sender_id": {"open_id": "ou_target"}},
                        "body": {"content": '{"text":"{\\"json_card\\":\\"...\\",\\"meta\\":{}}"}'},
                    },
                    {
                        "message_id": "n4",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000003",
                        "sender": {"sender_id": {"open_id": "ou_target"}},
                        "body": {"content": '{"text":"@SoFree 请生成一份午饭计划，要求如下：1. 2. 3."}'},
                    },
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=NoiseClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )
    message_ids = [item.get("message_id") for item in out["messages"]]
    assert "n1" not in message_ids
    assert "n2" in message_ids
    assert "n3" not in message_ids
    assert "n4" not in message_ids


def test_collect_online_personal_inputs_applies_recent_days(monkeypatch):
    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=FakeClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=7,
    )
    assert out["meta"]["recent_days"] == 7
    assert out["messages"] == []
    assert out["documents"] == []


def test_collect_online_personal_inputs_builds_message_url_when_user_chat_visible(monkeypatch):
    class UserTokenClient(FakeClient):
        user_access_token = "u-token"

        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "m1",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"请看文档 https://foo.feishu.cn/docx/abc123 今天截止"}'},
                    }
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=UserTokenClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )
    assert out["messages"]
    message_url = str(out["messages"][0].get("message_url") or "")
    assert message_url.startswith("https://applink.feishu.cn/client/chat/open?")
    assert "openChatId=oc_1" in message_url


def test_collect_online_personal_inputs_builds_message_url_without_user_token(monkeypatch):
    class OtherSenderClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "m1",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"请看文档 https://foo.feishu.cn/docx/abc123 今天截止"}'},
                    }
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=OtherSenderClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )
    assert out["messages"]
    message_url = str(out["messages"][0].get("message_url") or "")
    assert message_url.startswith("https://applink.feishu.cn/client/chat/open?")
    assert "openChatId=oc_1" in message_url


def test_collect_online_personal_inputs_filters_self_messages_by_sender_id(monkeypatch):
    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})

    out = collect_online_personal_inputs(
        client=FakeClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )

    assert out["messages"] == []
    assert out["access_records"]


def test_collect_online_personal_inputs_resolves_sender_name_with_tenant_fallback(monkeypatch):
    class TenantFallbackClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "m_sender",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"请尽快确认发布排期"}'},
                    }
                ],
                "has_more": False,
                "page_token": "",
            }

        def get_tenant_access_token(self):
            return "tenant_token"

        def request(self, method, path, access_token=None, **kwargs):
            if access_token is None:
                raise FeishuAPIError("401")
            return {
                "data": {
                    "user": {
                        "name": "曹林江",
                        "avatar": {"avatar_72": "https://example/avatar.png"},
                    }
                }
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})

    out = collect_online_personal_inputs(
        client=TenantFallbackClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )

    assert out["messages"]
    assert out["messages"][0]["sender_name"] == "曹林江"


def test_collect_online_personal_inputs_resolves_wiki_title_from_get_node(monkeypatch):
    class WikiClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "m_wiki",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"请查看 https://foo.feishu.cn/wiki/TZn0wSoSabc123"}'},
                    }
                ],
                "has_more": False,
                "page_token": "",
            }

        def list_drive_files(self, page_size=50, page_token=""):
            return {"items": [], "has_more": False, "page_token": ""}

        def request(self, method, path, access_token=None, params=None, **kwargs):
            if path == "/open-apis/wiki/v2/spaces/get_node":
                assert params == {"token": "TZn0wSoSabc123"}
                return {
                    "data": {
                        "node": {
                            "title": "关键词提取排期",
                            "url": "https://foo.feishu.cn/wiki/TZn0wSoSabc123",
                        }
                    }
                }
            raise FeishuAPIError(f"unexpected path: {path}")

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})

    out = collect_online_personal_inputs(
        client=WikiClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )

    wiki_docs = [item for item in out["documents"] if item.get("doc_id") == "TZn0wSoSabc123"]
    assert wiki_docs
    assert wiki_docs[0]["title"] == "关键词提取排期"


def test_collect_online_personal_inputs_marks_only_targeted_mentions(monkeypatch):
    class MentionClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "m_target",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "content": "@ou_target 请确认一下",
                        "body": {"content": '{"text":"@ou_target 请确认一下"}'},
                    },
                    {
                        "message_id": "m_other",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000001",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "content": "@ou_someone_else 请确认一下",
                        "body": {"content": '{"text":"@ou_someone_else 请确认一下"}'},
                    },
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=MentionClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )

    by_id = {item["message_id"]: item for item in out["messages"]}
    assert by_id["m_target"]["mentions_target_user"] is True
    assert by_id["m_other"]["mentions_target_user"] is False


def test_collect_online_personal_inputs_skips_system_messages(monkeypatch):
    class SystemClient(FakeClient):
        def list_chat_messages(self, chat_id, start_time="", end_time="", page_size=50, page_token="", sort="asc"):
            return {
                "items": [
                    {
                        "message_id": "sys_1",
                        "chat_id": chat_id,
                        "msg_type": "system",
                        "create_time": "1710000000",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {
                            "content": '{"template":"{from_user} invited {to_chatters} to the group.","from_user":["A"],"to_chatters":["B"]}'
                        },
                    },
                    {
                        "message_id": "txt_1",
                        "chat_id": chat_id,
                        "msg_type": "text",
                        "create_time": "1710000001",
                        "sender": {"sender_id": {"open_id": "ou_other"}},
                        "body": {"content": '{"text":"飞书知识推送今晚处理"}'},
                    },
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr("sofree_knowledge.assistant_online.get_user_identity", lambda token_file=None: {"open_id": "ou_target"})
    out = collect_online_personal_inputs(
        client=SystemClient(),
        target_user_id="",
        include_visible_chats=True,
        max_chats=5,
        max_messages_per_chat=20,
        max_drive_docs=10,
        recent_days=3650,
    )

    message_ids = [item.get("message_id") for item in out["messages"]]
    assert "sys_1" not in message_ids
    assert "txt_1" in message_ids
