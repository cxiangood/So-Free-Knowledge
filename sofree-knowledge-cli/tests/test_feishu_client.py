import json

import pytest

from sofree_knowledge.feishu_client import FeishuClient
from sofree_knowledge.feishu_client import MissingFeishuConfigError


def test_format_feishu_error_includes_chat_membership_hint():
    client = FeishuClient(user_access_token="u_token")
    message = client._format_feishu_error(  # noqa: SLF001 - validate user-facing error text
        {
            "code": 230002,
            "msg": "Bot/User can NOT be out of the chat.",
            "error": {"log_id": "log_xxx"},
        },
        status_code=400,
    )

    assert "code=230002" in message
    assert "hint=Bot/User is not in the target chat." in message
    assert "log_id=log_xxx" in message


def test_format_network_error_for_winerror_10013():
    client = FeishuClient(user_access_token="u_token")

    class Win10013Error(OSError):
        @property
        def winerror(self):  # type: ignore[override]
            return 10013

    err = Win10013Error(10013, "denied")

    message = client._format_network_error(err)  # noqa: SLF001 - validate user-facing error text

    assert "WinError 10013" in message


def test_constructor_does_not_auto_load_user_token(monkeypatch):
    monkeypatch.setattr("sofree_knowledge.feishu_client.get_user_access_token", lambda token_file=None: "loaded_token")

    client = FeishuClient()

    assert client.user_access_token is None


def test_from_user_context_loads_token_file(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"access_token": "u_loaded"}, ensure_ascii=False), encoding="utf-8")

    client = FeishuClient.from_user_context(token_file=token_file)

    assert client.user_access_token == "u_loaded"


def test_from_user_context_can_require_user_token(tmp_path):
    with pytest.raises(MissingFeishuConfigError):
        FeishuClient.from_user_context(token_file=tmp_path / "missing.json", require_user_token=True)
