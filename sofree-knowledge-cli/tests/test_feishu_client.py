import json

import pytest
import httpx

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


def test_request_refreshes_user_token_on_401(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(
        json.dumps(
            {
                "access_token": "expired_token",
                "refresh_token": "refresh_token_value",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = FeishuClient(user_access_token="expired_token", token_file=token_file)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, path, headers=None, **kwargs):
            self.calls += 1
            auth = (headers or {}).get("Authorization")
            if self.calls == 1:
                return httpx.Response(401, json={"code": 99991663, "msg": "invalid access token"})
            assert auth == "Bearer refreshed_user_token"
            return httpx.Response(200, json={"code": 0, "data": {"ok": True}})

    def fake_post(url, json=None, headers=None, timeout=30):
        assert url.endswith("/open-apis/authen/v1/refresh_access_token")
        assert json == {"grant_type": "refresh_token", "refresh_token": "refresh_token_value"}
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "code": 0,
                "data": {
                    "access_token": "refreshed_user_token",
                    "refresh_token": "refresh_token_value_2",
                },
            },
        )

    monkeypatch.setattr("sofree_knowledge.feishu_client.httpx.Client", FakeClient)
    monkeypatch.setattr("sofree_knowledge.feishu_client.httpx.post", fake_post)
    monkeypatch.setattr(FeishuClient, "get_app_access_token", lambda self: "app_token")

    result = client.request("GET", "/open-apis/im/v1/chats")

    assert result["data"]["ok"] is True
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved["access_token"] == "refreshed_user_token"
    assert saved["refresh_token"] == "refresh_token_value_2"
