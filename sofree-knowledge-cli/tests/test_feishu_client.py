from sofree_knowledge.feishu_client import FeishuClient


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
