import json

from sofree_knowledge.auth_device_flow import has_required_scopes
from sofree_knowledge.auth import DEFAULT_SCOPE, auth_status, build_authorization_url, extract_code, save_token


def test_build_authorization_url_uses_env(monkeypatch):
    monkeypatch.setenv("APP_ID", "cli_test")

    url = build_authorization_url(
        redirect_uri="http://localhost:8000/callback",
        scope="im:chat:read",
        state="state123",
    )

    assert "client_id=cli_test" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fcallback" in url
    assert "scope=im%3Achat%3Aread" in url
    assert "state=state123" in url


def test_extract_code_accepts_redirect_url():
    code = extract_code("http://localhost:8000/callback?code=abc&state=state123")

    assert code == "abc"


def test_auth_status_redacts_token(tmp_path):
    token_file = tmp_path / "token.json"
    save_token(
        {
            "access_token": "u-secret",
            "refresh_token": "r-secret",
            "token_type": "Bearer",
            "open_id": "ou_test",
        },
        token_file=token_file,
    )

    status = auth_status(token_file=token_file)
    status_text = json.dumps(status)

    assert status["has_access_token"] is True
    assert status["has_refresh_token"] is True
    assert status["open_id"] == "ou_test"
    assert "u-secret" not in status_text
    assert "r-secret" not in status_text


def test_has_required_scopes_matches_space_separated_scopes():
    assert has_required_scopes("im:chat:read drive:file:read offline_access", "im:chat:read")
    assert has_required_scopes(
        "im:chat:read drive:file:read offline_access",
        ["im:chat:read", "drive:file:read"],
    )
    assert not has_required_scopes("im:chat:read offline_access", "drive:file:read")


def test_default_scope_uses_existing_contact_scope_name():
    assert "contact:contact.base:readonly" in DEFAULT_SCOPE
    assert "contact:user.base:readonly" in DEFAULT_SCOPE
