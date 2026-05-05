from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from .auth_token_manager import TokenManager
from .auth_token_store import TokenStore
from .config import DEFAULT_TOKEN_FILE, get_app_credentials
from .feishu_client import FeishuAPIError, MissingFeishuConfigError


DEFAULT_REDIRECT_URI = "http://localhost:8000/callback"
DEFAULT_SCOPE = (
    "im:chat:read im:message:readonly "
    "im:message.group_msg:get_as_user im:message.p2p_msg:get_as_user "
    "search:message contact:contact.base:readonly "
    "drive:file:readonly docs:doc:readonly wiki:node:read wiki:space:read"
)


def build_authorization_url(
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    scope: str = DEFAULT_SCOPE,
    state: str | None = None,
) -> str:
    app_id, _ = get_app_credentials()
    if not app_id:
        raise MissingFeishuConfigError("APP_ID is required to build an authorization URL.")
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state or secrets.token_urlsafe(16),
    }
    return "https://accounts.feishu.cn/open-apis/authen/v1/authorize?" + urlencode(params)


def exchange_code_for_token(
    code_or_url: str,
    token_file: str | Path | None = None,
) -> dict[str, Any]:
    code = extract_code(code_or_url)
    app_access_token = get_app_access_token()
    response = httpx.post(
        "https://open.feishu.cn/open-apis/authen/v1/access_token",
        json={"grant_type": "authorization_code", "code": code},
        headers={"Authorization": f"Bearer {app_access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code", 0) not in (0, None):
        raise FeishuAPIError(f"Feishu token exchange failed: {data}")
    token_data = data.get("data", data)
    save_token(token_data, token_file=token_file)
    return redact_token_data(token_data)


def init_token(
    enable_autofill: bool = False,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    scope: str = DEFAULT_SCOPE,
    token_file: str | Path | None = None,
) -> dict[str, Any]:
    state = secrets.token_urlsafe(16)
    url = build_authorization_url(redirect_uri=redirect_uri, scope=scope, state=state)
    if enable_autofill:
        code = capture_code_from_local_callback(
            expected_state=state,
            redirect_uri=redirect_uri,
            scope=scope,
        )
    else:
        webbrowser.open(url)
        return {
            "authorization_url": url,
            "next": "Open the URL, copy the redirected URL or code, then run exchange-code.",
        }
    return exchange_code_for_token(code, token_file=token_file)


def auth_status(token_file: str | Path | None = None) -> dict[str, Any]:
    store = TokenStore(token_file=token_file)
    path = store.path
    exists = path.exists()
    result: dict[str, Any] = store.status()
    if not exists:
        return result
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        result["parseable_json"] = False
        result["has_access_token"] = '"access_token"' in raw
        result["has_refresh_token"] = '"refresh_token"' in raw
        return result
    result["parseable_json"] = True
    result["has_access_token"] = bool(data.get("access_token"))
    result["has_refresh_token"] = bool(data.get("refresh_token"))
    for key in ("token_type", "expires_in", "refresh_expires_in", "open_id", "user_id", "tenant_key"):
        if key in data:
            result[key] = data[key]
    if "scope" in data:
        result["scope"] = data["scope"]
    return result


def ensure_user_auth(
    required_scopes: str | list[str] | tuple[str, ...] = (),
    *,
    token_file: str | Path | None = None,
) -> dict[str, Any]:
    return TokenManager(token_file=token_file).ensure_user_auth(required_scopes=required_scopes)


def get_app_access_token() -> str:
    app_id, app_secret = get_app_credentials()
    if not app_id or not app_secret:
        raise MissingFeishuConfigError("APP_ID and SECRET_ID are required.")
    response = httpx.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code", 0) not in (0, None):
        raise FeishuAPIError(f"Feishu app_access_token failed: {data}")
    token = data.get("app_access_token") or data.get("data", {}).get("app_access_token")
    if not token:
        raise FeishuAPIError(f"Feishu app_access_token response missing token: {data}")
    return str(token)


def save_token(token_data: dict[str, Any], token_file: str | Path | None = None) -> Path:
    return TokenStore(token_file=token_file).save(token_data)


def extract_code(code_or_url: str) -> str:
    value = code_or_url.strip()
    parsed = urlparse(value)
    if parsed.scheme and parsed.query:
        params = parse_qs(parsed.query)
        error = params.get("error", [""])[0]
        if error:
            description = params.get("error_description", [""])[0]
            raise FeishuAPIError(f"Feishu authorization failed: {error} {description}".strip())
        value = params.get("code", [""])[0]
    if not value:
        raise ValueError("Authorization code is empty.")
    return value


def capture_code_from_local_callback(
    expected_state: str,
    redirect_uri: str,
    scope: str,
) -> str:
    redirect = urlparse(redirect_uri)
    host = redirect.hostname or "localhost"
    port = redirect.port or 80
    path = redirect.path or "/callback"
    result: dict[str, str] = {}
    ready = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path != path:
                self.send_response(404)
                self.end_headers()
                return
            if params.get("state", [""])[0] != expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state")
                ready.set()
                return
            code = params.get("code", [""])[0]
            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code")
                ready.set()
                return
            result["code"] = code
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization captured. You can close this page.")
            ready.set()

    server = HTTPServer((host, port), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    webbrowser.open(build_authorization_url(redirect_uri=redirect_uri, scope=scope, state=expected_state))
    ready.wait(timeout=180)
    server.shutdown()
    thread.join(timeout=5)
    if "code" not in result:
        raise TimeoutError("Timed out waiting for Feishu authorization callback.")
    return result["code"]


def redact_token_data(token_data: dict[str, Any]) -> dict[str, Any]:
    result = {
        "token_file": str(DEFAULT_TOKEN_FILE),
        "has_access_token": bool(token_data.get("access_token")),
        "has_refresh_token": bool(token_data.get("refresh_token")),
    }
    for key in ("token_type", "expires_in", "refresh_expires_in", "open_id", "user_id", "tenant_key"):
        if key in token_data:
            result[key] = token_data[key]
    return result
