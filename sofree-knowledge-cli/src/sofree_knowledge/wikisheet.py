from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .config import load_env_file, resolve_env_file
from .feishu_client import FeishuAPIError, FeishuClient, MissingFeishuConfigError
from .logging_config import configure_logging, get_logger


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sofree_knowledge.wikisheet",
        description=(
            "Wiki Sheet manager: create sheet in Wiki knowledge base, "
            "append/update/delete sheet data, and delete sheet."
        ),
    )
    parser.add_argument("--env-file", default="", help="Path to .env file.")
    parser.add_argument("--output-dir", default=".", help="Used for .env auto-discovery.")
    parser.add_argument("--log-level", default="", help="Logging level. Defaults to config.yaml logging.level.")
    parser.add_argument("--log-file", default="", help="Optional path for persistent logs. Defaults to config.yaml logging.log_file.")
    parser.add_argument("--quiet", action="store_true", help="Disable terminal logs; file logs still work when --log-file is set.")
    subparsers = parser.add_subparsers(dest="command")

    create = subparsers.add_parser("create-sheet", help="Create a sheet node in Wiki.")
    create.add_argument("--title", required=True, help="Sheet title.")
    create.add_argument("--space-id", default="", help="Wiki space id. Supports literal 'my_library'.")
    create.add_argument("--parent-node-token", default="", help="Create under this parent wiki node token.")
    create.set_defaults(func=cmd_create_sheet)

    append = subparsers.add_parser("append-data", help="Append rows to a sheet.")
    append.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    append.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1 or <sheetId>!A1:D10.")
    append.add_argument("--values", required=True, help="2D JSON array.")
    append.set_defaults(func=cmd_append_data)

    update = subparsers.add_parser("update-data", help="Overwrite values in a range.")
    update.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    update.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1:D10.")
    update.add_argument("--values", required=True, help="2D JSON array.")
    update.set_defaults(func=cmd_update_data)

    delete_data = subparsers.add_parser("delete-data", help="Delete data in a range (clear values).")
    delete_data.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    delete_data.add_argument("--range", required=True, help="Target range, e.g. <sheetId>!A1:D10.")
    delete_data.set_defaults(func=cmd_delete_data)

    delete_sheet = subparsers.add_parser("delete-sheet", help="Delete the spreadsheet file.")
    delete_sheet.add_argument("--spreadsheet-token", required=True, help="Spreadsheet token.")
    delete_sheet.set_defaults(func=cmd_delete_sheet)
    return parser


def _load_env(args: argparse.Namespace) -> None:
    env_path = resolve_env_file(args.env_file, output_dir=args.output_dir)
    load_env_file(env_path)


def _parse_values_json(raw: str) -> list[list[Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--values must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or any(not isinstance(row, list) for row in parsed):
        raise ValueError("--values must be a 2D JSON array, e.g. [[\"A\",1],[\"B\",2]]")
    return parsed


def _resolve_identity_token(client: FeishuClient, args: argparse.Namespace) -> tuple[str | None, str]:
    identity = str(getattr(args, "identity", "auto") or "auto").strip().lower()
    if identity not in {"user", "bot", "auto"}:
        raise ValueError("--identity must be one of: user, bot, auto")
    if identity == "bot":
        return client.get_tenant_access_token(), "bot"
    if identity == "user":
        if not client.user_access_token:
            raise MissingFeishuConfigError(
                "Missing user access token. Run sofree-knowledge init-token/exchange-code first."
            )
        return None, "user"
    if client.user_access_token:
        return None, "user"
    return client.get_tenant_access_token(), "bot_fallback"


def _request(
    client: FeishuClient,
    method: str,
    path: str,
    *,
    access_token: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    if access_token is None:
        return client.request(method, path, **kwargs)
    return client.request(method, path, access_token=access_token, **kwargs)


def _resolve_space(
    client: FeishuClient,
    *,
    space_id: str,
    parent_node_token: str,
    access_token: str | None,
) -> tuple[str, str, dict[str, Any] | None]:
    normalized_space_id = str(space_id or "").strip()
    normalized_parent = str(parent_node_token or "").strip()
    parent_node: dict[str, Any] | None = None
    parent_space_id = ""

    if normalized_parent:
        parent_data = _request(
            client,
            "GET",
            "/open-apis/wiki/v2/spaces/get_node",
            access_token=access_token,
            params={"token": normalized_parent},
        )
        body = parent_data.get("data", parent_data)
        parent_node = body.get("node", body) if isinstance(body, dict) else {}
        if not isinstance(parent_node, dict):
            parent_node = {}
        parent_space_id = str(parent_node.get("space_id") or "").strip()
        if not parent_space_id:
            raise FeishuAPIError("Failed to resolve space_id from parent_node_token")

    resolved_by = ""
    resolved_space_id = normalized_space_id

    if normalized_space_id.lower() == "my_library":
        data = _request(
            client,
            "GET",
            "/open-apis/wiki/v2/spaces/my_library",
            access_token=access_token,
        )
        body = data.get("data", data)
        space_obj = body.get("space", body) if isinstance(body, dict) else {}
        resolved_space_id = str((space_obj or {}).get("space_id") or (space_obj or {}).get("id") or "").strip()
        if not resolved_space_id:
            raise FeishuAPIError("Failed to resolve 'my_library' to real wiki space_id")
        resolved_by = "my_library"
    elif normalized_space_id:
        resolved_by = "explicit_space_id"
    elif parent_space_id:
        resolved_space_id = parent_space_id
        resolved_by = "parent_node_token"
    else:
        data = _request(
            client,
            "GET",
            "/open-apis/wiki/v2/spaces/my_library",
            access_token=access_token,
        )
        body = data.get("data", data)
        space_obj = body.get("space", body) if isinstance(body, dict) else {}
        resolved_space_id = str((space_obj or {}).get("space_id") or (space_obj or {}).get("id") or "").strip()
        if not resolved_space_id:
            raise FeishuAPIError(
                "Missing target space. Provide --space-id/--parent-node-token or ensure my_library is available."
            )
        resolved_by = "my_library_fallback"

    if parent_space_id and resolved_space_id and parent_space_id != resolved_space_id:
        raise ValueError(
            f"space mismatch: --space-id resolves to {resolved_space_id}, "
            f"but parent_node_token belongs to {parent_space_id}"
        )
    return resolved_space_id, resolved_by, parent_node


def cmd_create_sheet(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args)
    client = FeishuClient()
    access_token, identity = _resolve_identity_token(client, args)
    title = str(args.title or "").strip()
    if not title:
        raise ValueError("--title is required")
    resolved_space_id, resolved_by, parent_node = _resolve_space(
        client,
        space_id=args.space_id,
        parent_node_token=args.parent_node_token,
        access_token=access_token,
    )

    payload: dict[str, Any] = {"title": title, "obj_type": "sheet", "node_type": "origin"}
    parent_node_token = str(args.parent_node_token or "").strip()
    if parent_node_token:
        payload["parent_node_token"] = parent_node_token

    data = _request(
        client,
        "POST",
        f"/open-apis/wiki/v2/spaces/{resolved_space_id}/nodes",
        access_token=access_token,
        json=payload,
    )
    body = data.get("data", data)
    node = body.get("node", body) if isinstance(body, dict) else {}
    if not isinstance(node, dict):
        node = {}
    return {
        "ok": True,
        "action": "create-sheet",
        "identity": identity,
        "title": title,
        "resolved_space_id": resolved_space_id,
        "resolved_by": resolved_by,
        "parent_node_token": parent_node_token,
        "node_token": str(node.get("node_token") or node.get("token") or ""),
        "obj_type": str(node.get("obj_type") or "sheet"),
        "spreadsheet_token": str(node.get("obj_token") or ""),
        "url": str(node.get("url") or ""),
        "parent_node": parent_node,
        "raw": node,
    }


def cmd_append_data(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args)
    client = FeishuClient()
    access_token, identity = _resolve_identity_token(client, args)
    values = _parse_values_json(args.values)
    token = str(args.spreadsheet_token or "").strip()
    range_name = str(args.range or "").strip()
    data = _request(
        client,
        "POST",
        f"/open-apis/sheets/v2/spreadsheets/{token}/values_append",
        access_token=access_token,
        json={
            "valueRange": {"range": range_name, "values": values},
            "valueInputOption": "USER_ENTERED",
            "insertDataOption": "INSERT_ROWS",
        },
    )
    body = data.get("data", data)
    return {
        "ok": True,
        "action": "append-data",
        "identity": identity,
        "spreadsheet_token": token,
        "range": range_name,
        "raw": body,
    }


def cmd_update_data(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args)
    client = FeishuClient()
    access_token, identity = _resolve_identity_token(client, args)
    values = _parse_values_json(args.values)
    token = str(args.spreadsheet_token or "").strip()
    range_name = str(args.range or "").strip()
    data = _request(
        client,
        "PUT",
        f"/open-apis/sheets/v2/spreadsheets/{token}/values",
        access_token=access_token,
        json={"valueRange": {"range": range_name, "values": values}},
    )
    body = data.get("data", data)
    return {
        "ok": True,
        "action": "update-data",
        "identity": identity,
        "spreadsheet_token": token,
        "range": range_name,
        "raw": body,
    }


def cmd_delete_data(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args)
    client = FeishuClient()
    access_token, identity = _resolve_identity_token(client, args)
    token = str(args.spreadsheet_token or "").strip()
    range_name = str(args.range or "").strip()

    attempts: list[tuple[str, str, dict[str, Any]]] = [
        ("DELETE", f"/open-apis/sheets/v2/spreadsheets/{token}/values", {"valueRange": {"range": range_name}}),
        ("DELETE", f"/open-apis/sheets/v2/spreadsheets/{token}/values", {"range": range_name}),
        ("POST", f"/open-apis/sheets/v2/spreadsheets/{token}/values_clear", {"range": range_name}),
        ("POST", f"/open-apis/sheets/v2/spreadsheets/{token}/values_clear", {"valueRange": {"range": range_name}}),
    ]
    last_error = ""
    for method, path, payload in attempts:
        try:
            data = _request(
                client,
                method,
                path,
                access_token=access_token,
                json=payload,
            )
            body = data.get("data", data)
            return {
                "ok": True,
                "action": "delete-data",
                "identity": identity,
                "spreadsheet_token": token,
                "range": range_name,
                "resolved_endpoint": f"{method} {path}",
                "raw": body,
            }
        except Exception as exc:
            last_error = str(exc)

    raise FeishuAPIError(
        "delete-data failed after trying multiple Sheets clear endpoints. "
        f"last_error={last_error}"
    )


def cmd_delete_sheet(args: argparse.Namespace) -> dict[str, Any]:
    _load_env(args)
    client = FeishuClient()
    access_token, identity = _resolve_identity_token(client, args)
    token = str(args.spreadsheet_token or "").strip()
    data = _request(
        client,
        "DELETE",
        f"/open-apis/drive/v1/files/{token}",
        access_token=access_token,
    )
    body = data.get("data", data)
    return {"ok": True, "action": "delete-sheet", "identity": identity, "spreadsheet_token": token, "raw": body}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        configure_logging(
            level=args.log_level,
            log_file=args.log_file,
            app_name="SOFREE-WIKISHEET",
            quiet=args.quiet,
            force=True,
        )
        LOGGER.debug("running command: %s", args.command)
        result = args.func(args)
    except Exception as exc:
        if logging.getLogger().handlers:
            LOGGER.exception("command failed")
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
