from __future__ import annotations

import json
import uuid
from pathlib import Path

import sofree_knowledge.chat_keyword_workflow as workflow


def _make_temp_dir() -> Path:
    base = Path.cwd() / "pytest_tmp_chat_keyword_workflow"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"case_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_messages_jsonl(path: Path, contents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, text in enumerate(contents, start=1):
            record = {
                "message_id": f"m{index}",
                "chat_id": "oc_test",
                "msg_type": "text",
                "content": text,
                "create_time": "1710000000000",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_workflow_reuses_existing_sheet_and_appends_rows(monkeypatch):
    tmp_path = _make_temp_dir()
    messages_file = tmp_path / "message_archive" / "run1" / "messages.jsonl"
    _write_messages_jsonl(messages_file, ["上线风险提示", "发布节奏确认"])

    monkeypatch.setattr(
        workflow,
        "collect_messages",
        lambda **kwargs: {
            "run_id": "run1",
            "archive_dir": str(messages_file.parent),
            "files": {"messages": str(messages_file)},
        },
    )

    monkeypatch.setattr(
        workflow,
        "_load_message_extract_functions",
        lambda: (
            lambda _: [{"msg_type": "text", "content": "上线风险提示"}, {"msg_type": "text", "content": "发布节奏确认"}],
            lambda records, include_types=None: [str(item.get("content") or "") for item in records],
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_load_classify_function",
        lambda: (
            lambda data, config=None: {
                "top_keywords": ["上线"],
                "classification_results": {
                    "上线": [
                        {
                            "type": "key",
                            "sense": "发布上线动作",
                            "contexts": ["上线风险提示"],
                            "count": 1,
                            "ratio": 1.0,
                        }
                    ]
                },
            }
        ),
    )

    append_calls: list[dict] = []
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_append_data",
        lambda args: append_calls.append(
            {
                "spreadsheet_token": args.spreadsheet_token,
                "range": args.range,
                "values": json.loads(args.values),
            }
        )
        or {"ok": True},
    )
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_create_sheet",
        lambda args: (_ for _ in ()).throw(AssertionError("should reuse existing sheet")),
    )

    monkeypatch.setattr(workflow, "_sheet_appears_empty", lambda client, token, sid: True)
    monkeypatch.setattr(workflow, "_get_first_sheet_id", lambda client, token: "sheet_1")

    class FakeClient:
        def list_drive_files(self, page_size=50, page_token=""):
            return {
                "items": [
                    {
                        "name": "chat_keywords_oc_test",
                        "type": "sheet",
                        "token": "sht_existing",
                        "url": "https://example.com/sheets/sht_existing",
                    }
                ],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr(workflow, "FeishuClient", lambda: FakeClient())

    result = workflow.run_chat_keyword_to_wikisheet_workflow(chat_id="oc_test", output_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["created_or_reused"] == "reused"
    assert result["spreadsheet_token"] == "sht_existing"
    assert result["rows_written"] == 1
    assert result["header_appended"] is True
    assert len(append_calls) == 2
    assert append_calls[0]["values"] == [workflow.HEADER_COLUMNS]
    assert append_calls[1]["values"][0][3] == "上线"


def test_workflow_creates_sheet_when_missing_and_appends(monkeypatch):
    tmp_path = _make_temp_dir()
    messages_file = tmp_path / "message_archive" / "run2" / "messages.jsonl"
    _write_messages_jsonl(messages_file, ["需求评审推进"])

    monkeypatch.setattr(
        workflow,
        "collect_messages",
        lambda **kwargs: {
            "run_id": "run2",
            "archive_dir": str(messages_file.parent),
            "files": {"messages": str(messages_file)},
        },
    )
    monkeypatch.setattr(
        workflow,
        "_load_message_extract_functions",
        lambda: (
            lambda _: [{"msg_type": "text", "content": "需求评审推进"}],
            lambda records, include_types=None: [str(records[0]["content"])],
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_load_classify_function",
        lambda: (
            lambda data, config=None: {
                "top_keywords": ["评审"],
                "classification_results": {
                    "评审": [{"type": "key", "sense": "流程评审", "contexts": ["需求评审推进"], "count": 1, "ratio": 1.0}]
                },
            }
        ),
    )
    monkeypatch.setattr(workflow, "_sheet_appears_empty", lambda client, token, sid: False)
    monkeypatch.setattr(workflow, "_get_first_sheet_id", lambda client, token: "sheet_new")

    class FakeClient:
        def list_drive_files(self, page_size=50, page_token=""):
            return {"items": [], "has_more": False, "page_token": ""}

    monkeypatch.setattr(workflow, "FeishuClient", lambda: FakeClient())

    create_calls: list[dict] = []
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_create_sheet",
        lambda args: create_calls.append({"title": args.title, "space_id": args.space_id})
        or {"ok": True, "spreadsheet_token": "sht_created", "url": "https://example.com/sheets/sht_created"},
    )
    append_calls: list[dict] = []
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_append_data",
        lambda args: append_calls.append({"token": args.spreadsheet_token, "values": json.loads(args.values)}) or {"ok": True},
    )

    result = workflow.run_chat_keyword_to_wikisheet_workflow(
        chat_id="oc_test",
        output_dir=str(tmp_path),
        space_id="my_library",
    )
    assert result["created_or_reused"] == "created"
    assert result["spreadsheet_token"] == "sht_created"
    assert create_calls and create_calls[0]["title"] == "chat_keywords_oc_test"
    assert len(append_calls) == 1
    assert append_calls[0]["values"][0][3] == "评审"


def test_workflow_flattens_classification_results_to_expected_rows():
    rows = workflow._flatten_classification_results(
        run_id="run_flat",
        chat_id="oc_flat",
        sheet_title="chat_keywords_oc_flat",
        classification_results={
            "上线": [
                {"type": "key", "sense": "发布动作", "contexts": ["上线提示"], "count": 2, "ratio": 0.66},
                {"type": "confused", "sense": "未确定", "contexts": ["上线流程"], "count": 1, "ratio": 0.34},
            ]
        },
        source_message_count=6,
        created_at_utc="2026-04-27T00:00:00+00:00",
    )
    assert len(rows) == 2
    first = rows[0]
    assert first[:6] == ["run_flat", "oc_flat", "chat_keywords_oc_flat", "上线", "key", "发布动作"]
    assert first[9] == 6
    assert json.loads(first[8]) == ["上线提示"]


def test_workflow_handles_empty_plain_messages_gracefully(monkeypatch):
    tmp_path = _make_temp_dir()
    messages_file = tmp_path / "message_archive" / "run3" / "messages.jsonl"
    _write_messages_jsonl(messages_file, ["系统消息"])

    monkeypatch.setattr(
        workflow,
        "collect_messages",
        lambda **kwargs: {
            "run_id": "run3",
            "archive_dir": str(messages_file.parent),
            "files": {"messages": str(messages_file)},
        },
    )
    monkeypatch.setattr(
        workflow,
        "_load_message_extract_functions",
        lambda: (
            lambda _: [{"msg_type": "system", "content": "系统消息"}],
            lambda records, include_types=None: [],
        ),
    )
    monkeypatch.setattr(workflow, "_sheet_appears_empty", lambda client, token, sid: False)
    monkeypatch.setattr(workflow, "_get_first_sheet_id", lambda client, token: "sheet_1")

    class FakeClient:
        def list_drive_files(self, page_size=50, page_token=""):
            return {
                "items": [{"name": "chat_keywords_oc_test", "type": "sheet", "token": "sht_empty"}],
                "has_more": False,
                "page_token": "",
            }

    monkeypatch.setattr(workflow, "FeishuClient", lambda: FakeClient())

    called = {"append": 0}
    monkeypatch.setattr(
        workflow,
        "_load_classify_function",
        lambda: (_ for _ in ()).throw(AssertionError("classify should not be called for empty plain messages")),
    )
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_append_data",
        lambda args: called.__setitem__("append", called["append"] + 1) or {"ok": True},
    )

    result = workflow.run_chat_keyword_to_wikisheet_workflow(chat_id="oc_test", output_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["plain_message_count"] == 0
    assert result["rows_written"] == 0
    assert called["append"] == 0


def test_workflow_local_integration_with_fixture_messages(monkeypatch):
    tmp_path = _make_temp_dir()
    # Use real load_records + extract_plain_messages + classify; mock Feishu writes only.
    messages_file = tmp_path / "message_archive" / "run4" / "messages.jsonl"
    _write_messages_jsonl(
        messages_file,
        [
            "上线风险需要提前评估",
            "发布上线前要完成评审",
            "风险控制和回滚预案要同步",
        ],
    )

    monkeypatch.setattr(
        workflow,
        "collect_messages",
        lambda **kwargs: {
            "run_id": "run4",
            "archive_dir": str(messages_file.parent),
            "files": {"messages": str(messages_file)},
        },
    )
    monkeypatch.setattr(workflow, "_get_first_sheet_id", lambda client, token: "sheet_1")
    monkeypatch.setattr(workflow, "_sheet_appears_empty", lambda client, token, sid: True)

    class FakeClient:
        def list_drive_files(self, page_size=50, page_token=""):
            return {"items": [], "has_more": False, "page_token": ""}

    monkeypatch.setattr(workflow, "FeishuClient", lambda: FakeClient())
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_create_sheet",
        lambda args: {"ok": True, "spreadsheet_token": "sht_integration", "url": "https://example.com/sheets/sht_integration"},
    )
    append_calls: list[dict] = []
    monkeypatch.setattr(
        workflow.wikisheet_module,
        "cmd_append_data",
        lambda args: append_calls.append({"values": json.loads(args.values)}) or {"ok": True},
    )

    result = workflow.run_chat_keyword_to_wikisheet_workflow(chat_id="oc_test", output_dir=str(tmp_path))
    assert result["ok"] is True
    assert result["created_or_reused"] == "created"
    assert result["spreadsheet_token"] == "sht_integration"
    assert result["plain_message_count"] == 3
    assert result["rows_written"] >= 1
    assert len(append_calls) >= 2  # header + data
    assert append_calls[0]["values"] == [workflow.HEADER_COLUMNS]
