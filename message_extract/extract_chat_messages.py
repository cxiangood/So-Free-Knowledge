import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_records(file_path: Path) -> List[Dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                return [item for item in data["messages"] if isinstance(item, dict)]
            return [data]

    raise ValueError("只支持 .json 或 .jsonl 文件")


def format_create_time(create_time: Any) -> str:
    if create_time is None:
        return ""
    value = str(create_time).strip()
    if not value:
        return ""

    try:
        ms = int(value)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone()
        return dt.isoformat(timespec="seconds")
    except (ValueError, OSError, OverflowError):
        return value


def extract_text_from_post_payload(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    content = payload.get("content", [])
    if not isinstance(content, list):
        return ""

    for block in content:
        if not isinstance(block, list):
            continue
        segs: List[str] = []
        for item in block:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    segs.append(text)
        line = "".join(segs).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_message_text(record: Dict[str, Any]) -> str:
    msg_type = str(record.get("msg_type", "")).lower()

    content = record.get("content")
    if isinstance(content, str) and content.strip() and msg_type == "text":
        return content.strip()

    raw_content = record.get("raw_content")
    parsed_raw: Optional[Dict[str, Any]] = None
    if isinstance(raw_content, str) and raw_content.strip():
        try:
            loaded = json.loads(raw_content)
            if isinstance(loaded, dict):
                parsed_raw = loaded
        except json.JSONDecodeError:
            parsed_raw = None

    if parsed_raw:
        if msg_type == "text":
            text = parsed_raw.get("text")
            if isinstance(text, str):
                return text.strip()

        if msg_type == "post":
            return extract_text_from_post_payload(parsed_raw)

        if msg_type == "system":
            template = parsed_raw.get("template")
            if isinstance(template, str):
                return template.strip()

        # interactive / other types: fallback to common fields
        for key in ("text", "title"):
            value = parsed_raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(content, str) and content.strip():
        return content.strip()

    return ""


def replace_mention_keys(text: str, mentions: List[Dict[str, Any]]) -> str:
    result = text
    for item in mentions:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        name = item.get("name")
        if isinstance(key, str) and key and isinstance(name, str) and name:
            result = result.replace(key, f"@{name}")
    return result


def build_global_mention_map(records: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for record in records:
        mentions_raw = record.get("mentions")
        if not isinstance(mentions_raw, list):
            continue
        for item in mentions_raw:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            name = item.get("name")
            if isinstance(key, str) and key and isinstance(name, str) and name:
                mapping[key] = name
    return mapping


def replace_by_global_mapping(text: str, mapping: Dict[str, str]) -> str:
    result = text
    for key, name in mapping.items():
        result = result.replace(key, f"@{name}")
    return result


def extract_plain_messages(
    records: List[Dict[str, Any]],
    include_types: Optional[set[str]] = None,
) -> List[str]:
    print("正在提取纯聊天内容列表...")
    include_types = include_types or {"text", "post"}
    global_mention_map = build_global_mention_map(records)
    messages: List[str] = []
    for record in records:
        msg_type = str(record.get("msg_type", "")).lower()
        if msg_type not in include_types:
            continue
        text = extract_message_text(record)
        if text:
            text = replace_by_global_mapping(text, global_mention_map)
            messages.append(text)
    return messages


def extract_messages_with_metadata(
    records: List[Dict[str, Any]],
    include_types: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    print("正在提取带元数据的聊天内容列表...")
    include_types = include_types or {"text", "post"}
    global_mention_map = build_global_mention_map(records)
    items: List[Dict[str, Any]] = []
    for record in records:
        msg_type = str(record.get("msg_type", "")).lower()
        if msg_type not in include_types:
            continue

        text = extract_message_text(record)
        if not text:
            continue

        mentions_raw = record.get("mentions")
        mentions = [m for m in mentions_raw if isinstance(m, dict)] if isinstance(mentions_raw, list) else []
        mention_names = [m.get("name") for m in mentions if isinstance(m.get("name"), str)]
        mention_names = [name for name in mention_names if name]

        replaced_text = replace_mention_keys(text, mentions)
        replaced_text = replace_by_global_mapping(replaced_text, global_mention_map)

        sender_raw = record.get("sender")
        sender: Dict[str, Any] = sender_raw if isinstance(sender_raw, dict) else {}
        sender_label = ""
        if isinstance(sender.get("name"), str) and sender.get("name"):
            sender_label = sender["name"]
        elif isinstance(sender.get("id"), str):
            sender_label = sender["id"]

        items.append(
            {
                "sender": sender_label,
                "send_time": format_create_time(record.get("create_time")),
                "mentions": mention_names,
                "content": replaced_text,
            }
        )

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="提取聊天记录内容")
    parser.add_argument("file_path", help="message.json 或 message.jsonl 文件路径")
    parser.add_argument(
        "--only",
        choices=["plain", "meta", "both"],
        default="both",
        help="输出类型：plain=纯文本列表，meta=带元数据列表，both=都输出",
    )
    parser.add_argument(
        "--include-types",
        default="text,post",
        help="按消息类型过滤，逗号分隔。默认: text,post",
    )
    args = parser.parse_args()

    path = Path(args.file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    records = load_records(path)
    include_types = {x.strip().lower() for x in args.include_types.split(",") if x.strip()}
    if not include_types:
        include_types = {"text", "post"}

    if args.only in ("plain", "both"):
        plain = extract_plain_messages(records, include_types=include_types)
        print("=== 纯聊天内容列表 ===")
        print(json.dumps(plain, ensure_ascii=False, indent=2))

    if args.only in ("meta", "both"):
        meta = extract_messages_with_metadata(records, include_types=include_types)
        print("=== 带元数据的聊天内容列表 ===")
        print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
