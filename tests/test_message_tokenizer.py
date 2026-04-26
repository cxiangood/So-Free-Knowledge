import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from token_classify.domain_tokenizer import tokenize_text


def load_extract_chat_module():
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "message_archive" / "extract_chat_messages.py",
        project_root / "message_extract" / "extract_chat_messages.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("extract_chat_messages", path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        "extract_chat_messages.py not found under message_archive/ or message_extract/."
    )


def resolve_default_message_file() -> Optional[Path]:
    candidates = [
        Path("message_archive/current_visible/messages.jsonl"),
        Path("message_archive/current_visible/messages.json"),
        Path("message_archive/current_visible/chats.jsonl"),
        Path("message_archive/current_visible/chats.json"),
    ]
    for path in candidates:
        if path.exists():
            return path

    archive_root = Path("message_archive")
    if archive_root.exists():
        timestamp_dirs = [p for p in archive_root.iterdir() if p.is_dir()]
        timestamp_dirs.sort(key=lambda p: p.name, reverse=True)
        for folder in timestamp_dirs:
            for name in ("messages.jsonl", "messages.json", "chats.jsonl", "chats.json"):
                candidate = folder / name
                if candidate.exists():
                    return candidate
    return None


def parse_custom_terms(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    default_file = resolve_default_message_file()
    parser = argparse.ArgumentParser(
        description="Extract plain chat messages, then tokenize with domain_tokenizer."
    )
    parser.add_argument(
        "--file",
        default=str(default_file) if default_file else None,
        help="Path to message json/jsonl file. Default: message_archive/current_visible/messages.jsonl",
    )
    parser.add_argument(
        "--include-types",
        default="text,post",
        help="Comma separated message types for extract_plain_messages.",
    )
    parser.add_argument(
        "--custom-terms",
        default="",
        help="Comma separated custom terms for tokenizer.",
    )
    parser.add_argument("--ngram-min-count", type=int, default=2)
    parser.add_argument("--ngram-min-pmi", type=float, default=4.0)
    parser.add_argument("--ngram-max-n", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many messages to print. Use -1 to print all.",
    )
    args = parser.parse_args()

    if not args.file:
        raise FileNotFoundError("No default message file found. Please provide --file.")

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Message file not found: {path}")

    include_types = {x.strip().lower() for x in args.include_types.split(",") if x.strip()}
    if not include_types:
        include_types = {"text", "post"}

    extract_module = load_extract_chat_module()
    records = extract_module.load_records(path)
    plain_messages = extract_module.extract_plain_messages(records, include_types=include_types)
    custom_terms = parse_custom_terms(args.custom_terms)

    output_messages = plain_messages if args.limit < 0 else plain_messages[: args.limit]
    merged_text = "\n".join(output_messages)
    tokens = tokenize_text(
        merged_text,
        custom_terms=custom_terms,
        ngram_min_count=args.ngram_min_count,
        ngram_min_pmi=args.ngram_min_pmi,
        ngram_max_n=args.ngram_max_n,
    )

    print("[Merged Original Text]")
    print(merged_text)
    print("-" * 100)
    print("[Merged Tokens]")
    print(tokens)


if __name__ == "__main__":
    main()
