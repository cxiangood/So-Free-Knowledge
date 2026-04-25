import argparse
from pathlib import Path
from typing import List, Optional

import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from message_archive.extract_chat_messages import extract_plain_messages, load_records
from token_classify.domain_tokenizer import tokenize_text


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

    records = load_records(path)
    plain_messages = extract_plain_messages(records, include_types=include_types)
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
