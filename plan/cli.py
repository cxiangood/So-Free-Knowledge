from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feishu.openclaw_controller import send_openclaw_message
from plan.models import PlanRecord
from plan.prompt_builder import build_create_draft_prompt, build_materialize_prompt, build_status_prompt
from plan.store import PlanStore


def slugify(text: str) -> str:
    asciiish = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return asciiish[:48] or "plan"


def cmd_create_draft(args: argparse.Namespace) -> None:
    plan_id = args.plan_id or f"plan_{slugify(args.title or args.goal)}"
    record = PlanRecord(plan_id=plan_id, title=args.title or args.goal[:40], goal=args.goal, owner=args.owner or "")
    prompt = build_create_draft_prompt(record, context=args.context or "")
    if args.print_only:
        print(prompt)
        return
    message_id = send_openclaw_message(prompt)
    record.last_openclaw_message_id = message_id
    PlanStore().save(record)
    print(f"sent message_id={message_id}")
    print(f"saved plan_id={record.plan_id}")


def cmd_materialize(args: argparse.Namespace) -> None:
    record = PlanStore().load(args.plan_id)
    prompt = build_materialize_prompt(record, approved_scope=args.scope)
    if args.print_only:
        print(prompt)
        return
    message_id = send_openclaw_message(prompt)
    record.last_openclaw_message_id = message_id
    PlanStore().save(record)
    print(f"sent message_id={message_id}")


def cmd_status(args: argparse.Namespace) -> None:
    record = PlanStore().load(args.plan_id)
    prompt = build_status_prompt(record)
    if args.print_only:
        print(prompt)
        return
    message_id = send_openclaw_message(prompt)
    record.last_openclaw_message_id = message_id
    PlanStore().save(record)
    print(f"sent message_id={message_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Controller for OpenClaw over Feishu messages.")
    sub = parser.add_subparsers(required=True)

    create = sub.add_parser("create-draft", help="Create and send a preview-only plan draft command.")
    create.add_argument("--goal", required=True)
    create.add_argument("--title")
    create.add_argument("--owner")
    create.add_argument("--context")
    create.add_argument("--plan-id")
    create.add_argument("--print-only", action="store_true")
    create.set_defaults(func=cmd_create_draft)

    materialize = sub.add_parser("materialize", help="Ask OpenClaw to prepare Feishu write operations.")
    materialize.add_argument("plan_id")
    materialize.add_argument("--scope", default="doc_and_tasks")
    materialize.add_argument("--print-only", action="store_true")
    materialize.set_defaults(func=cmd_materialize)

    status = sub.add_parser("status", help="Ask OpenClaw for a plan status summary.")
    status.add_argument("plan_id")
    status.add_argument("--print-only", action="store_true")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
