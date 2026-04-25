from __future__ import annotations

from plan.models import PlanRecord


PLAN_RULES = """\
你现在按 plan-manager 规则工作：
1. 先把目标拆成 Plan、Milestone、Task、Risk。
2. 写入飞书前必须先给我预览。
3. 计划正文优先保存到飞书文档。
4. 可执行事项优先创建飞书任务。
5. 结构化状态需要长期维护时，再创建飞书多维表格。
6. 会议和评审节点才创建日历。
7. 涉及指派别人、发消息、建会议、批量修改时，必须先确认。
8. 每次完成后返回创建或更新的飞书链接。
"""


def build_create_draft_prompt(record: PlanRecord, context: str = "") -> str:
    return f"""[PLAN_COMMAND]
command: create_draft
plan_id: {record.plan_id}
title: {record.title}
write_policy: preview_only
output_format: markdown
[/PLAN_COMMAND]

{PLAN_RULES}

目标：
{record.goal}

补充上下文：
{context or "无"}

请只输出计划草案，不要创建飞书文档、任务、多维表格或日历。
草案必须包含：Goal、Assumptions、Milestones、Tasks、Risks、Open Questions。
""".strip()


def build_materialize_prompt(record: PlanRecord, approved_scope: str = "doc_and_tasks") -> str:
    return f"""[PLAN_COMMAND]
command: materialize
plan_id: {record.plan_id}
title: {record.title}
write_policy: confirm_before_write
approved_scope: {approved_scope}
[/PLAN_COMMAND]

{PLAN_RULES}

请基于上一版已确认的计划草案执行落地。

执行前先列出你将创建或更新的内容，包括：
- 飞书文档标题和主要章节
- 飞书任务列表、负责人、截止时间
- 是否需要多维表格或日历

在我确认之前，不要实际创建或修改任何飞书内容。
""".strip()


def build_status_prompt(record: PlanRecord) -> str:
    return f"""[PLAN_COMMAND]
command: status
plan_id: {record.plan_id}
title: {record.title}
write_policy: read_only
[/PLAN_COMMAND]

{PLAN_RULES}

请汇总这个计划的当前状态。
如果你能访问关联的飞书文档、任务或多维表格，请读取后总结。
如果不能访问，请基于当前会话上下文说明缺少哪些链接或权限。
""".strip()
