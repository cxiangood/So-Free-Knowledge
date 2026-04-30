from __future__ import annotations

from typing import Any

from ..assistant_online import collect_online_personal_inputs


def collect_personal_inputs_online(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return collect_online_personal_inputs(*args, **kwargs)
