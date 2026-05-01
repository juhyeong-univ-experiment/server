from __future__ import annotations

from typing import Any


def make_event(
    status: str,
    node: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "node": node,
        "message": message,
        "data": data or {},
    }
