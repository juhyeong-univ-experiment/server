from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer


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


def emit_progress(node: str):
    """Return an emitter that streams events immediately (LangGraph 'custom' stream mode)."""
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    def _emit(status: str, data: dict[str, Any]) -> None:
        if writer is not None:
            writer(make_event(status, node, data.get("message", ""), data))

    return _emit
