from __future__ import annotations

from typing import Any, Callable

from langgraph.config import get_stream_writer

# user_id -> callback that pushes an event to that user's open WebSocket(s).
# Used by background jobs that outlive the graph run (e.g. thin-record upgrades).
_notifiers: dict[str, Callable[[dict[str, Any]], None]] = {}


def register_notifier(user_id: str, fn: Callable[[dict[str, Any]], None]) -> None:
    _notifiers[user_id] = fn


def unregister_notifier(user_id: str, fn: Callable[[dict[str, Any]], None]) -> None:
    if _notifiers.get(user_id) is fn:
        _notifiers.pop(user_id, None)


def notify_user(user_id: str | None, event: dict[str, Any]) -> None:
    fn = _notifiers.get(user_id or "")
    if fn is not None:
        fn(event)


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
