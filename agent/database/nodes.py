from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent.base.events import make_event
from agent.base.states import AgentState
from database import fetch_books_by_ids, search_similar_items

logger = logging.getLogger(__name__)

def memory_exclude_ids(profile: dict[str, Any] | None) -> list[str]:
    """Books the user already disliked/read or was already recommended are not recommended again."""
    profile = profile or {}
    ids = [e.get("book_id") for key in ("disliked", "liked", "recommended") for e in profile.get(key) or []]
    return [i for i in dict.fromkeys(ids) if i]


async def vector_db_node(state: AgentState) -> dict[str, Any]:
    intention = state.get("intention") or {}
    query_text = intention.get("query_text") or state.get("request_text") or ""
    top_k = 10
    error = None
    started = time.perf_counter()
    exclude_ids = memory_exclude_ids(state.get("user_profile"))
    try:
        result = await asyncio.to_thread(search_similar_items, query_text, top_k, exclude_ids)
        result["excluded_by_memory"] = len(exclude_ids)
    except Exception as exc:
        # Fail-safe: keep pipeline alive with empty result and explicit error payload.
        error = str(exc)
        result = {"query": query_text, "items": []}
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "node=vector_db elapsed_ms=%.2f top_k=%d items=%d error=%s",
        elapsed_ms,
        top_k,
        len(result.get("items", [])),
        bool(error),
    )

    return {
        "vdb_result": result,
        "events": [
            make_event(
                "TOOL_DONE",
                "vector_db",
                "벡터 데이터베이스에서 관련 도서를 찾았어요." if not error else "벡터 데이터베이스 조회 중 오류가 발생했어요.",
                {**result, "error": error},
            )
        ],
    }


async def database_node(state: AgentState) -> dict[str, Any]:
    ids = [item["id"] for item in (state.get("vdb_result") or {}).get("items", [])]
    score_by_id = {
        str(item.get("id")): item.get("score")
        for item in (state.get("vdb_result") or {}).get("items", [])
        if item.get("id") is not None
    }
    error = None
    started = time.perf_counter()
    try:
        rows = await asyncio.to_thread(fetch_books_by_ids, ids)
    except Exception as exc:
        error = str(exc)
        rows = []
    elapsed_ms = (time.perf_counter() - started) * 1000

    enriched_rows = []
    for row in rows:
        item_id = str(row.get("id"))
        enriched_rows.append({**row, "score": score_by_id.get(item_id)})
    logger.info(
        "node=database elapsed_ms=%.2f request_ids=%d fetched_rows=%d error=%s",
        elapsed_ms,
        len(ids),
        len(enriched_rows),
        bool(error),
    )

    return {
        "db_result": enriched_rows,
        "book_recommendation": enriched_rows,
        "events": [
            make_event(
                "TOOL_DONE",
                "database",
                "데이터베이스에서 도서 상세 정보를 가져왔어요." if not error else "데이터베이스 조회 중 오류가 발생했어요.",
                {"count": len(enriched_rows), "items": enriched_rows, "error": error},
            )
        ],
    }
