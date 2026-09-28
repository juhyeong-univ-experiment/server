from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from agent.base.events import emit_progress, make_event, notify_user
from agent.base.states import AgentState
from agent.database.nodes import memory_exclude_ids
from agent.enrichment.pipeline import (
    enrich_existing_book,
    enrich_new_book,
    is_thin,
    run_limited,
    suggest_titles,
)
from agent.memory.profile_view import profile_to_prompt
from database import fetch_books_by_ids, get_raw_book, search_similar_items

logger = logging.getLogger(__name__)

LOW_SCORE_THRESHOLD = float(os.getenv("ENRICH_SCORE_THRESHOLD", "0.40"))
MIN_RESULTS = 3
TOP_K = 10
_background: set[asyncio.Task] = set()


def _schedule_background(coro) -> None:
    task = asyncio.ensure_future(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def enrichment_node(state: AgentState) -> dict[str, Any]:
    """Detect missing / weak coverage and grow the pool before hydrating results.

    Triggers
    1) The user named a reference book that is not in the DB -> fetch & load it, then
       search with its content as an anchor ("A 같은 책").
    2) Retrieval quality is low (top score < threshold or too few hits) -> ask the LLM for
       real candidate titles, fetch & load them, and search again.
    3) Hits whose metadata is thin -> upgraded in the background (non-blocking).
    """
    started = time.perf_counter()
    emit = emit_progress("enrichment")
    intention = state.get("intention") or {}
    vdb = dict(state.get("vdb_result") or {})
    items = list(vdb.get("items") or [])
    query_text = intention.get("query_text") or state.get("request_text") or ""
    exclude_ids = memory_exclude_ids(state.get("user_profile"))

    created: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    reasons: list[str] = []

    emit(
        "ENRICH_PROGRESS",
        {
            "step": "check",
            "message": (
                f"추천 풀 점검 중: 최고 유사도 {items[0]['score']:.2f}, 후보 {len(items)}권"
                if items
                else "추천 풀 점검 중: 검색 결과가 없어요."
            )
            + (
                " · 언급하신 책("
                + ", ".join(m["title"] for m in intention.get("mentioned_books") or [])
                + ")이 DB에 있는지 확인할게요."
                if intention.get("mentioned_books")
                else ""
            ),
        },
    )

    # 1) Mentioned reference books
    mentioned = intention.get("mentioned_books") or []
    if mentioned:
        results = await run_limited(
            [
                enrich_new_book(
                    m["title"],
                    m.get("author"),
                    [m["title_en"]] if m.get("title_en") else None,
                    emit=emit,
                )
                for m in mentioned[:3]
            ]
        )
        for book in filter(None, results):
            anchors.append(book)
            if book.get("enrich_status") == "created":
                created.append(book)
        if anchors:
            reasons.append("mentioned_reference_book")

    # 2) Low retrieval quality
    top_score = items[0]["score"] if items else 0.0
    if not anchors and (top_score < LOW_SCORE_THRESHOLD or len(items) < MIN_RESULTS):
        reasons.append(f"low_similarity(top={top_score:.2f})" if items else "no_results")
        emit(
            "ENRICH_PROGRESS",
            {
                "step": "detect",
                "message": (
                    f"DB에 딱 맞는 책이 부족해요(최고 유사도 {top_score:.2f}). "
                    "외부에서 후보 도서를 찾아 추천 풀을 보강할게요."
                ),
            },
        )
        try:
            suggestions = await suggest_titles(
                state.get("request_text", "").split("\n\n[conversation_context]")[0],
                n=3,
                context=f"[user_long_term_memory]\n{profile_to_prompt(state.get('user_profile'))}",
            )
        except Exception:
            logger.exception("suggest_titles_failed")
            suggestions = []
        if suggestions:
            emit(
                "ENRICH_PROGRESS",
                {
                    "step": "suggest",
                    "message": "보강 후보: " + ", ".join(f"{x['title']}({x.get('author')})" for x in suggestions),
                },
            )
        results = await run_limited(
            [
                enrich_new_book(
                    s.get("title") or s.get("title_en", ""),
                    s.get("author"),
                    [s["title_en"]] if s.get("title_en") else None,
                    emit=emit,
                )
                for s in suggestions
                if s.get("title") or s.get("title_en")
            ]
        )
        created += [b for b in results if b and b.get("enrich_status") == "created"]

    # Re-search when the pool changed or an anchor is available.
    if anchors or created:
        anchor_texts = []
        for book in anchors:
            raw = await asyncio.to_thread(get_raw_book, book["id"])
            if raw and raw.get("embedding_text"):
                anchor_texts.append(raw["embedding_text"][:1200])
        search_query = "\n\n".join([query_text, *anchor_texts])
        anchor_ids = [b["id"] for b in anchors]
        try:
            refreshed = await asyncio.to_thread(
                search_similar_items, search_query, TOP_K, [*exclude_ids, *anchor_ids]
            )
            items = refreshed["items"]
            vdb = {**vdb, **refreshed, "query": query_text}
        except Exception:
            logger.exception("re_search_failed")

    # 3) Thin metadata -> background upgrade (does not block the answer)
    top_books = await asyncio.to_thread(fetch_books_by_ids, [i["id"] for i in items[:5]])
    thin_ids = [b["id"] for b in top_books if is_thin(b)][:3]
    user_id = state.get("user_id")

    def _push(status: str, data: dict[str, Any]) -> None:
        # The graph run is over by the time this fires, so push straight to the user's socket.
        notify_user(user_id, make_event(status, "enrichment", data.get("message", ""), data))

    for book_id in thin_ids:
        _schedule_background(enrich_existing_book(book_id, emit=_push))

    enrichment = {
        "triggered": bool(reasons),
        "reasons": reasons,
        "created": [{"id": b["id"], "name": b["name"], "source": b.get("source")} for b in created],
        "anchors": [{"id": b["id"], "name": b["name"]} for b in anchors],
        "background_upgrades": thin_ids,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
    logger.info("node=enrichment %s", enrichment)
    if created:
        message = f"대화 중 새로 발견한 도서 {len(created)}권을 Vector DB에 적재했어요: " + ", ".join(
            b["name"] for b in created
        )
    elif anchors:
        message = "언급하신 책을 기준점으로 삼아 비슷한 책을 다시 찾았어요."
    elif thin_ids:
        message = f"정보가 빈약한 도서 {len(thin_ids)}권은 백그라운드에서 줄거리·키워드·감정선을 보강할게요."
    else:
        message = "현재 추천 풀로 충분해서 보강 없이 진행할게요."
    return {
        "vdb_result": vdb,
        "enrichment": enrichment,
        "events": [make_event("TOOL_DONE", "enrichment", message, enrichment)],
    }
