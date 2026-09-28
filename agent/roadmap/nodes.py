"""Context- and difficulty-aware reading roadmap (입문서 → 대중서 → 심화/전문서)."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.base.events import emit_progress, make_event
from agent.base.states import AgentState
from agent.enrichment.pipeline import enrich_new_book, run_limited
from agent.inference.llm import get_chat_model
from agent.memory.profile_view import profile_to_prompt
from database import fetch_books_by_ids, search_similar_items

logger = logging.getLogger(__name__)

StageKey = Literal["intro", "popular", "advanced"]
STAGE_ORDER: list[str] = ["intro", "popular", "advanced"]
STAGE_LABEL = {"intro": "입문서", "popular": "대중서", "advanced": "심화·전문서"}
LEVEL_TO_START = {"beginner": "intro", "intermediate": "popular", "advanced": "advanced"}


class CanonicalBook(BaseModel):
    title: str = Field(description="Title known to Korean readers (Korean edition title if it exists).")
    title_en: str | None = None
    author: str | None = None


class StagePlan(BaseModel):
    stage: StageKey
    query_en: str = Field(description="English semantic-search query for books at this stage.")
    canonical_books: list[CanonicalBook] = Field(description="2 real, well-known books that fit this stage.")


class RoadmapPlan(BaseModel):
    topic: str = Field(description="Topic in Korean.")
    user_level: Literal["beginner", "intermediate", "advanced"]
    level_evidence: str = Field(description="Korean: why you judged this level (from the message or memory).")
    emotional_state: str = Field(description="Korean: the user's current mood/situation if expressed, else '특별한 언급 없음'.")
    stages: list[StagePlan]


class CuratedBook(BaseModel):
    id: str
    reason: str = Field(description="Korean: why this book at this stage for this user.")


class CuratedStage(BaseModel):
    stage: StageKey
    goal: str = Field(description="Korean: what the reader gains at this stage.")
    books: list[CuratedBook] = Field(description="1-2 books chosen ONLY from the candidate ids of this stage.")
    bridge: str = Field(description="Korean: how this stage naturally leads to the next one. Empty for the last stage.")


class CuratedRoadmap(BaseModel):
    intro: str = Field(description="2-3 Korean sentences: level/mood reading and why the path starts where it does.")
    stages: list[CuratedStage]
    first_book_id: str = Field(description="The id of the very first book the user should read.")


def _latest_user_text(request_text: str) -> str:
    return (request_text or "").split("\n\n[conversation_context]\n", 1)[0].strip()


async def _gather_stage_candidates(
    plan: StagePlan, exclude_ids: list[str], emit
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Canonical books (enriched on demand) + vector-search hits for one stage."""
    enriched = await run_limited(
        [
            enrich_new_book(b.title, b.author, [b.title_en] if b.title_en else None, emit=emit)
            for b in plan.canonical_books[:2]
        ],
        limit=2,
    )
    canonical = [b for b in enriched if b]
    created = [b for b in canonical if b.get("enrich_status") == "created"]
    try:
        hits = await asyncio.to_thread(search_similar_items, plan.query_en, 4, exclude_ids)
        searched = await asyncio.to_thread(fetch_books_by_ids, [h["id"] for h in hits["items"]])
    except Exception:
        logger.exception("roadmap_search_failed stage=%s", plan.stage)
        searched = []
    seen: set[str] = set()
    pool = []
    for book in [*canonical, *searched]:
        if book["id"] in seen or book["id"] in exclude_ids:
            continue
        seen.add(book["id"])
        pool.append(book)
    return pool, created


async def roadmap_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    emit = emit_progress("roadmap")
    profile = state.get("user_profile")
    latest = _latest_user_text(state.get("request_text", ""))
    memory = profile_to_prompt(profile)

    planner = get_chat_model(temperature=0.2).with_structured_output(RoadmapPlan)
    plan: RoadmapPlan = await planner.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "너는 독서 로드맵 설계자다. 사용자의 현재 지식 수준과 감정 상태를 추정하고 "
                    "intro(입문서: 배경지식 불필요) → popular(대중서: 교양 수준) → advanced(심화/전문서) "
                    "세 단계 각각에 대해 검색 쿼리와 실존하는 대표 도서 2권을 제시하라. "
                    "한국 독자가 구할 수 있는 책을 우선하고, 존재가 불확실한 책은 넣지 마라. "
                    "장기기억에 있는 지식수준/기피 소재/선호 스타일을 반영하라."
                ),
            },
            {"role": "user", "content": f"요청: {latest}\n\n[user_long_term_memory]\n{memory}"},
        ]
    )
    emit(
        "ROADMAP_PROGRESS",
        {
            "step": "plan",
            "message": (
                f"'{plan.topic}' 로드맵을 설계 중이에요. 현재 수준은 {plan.user_level}로 판단했어요"
                f"({plan.level_evidence}). 단계별 후보 도서를 모을게요."
            ),
        },
    )

    exclude_ids = [
        e.get("book_id") for e in (profile or {}).get("disliked") or [] if e.get("book_id")
    ] or []
    stage_plans = {p.stage: p for p in plan.stages}
    results = await asyncio.gather(
        *(_gather_stage_candidates(stage_plans[s], exclude_ids, emit) for s in STAGE_ORDER if s in stage_plans)
    )
    pools: dict[str, list[dict[str, Any]]] = {}
    created: list[dict[str, Any]] = []
    for stage, (pool, new_books) in zip([s for s in STAGE_ORDER if s in stage_plans], results):
        pools[stage] = pool
        created += new_books

    candidate_view = {
        stage: [
            {
                "id": b["id"],
                "title": b["name"],
                "difficulty": b.get("difficulty"),
                "topics": b.get("topics"),
                "mood": b.get("mood"),
                "description": (b.get("description") or "")[:250],
            }
            for b in pool
        ]
        for stage, pool in pools.items()
    }
    curator = get_chat_model(temperature=0.3).with_structured_output(CuratedRoadmap)
    curated: CuratedRoadmap = await curator.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "단계별 후보 중에서 독서 경로를 큐레이션하라. 각 단계에서 해당 단계 후보 id만 1~2권 고른다. "
                    "주제와 무관한 후보는 고르지 마라. 사용자의 수준이 높으면 앞 단계는 1권만 가볍게, "
                    "감정 상태가 지쳐 있으면 부담 없는 책을 먼저 두는 식으로 첫 책을 정하라. "
                    "bridge에는 앞 책에서 다음 책으로 자연스럽게 넘어가는 이유를 써라."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"요청: {latest}\n주제: {plan.topic}\n추정 수준: {plan.user_level} ({plan.level_evidence})\n"
                    f"감정 상태: {plan.emotional_state}\n\n[user_long_term_memory]\n{memory}\n\n"
                    f"[candidates_by_stage]\n{candidate_view}"
                ),
            },
        ]
    )

    by_id = {b["id"]: b for pool in pools.values() for b in pool}
    stages_out = []
    used: set[str] = set()
    for cs in sorted(curated.stages, key=lambda s: STAGE_ORDER.index(s.stage)):
        allowed = {b["id"] for b in pools.get(cs.stage, [])}
        books = []
        for cb in cs.books:
            if cb.id in allowed and cb.id not in used:
                used.add(cb.id)
                books.append({**by_id[cb.id], "reason": cb.reason})
        if not books:
            continue
        stages_out.append(
            {"stage": cs.stage, "label": STAGE_LABEL[cs.stage], "goal": cs.goal, "bridge": cs.bridge, "books": books}
        )

    start_stage = LEVEL_TO_START[plan.user_level]
    first_book_id = curated.first_book_id if curated.first_book_id in used else None
    if not first_book_id and stages_out:
        first_book_id = stages_out[0]["books"][0]["id"]
    roadmap = {
        "topic": plan.topic,
        "user_level": plan.user_level,
        "level_evidence": plan.level_evidence,
        "emotional_state": plan.emotional_state,
        "start_stage": start_stage,
        "first_book_id": first_book_id,
        "intro": curated.intro,
        "stages": stages_out,
        "enriched": [{"id": b["id"], "name": b["name"]} for b in created],
    }
    logger.info(
        "node=roadmap elapsed_ms=%.2f stages=%d enriched=%d",
        (time.perf_counter() - started) * 1000,
        len(stages_out),
        len(created),
    )
    return {
        "roadmap": roadmap,
        "events": [
            make_event(
                "TOOL_DONE",
                "roadmap",
                f"{len(stages_out)}단계 독서 로드맵을 만들었어요."
                + (f" 이 과정에서 새 도서 {len(created)}권을 DB에 적재했어요." if created else ""),
                {"stages": len(stages_out), "enriched": roadmap["enriched"]},
            )
        ],
    }
