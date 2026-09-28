"""Long-term memory: load the profile, personalize results, and learn from every turn."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.base.events import emit_progress, make_event
from agent.base.states import AgentState
from agent.inference.llm import get_chat_model
from agent.memory.profile_view import profile_to_prompt
from database.profile_store import (
    add_recommended,
    apply_memory_update,
    get_profile,
    has_memory,
    save_profile,
)

logger = logging.getLogger(__name__)


def _latest_user_text(request_text: str) -> str:
    return (request_text or "").split("\n\n[conversation_context]\n", 1)[0].strip()


async def load_memory_node(state: AgentState) -> dict[str, Any]:
    user_id = state.get("user_id") or ""
    try:
        profile = await asyncio.to_thread(get_profile, user_id)
    except Exception:
        logger.exception("load_memory_failed user_id=%s", user_id)
        profile = None
    summary = profile_to_prompt(profile) if profile and has_memory(profile) else ""
    return {
        "user_profile": profile,
        "events": [
            make_event(
                "THINKING",
                "memory",
                _loaded_message(profile) if summary else "아직 기억된 취향이 없어요. 대화하면서 알아갈게요.",
                {"has_memory": bool(summary), "summary": summary},
            )
        ],
    }


def _loaded_message(profile: dict[str, Any]) -> str:
    parts = []
    if profile.get("liked"):
        parts.append(f"좋아한 책 {len(profile['liked'])}권")
    if profile.get("disliked"):
        parts.append(f"별로였던 책 {len(profile['disliked'])}권")
    if profile.get("preferences"):
        parts.append("선호 '" + "', '".join(profile["preferences"][-3:]) + "'")
    if profile.get("avoid"):
        parts.append("기피 '" + "', '".join(profile["avoid"][-3:]) + "'")
    if profile.get("recommended"):
        parts.append(f"추천 이력 {len(profile['recommended'])}권")
    return "장기기억을 불러왔어요: " + " · ".join(parts)


def _only_new(profile: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """The extractor tends to restate known facts; keep only what is actually new."""
    out = dict(update)
    for key in ("interests", "avoid", "preferences"):
        known = {x.strip().lower() for x in profile.get(key) or []}
        out[key] = [x for x in update.get(key) or [] if x.strip().lower() not in known]
    known_fb = {
        ((e.get("title") or "").strip().lower(), rating)
        for rating, key in (("like", "liked"), ("dislike", "disliked"))
        for e in profile.get(key) or []
    }
    out["feedback"] = [
        fb for fb in update.get("feedback") or [] if (fb["title"].strip().lower(), fb["rating"]) not in known_fb
    ]
    levels = profile.get("knowledge_levels") or {}
    out["knowledge_levels"] = [
        kl for kl in update.get("knowledge_levels") or [] if levels.get(kl["topic"]) != kl["level"]
    ]
    return out


def _diff_message(update: dict[str, Any]) -> str:
    parts = []
    for fb in update.get("feedback") or []:
        icon = "👍" if fb["rating"] == "like" else "👎"
        parts.append(f"{icon} '{fb['title']}'" + (f"({fb['reason']})" if fb.get("reason") else ""))
    for key, label in (("preferences", "선호"), ("interests", "관심사"), ("avoid", "기피")):
        for item in update.get(key) or []:
            parts.append(f"+{label} '{item}'")
    for kl in update.get("knowledge_levels") or []:
        parts.append(f"지식수준 {kl['topic']}={kl['level']}")
    return "장기기억 업데이트: " + ", ".join(parts) if parts else ""


class RerankedBook(BaseModel):
    id: str
    conflict_evidence: str = Field(
        description="Exact phrase from the candidate's description showing a 기피 소재. Empty string if none."
    )
    keep: bool = Field(
        description="False ONLY if conflict_evidence is non-empty and clearly matches a 기피 소재. When unsure, keep=true."
    )
    personal_reason: str = Field(description="One short Korean sentence connecting this book to the user's memory. Empty if none.")


class Personalization(BaseModel):
    intro: str = Field(
        description=(
            "1-2 Korean sentences that reference concrete memory, e.g. "
            "'지난번 추천드린 A처럼 빠른 전개를 좋아하시니, 이번엔 B를 먼저 추천해요.'"
        )
    )
    books: list[RerankedBook] = Field(description="Candidates reordered best-first for this user.")


async def personalize_node(state: AgentState) -> dict[str, Any]:
    """Rerank/filter hydrated results with the user's memory and write a personal intro."""
    profile = state.get("user_profile") or {}
    items = state.get("db_result") or []
    if not items or not has_memory(profile):
        return {"personal_intro": None, "events": []}
    started = time.perf_counter()
    emit_progress("personalize")(
        "MEMORY",
        {"step": "personalize", "message": f"장기기억 기준으로 후보 {len(items)}권을 개인화(기피 소재 필터·재정렬) 중이에요."},
    )
    candidates = [
        {
            "id": b["id"],
            "title": b["name"],
            "category": b.get("category"),
            "keywords": b.get("keywords"),
            "mood": b.get("mood"),
            "pace": b.get("pace"),
            "description": (b.get("description") or "")[:300],
        }
        for b in items
    ]
    try:
        llm = get_chat_model(temperature=0.3).with_structured_output(Personalization)
        result = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 사용자의 장기기억을 바탕으로 추천 목록을 개인화한다. "
                        "후보 id만 사용하고 새로운 책을 만들지 마라. 설명에 기피 소재가 핵심으로 드러난 경우에만 keep=false, "
                        "애매하면 keep=true로 두고 순서만 뒤로 보내라. "
                        "intro에서는 기억 속 구체적인 책/취향을 한 번 언급해 초개인화된 느낌을 줘라 "
                        "(예: '지난번 추천드린 A처럼 빠른 전개를 좋아하시니, 이번엔 B를 먼저 추천해요'). "
                        "intro에서 과거를 언급할 때는 기억에 적힌 항목(좋아한 책, 별로였던 책과 그 이유, 선호 스타일)만 그대로 인용하라. "
                        "'지난번 추천한 X 장르'처럼 기억에 없는 과거를 지어내지 마라. 좋아한 책이 없으면 선호 스타일/별로였던 책을 근거로 삼아라."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"이번 요청: {_latest_user_text(state.get('request_text', ''))}\n\n"
                        f"[user_long_term_memory]\n{profile_to_prompt(profile)}\n\n"
                        f"[candidates]\n{candidates}"
                    ),
                },
            ]
        )
    except Exception:
        logger.exception("personalize_failed")
        return {"personal_intro": None, "events": []}

    by_id = {b["id"]: b for b in items}
    ordered: list[dict[str, Any]] = []
    dropped: list[str] = []
    for rb in result.books:
        book = by_id.pop(rb.id, None)
        if book is None:
            continue
        if not rb.keep and rb.conflict_evidence.strip():
            dropped.append(book["name"])
            continue
        ordered.append({**book, "personal_reason": rb.personal_reason or None})
    ordered += list(by_id.values())  # keep anything the LLM forgot, at the end
    # Never let filtering starve the answer: backfill from dropped ones if fewer than 3 remain.
    if len(ordered) < 3:
        dropped_books = [b for b in items if b["name"] in dropped]
        ordered += dropped_books[: 3 - len(ordered)]
        dropped = [name for name in dropped if name not in {b["name"] for b in ordered}]
    logger.info(
        "node=personalize elapsed_ms=%.2f kept=%d dropped=%d",
        (time.perf_counter() - started) * 1000,
        len(ordered),
        len(dropped),
    )
    message = "기억해둔 취향으로 추천 순서를 조정했어요."
    if dropped:
        message += f" 기피 소재와 겹치는 {len(dropped)}권은 뺐어요."
    return {
        "db_result": ordered,
        "book_recommendation": ordered,
        "personal_intro": result.intro,
        "events": [make_event("TOOL_DONE", "personalize", message, {"intro": result.intro, "dropped": dropped})],
    }


class FeedbackItem(BaseModel):
    title: str = Field(description="Book title as it appears in the recommended list or the user's words.")
    rating: Literal["like", "dislike"]
    reason: str | None = Field(default=None, description="Why, in short Korean, e.g. '전개가 빨라서'.")


class KnowledgeLevel(BaseModel):
    topic: str
    level: Literal["beginner", "intermediate", "advanced"]


class MemoryUpdate(BaseModel):
    feedback: list[FeedbackItem] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list, description="New stable interests (Korean nouns).")
    avoid: list[str] = Field(default_factory=list, description="Subjects/styles the user wants to avoid.")
    preferences: list[str] = Field(
        default_factory=list, description="Reading-style preferences, e.g. '빠른 전개', '짧은 분량', '열린 결말 싫음'."
    )
    knowledge_levels: list[KnowledgeLevel] = Field(default_factory=list)
    learned: str = Field(description="One Korean sentence summarizing what was learned. Empty if nothing.")


async def memory_update_node(state: AgentState) -> dict[str, Any]:
    """Extract durable preferences from this turn and persist them."""
    user_id = state.get("user_id")
    if not user_id:
        return {"events": []}
    started = time.perf_counter()
    emit_progress("memory_update")(
        "MEMORY", {"step": "extract", "message": "이번 대화에서 기억해둘 취향이 있는지 정리 중이에요."}
    )
    profile = state.get("user_profile") or await asyncio.to_thread(get_profile, user_id)
    latest = _latest_user_text(state.get("request_text", ""))
    recent_titles = [e["title"] for e in (profile.get("recommended") or [])[-15:] if e.get("title")]

    update = MemoryUpdate(learned="")
    if latest:
        try:
            llm = get_chat_model(temperature=0).with_structured_output(MemoryUpdate)
            update = await llm.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "사용자 발화에서 장기적으로 기억할 독서 취향만 추출한다. "
                            "일회성 요청(예: '판타지 추천해줘', '성장소설 추천해줘')의 장르는 interests에 넣지 마라. "
                            "사용자가 '좋아한다/관심 있다/싫다'고 명시한 지속적 취향만 넣어라. "
                            "책에 대한 감상(재밌었다/지루했다/별로였다)은 feedback으로. "
                            "'~에 대해 잘 모른다/처음이다/전공자다'는 knowledge_levels로. "
                            "추출할 것이 없으면 모든 리스트를 비우고 learned는 빈 문자열."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"사용자 발화: {latest}\n"
                            f"최근 추천했던 책 제목: {recent_titles}\n"
                            f"기존 기억:\n{profile_to_prompt(profile)}"
                        ),
                    },
                ]
            )
        except Exception:
            logger.exception("memory_extract_failed")

    # Resolve feedback titles to book ids from the recommendation history.
    by_title = {(e.get("title") or "").strip().lower(): e.get("book_id") for e in profile.get("recommended") or []}
    update_dict = update.model_dump()
    for fb in update_dict["feedback"]:
        fb["book_id"] = by_title.get(fb["title"].strip().lower())
        if not fb["book_id"]:
            for title, book_id in by_title.items():
                if title and (fb["title"].lower() in title or title in fb["title"].lower()):
                    fb["book_id"], fb["title"] = book_id, next(
                        e["title"] for e in profile["recommended"] if e.get("book_id") == book_id
                    )
                    break

    update_dict = _only_new(profile, update_dict)
    profile = apply_memory_update(profile, update_dict)
    recommended = state.get("book_recommendation") or []
    roadmap = state.get("roadmap") or {}
    for stage in roadmap.get("stages") or []:
        recommended = [*recommended, *stage.get("books", [])]
    profile = add_recommended(profile, recommended)
    profile["turns"] = int(profile.get("turns") or 0) + 1
    await asyncio.to_thread(save_profile, profile)

    changed = any(update_dict[k] for k in ("feedback", "interests", "avoid", "preferences", "knowledge_levels"))
    logger.info("node=memory_update elapsed_ms=%.2f changed=%s", (time.perf_counter() - started) * 1000, changed)
    return {
        "memory_update": update_dict,
        "user_profile": profile,
        "events": [
            make_event(
                "MEMORY",
                "memory_update",
                _diff_message(update_dict) if changed else "새로 기억할 취향은 없었어요.",
                {"changed": changed, "update": update_dict, "profile": _public_profile(profile)},
            )
        ],
    }


def _public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in profile.items() if k not in ("_id", "updated_at")}
