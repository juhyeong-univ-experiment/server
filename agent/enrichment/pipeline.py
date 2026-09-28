"""Real-time enrichment: collect external evidence -> LLM extraction -> Mongo + Milvus.

The recommendation pool grows from conversations: whenever the agent meets a book that is
missing (or thinly described) in the DB, this pipeline fetches it, extracts plot/keywords/
emotional arc with the LLM and loads it into the Vector DB so it can be retrieved right away.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from agent.enrichment.sources import collect_book_evidence
from agent.inference.llm import get_chat_model
from database import (
    fetch_books_by_ids,
    find_book_by_title,
    get_raw_book,
    insert_book,
    update_book,
    upsert_book_embeddings,
)

logger = logging.getLogger(__name__)

Emit = Callable[[str, dict[str, Any]], None]
THIN_DESCRIPTION_CHARS = 200
_inflight: dict[str, asyncio.Task] = {}


class ExtractedBook(BaseModel):
    matched_sources: list[str] = Field(
        description=(
            "Source tags whose text describes THIS BOOK itself (not an academic field, a person, "
            "a film, or a different book). Empty if none."
        )
    )
    is_real_book: bool = Field(description="True only if matched_sources confirm this is a real, published book.")
    title: str = Field(description="Title as commonly known to Korean readers (Korean edition title if one exists).")
    title_en: str = Field(description="English/original title.")
    author: str
    year: int | None = None
    summary_ko: str = Field(description="3-4 sentence Korean plot/content summary. No spoilers of the ending.")
    keywords: list[str] = Field(description="5-8 Korean keywords (themes, motifs).")
    mood: str = Field(description="Overall mood in Korean, e.g. '잔잔하고 사색적인', '긴장감 넘치는'.")
    emotional_arc: str = Field(description="One Korean sentence describing the reader's emotional journey (감정선).")
    pace: Literal["slow", "medium", "fast"]
    difficulty: Literal["intro", "popular", "advanced"] = Field(
        description="intro=입문서(배경지식 불필요), popular=대중서(교양 수준), advanced=심화/전문서."
    )
    topics: list[str] = Field(description="2-5 Korean subject/topic tags, e.g. '행동경제학', '성장소설'.")
    category: list[str] = Field(description="3-6 lowercase English shelf tags like goodreads, e.g. 'fiction', 'psychology'.")
    embedding_text_en: str = Field(
        description="English paragraph for semantic search: title, author, genres, 3-4 sentence summary, keywords, mood."
    )


class BookRef(BaseModel):
    title: str = Field(description="Title known to Korean readers (Korean edition title if it exists).")
    title_en: str = Field(description="Original / English title. Required - used to look the book up in external sources.")
    author: str = Field(description="Author name (original spelling if foreign).")


class SuggestedTitles(BaseModel):
    titles: list[BookRef] = Field(description="Real, well-known books only.")


def _noop_emit(_: str, __: dict[str, Any]) -> None:
    return None


async def _extract(title: str, author: str | None, evidence: list[dict[str, Any]]) -> ExtractedBook:
    llm = get_chat_model(temperature=0).with_structured_output(ExtractedBook)
    evidence_text = "\n\n".join(
        f"[{ev['source']}] title={ev.get('title')} author={ev.get('author')} year={ev.get('year')} "
        f"subjects={ev.get('subjects')}\n{(ev.get('description') or '')[:1500]}"
        for ev in evidence
    ) or "(외부 소스에서 찾지 못함)"
    return await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "너는 도서 메타데이터 정제기다. 외부 소스 근거를 우선으로 책 정보를 구조화한다. "
                    "근거가 요청 도서가 아닌 다른 책/인물/동명 작품에 대한 것이면 그 근거는 무시한다. "
                    "요청 도서와 일치하는 근거가 하나도 없으면 is_real_book=false로 둔다. "
                    "근거가 있으면 널리 알려진 사실로만 보완하고, 절대 없는 책을 지어내지 마라."
                ),
            },
            {"role": "user", "content": f"요청 도서: {title} / 저자 힌트: {author or '없음'}\n\n근거:\n{evidence_text}"},
        ]
    )


def _to_document(extracted: ExtractedBook, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next((ev for ev in evidence if ev.get("url")), None)
    return {
        "title": extracted.title,
        "title_en": extracted.title_en,
        "author": extracted.author,
        "year": extracted.year,
        "category": extracted.category,
        "description": extracted.summary_ko,
        "summary": extracted.summary_ko,
        "keywords": extracted.keywords,
        "mood": extracted.mood,
        "emotional_arc": extracted.emotional_arc,
        "pace": extracted.pace,
        "difficulty": extracted.difficulty,
        "topics": extracted.topics,
        "embedding_text": extracted.embedding_text_en,
        "source": "+".join(dict.fromkeys(ev["source"] for ev in evidence)) if evidence else "llm_knowledge",
        "source_url": primary.get("url") if primary else None,
        "evidence_count": len(evidence),
        "enriched_at": datetime.now(timezone.utc),
    }


def _lookup_existing(*titles: str | None) -> dict[str, Any] | None:
    for t in titles:
        if t:
            found = find_book_by_title(t)
            if found:
                return found
    return None


def _title_variants(title: str, alt_titles: list[str] | None) -> list[str]:
    variants = []
    for t in [*(alt_titles or [])]:
        variants += [t, t.split(":")[0].strip()]
    variants.append(title.split(":")[0].strip())
    return [v for v in dict.fromkeys(variants) if v and v != title]


async def _enrich_new_book(
    title: str, author: str | None, alt_titles: list[str] | None, emit: Emit
) -> dict[str, Any] | None:
    started = time.perf_counter()
    alt_titles = _title_variants(title, alt_titles)
    existing = await asyncio.to_thread(_lookup_existing, title, *(alt_titles or []))
    if existing:
        return {**existing, "enrich_status": "already_exists"}

    emit("ENRICH_PROGRESS", {"title": title, "step": "collect", "message": f"'{title}'이(가) DB에 없어요. 외부 소스(Open Library·Wikipedia)에서 수집 중이에요."})
    evidence = await collect_book_evidence(title, author, alt_titles)
    emit(
        "ENRICH_PROGRESS",
        {
            "title": title,
            "step": "extract",
            "sources": [ev["source"] for ev in evidence],
            "message": f"'{title}' 근거 {len(evidence)}건 수집 → LLM으로 줄거리·키워드·감정선을 추출 중이에요.",
        },
    )
    if not evidence:
        emit(
            "ENRICH_PROGRESS",
            {"title": title, "step": "rejected", "message": f"'{title}'은(는) 외부 소스에서 근거를 찾지 못해 적재하지 않았어요. (환각 방지)"},
        )
        return None
    extracted = await _extract(title, author, evidence)
    if not extracted.is_real_book or not extracted.matched_sources:
        emit(
            "ENRICH_PROGRESS",
            {
                "title": title,
                "step": "rejected",
                "message": f"'{title}'은(는) 수집된 근거가 이 책을 설명하지 않아(분야/인물/동명 작품 등) 적재하지 않았어요.",
            },
        )
        return None
    evidence = [ev for ev in evidence if ev["source"] in extracted.matched_sources] or evidence

    # The canonical title may differ from the user's wording (e.g. '생각에 관한 생각').
    existing = await asyncio.to_thread(_lookup_existing, extracted.title, extracted.title_en)
    if existing:
        return {**existing, "enrich_status": "already_exists"}

    doc = _to_document(extracted, evidence)
    book_id = await asyncio.to_thread(insert_book, doc)
    await asyncio.to_thread(upsert_book_embeddings, [book_id], [doc["embedding_text"]])
    book = (await asyncio.to_thread(fetch_books_by_ids, [book_id]))[0]
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("enrich_new title=%s id=%s sources=%s elapsed_ms=%.0f", title, book_id, doc["source"], elapsed)
    emit(
        "ENRICH_PROGRESS",
        {
            "title": book["name"],
            "step": "stored",
            "book_id": book_id,
            "message": (
                f"'{book['name']}'을(를) Vector DB에 새로 적재했어요. (출처: {doc['source']}) "
                f"키워드: {', '.join(doc['keywords'][:4])} · 감정선: {doc['emotional_arc']}"
            ),
        },
    )
    return {**book, "enrich_status": "created"}


async def enrich_new_book(
    title: str,
    author: str | None = None,
    alt_titles: list[str] | None = None,
    emit: Emit | None = None,
) -> dict[str, Any] | None:
    """Idempotent entry point: concurrent requests for the same title share one task."""
    from database.mongo_client import normalize_title

    key = normalize_title(title)
    task = _inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_enrich_new_book(title, author, alt_titles, emit or _noop_emit))
        _inflight[key] = task
        task.add_done_callback(lambda _: _inflight.pop(key, None))
    try:
        return await asyncio.shield(task)
    except Exception:
        logger.exception("enrich_new_failed title=%s", title)
        return None


async def enrich_existing_book(book_id: str, emit: Emit | None = None) -> dict[str, Any] | None:
    """Upgrade a thin record in place (keywords, mood, emotional arc) and re-embed it."""
    emit = emit or _noop_emit
    raw = await asyncio.to_thread(get_raw_book, book_id)
    if not raw or raw.get("enriched_at"):
        return None
    try:
        evidence = await collect_book_evidence(raw.get("title", ""), None)
        if raw.get("description"):
            evidence.insert(0, {"source": "seed", "title": raw.get("title"), "description": raw["description"]})
        extracted = await _extract(raw.get("title", ""), None, evidence)
        doc = _to_document(extracted, evidence)
        # keep the original identity fields of the seed record
        doc.update({"title": raw.get("title"), "author": raw.get("author") or doc["author"]})
        await asyncio.to_thread(update_book, book_id, doc)
        await asyncio.to_thread(upsert_book_embeddings, [book_id], [doc["embedding_text"]])
        logger.info("enrich_existing id=%s title=%s", book_id, raw.get("title"))
        emit(
            "ENRICH_PROGRESS",
            {
                "step": "upgraded",
                "book_id": book_id,
                "title": raw.get("title"),
                "message": (
                    f"[백그라운드] '{raw.get('title')}' 정보 보강 완료 — 키워드: {', '.join(doc['keywords'][:4])} · "
                    f"감정선: {doc['emotional_arc']}"
                ),
            },
        )
        return (await asyncio.to_thread(fetch_books_by_ids, [book_id]))[0]
    except Exception:
        logger.exception("enrich_existing_failed id=%s", book_id)
        return None


def is_thin(book: dict[str, Any]) -> bool:
    return not book.get("enriched") and len(book.get("description") or "") < THIN_DESCRIPTION_CHARS


async def suggest_titles(query: str, n: int = 3, context: str = "") -> list[dict[str, str]]:
    """Real candidate books for a query the DB cannot answer well."""
    llm = get_chat_model(temperature=0.2).with_structured_output(SuggestedTitles)
    result = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    f"요청에 가장 잘 맞는 실존 도서 {n}권을 제시해라. 한국 독자가 구할 수 있는 유명한 책을 우선한다. "
                    "존재가 확실하지 않은 책은 절대 넣지 마라."
                ),
            },
            {"role": "user", "content": f"요청: {query}\n{context}"},
        ]
    )
    return [t.model_dump() for t in result.titles[:n]]


async def run_limited(coros: list[Awaitable[Any]], limit: int = 3) -> list[Any]:
    sem = asyncio.Semaphore(limit)

    async def _wrap(c: Awaitable[Any]) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(_wrap(c) for c in coros))
