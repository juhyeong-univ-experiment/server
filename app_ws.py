from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.base.events import register_notifier, unregister_notifier
from agent.core.graph import stream_events
from agent.enrichment.pipeline import enrich_new_book
from agent.inference.llm import ensure_openai_api_key
from database import count_books, fetch_books_by_ids, list_enriched_books, search_similar_items
from database.profile_store import apply_feedback, delete_profile, get_profile, save_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(title="Book Recommendation WS API")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/search")
async def search_books(
    message: str = Query(..., min_length=1, description="User query text"),
    top_k: int = Query(10, ge=1, le=20, description="Maximum number of results"),
) -> dict[str, Any]:
    """
    Search books by message using Milvus cosine similarity + MongoDB hydration.
    """
    started = time.perf_counter()
    try:
        vdb_started = time.perf_counter()
        vdb_result = await asyncio.to_thread(search_similar_items, message, top_k)
        logger.info("api/search vdb_ms=%.2f", (time.perf_counter() - vdb_started) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VDB query failed: {exc}") from exc

    ids = [item.get("id") for item in vdb_result.get("items", []) if item.get("id")]
    score_by_id = {item["id"]: item.get("score") for item in vdb_result.get("items", []) if item.get("id")}

    try:
        mongo_started = time.perf_counter()
        books = await asyncio.to_thread(fetch_books_by_ids, ids)
        logger.info("api/search mongo_ms=%.2f ids=%d", (time.perf_counter() - mongo_started) * 1000, len(ids))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MongoDB query failed: {exc}") from exc

    merged = []
    for book in books:
        item_id = book.get("id")
        merged.append(
            {
                **book,
                "score": score_by_id.get(item_id),
            }
        )

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("api/search total_ms=%.2f query_len=%d count=%d", total_ms, len(message), len(merged))
    return {
        "query": message,
        "metric": "cosine_similarity",
        "top_k": top_k,
        "count": len(merged),
        "items": merged,
        "raw_vdb": vdb_result,
    }


def _public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in profile.items() if k not in ("_id", "updated_at")}


@app.get("/api/profile/{user_id}")
async def read_profile(user_id: str) -> dict[str, Any]:
    return _public_profile(await asyncio.to_thread(get_profile, user_id))


@app.delete("/api/profile/{user_id}")
async def reset_profile(user_id: str) -> dict[str, Any]:
    await asyncio.to_thread(delete_profile, user_id)
    return {"ok": True}


class EnrichRequest(BaseModel):
    title: str
    author: str | None = None
    title_en: str | None = None


@app.post("/api/enrich")
async def enrich_book(req: EnrichRequest) -> dict[str, Any]:
    """Manually trigger the enrichment pipeline (same one the agent uses mid-conversation)."""
    logs: list[dict[str, Any]] = []
    book = await enrich_new_book(
        req.title,
        req.author,
        [req.title_en] if req.title_en else None,
        emit=lambda status, data: logs.append({"status": status, **data}),
    )
    return {"book": book, "progress": logs}


@app.get("/api/books/enriched")
async def enriched_books(limit: int = Query(30, ge=1, le=200)) -> dict[str, Any]:
    items = await asyncio.to_thread(list_enriched_books, limit)
    return {"items": items, "stats": await asyncio.to_thread(count_books)}


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    return await asyncio.to_thread(count_books)


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False, default=str))


def _safe_decode_image_data_url(image_data_url: str | None) -> str | None:
    """
    Convert image data URL to base64 payload only.
    """
    if not image_data_url:
        return None
    if "," in image_data_url:
        _, payload = image_data_url.split(",", 1)
    else:
        payload = image_data_url
    payload = payload.strip()
    # Validate payload quickly.
    base64.b64decode(payload, validate=True)
    return payload


def _history_to_context(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines[-20:])


def _chunk_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


async def _stream_refined_text_from_openai(
    base_text: str,
    user_query: str,
    locale: str,
    output_type: str | None,
    items: list[dict[str, Any]] | None = None,
) -> Any:
    """
    True OpenAI streaming: refine base text and yield chunks as they arrive.
    """
    ensure_openai_api_key()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True, max_tokens=600)
    if output_type == "BOOK_RECOMMENDATIONS":
        system_prompt = (
            "You are a Korean book recommendation assistant. "
            "Generate line-based summaries only. "
            "For each book, provide a concise Korean-friendly summary from title/description only. "
            "Do not invent facts. "
            "If a book has personal_reason, weave it into that line naturally."
        )
        user_prompt = (
            f"locale={locale}\n"
            f"output_type={output_type}\n"
            f"user_query={user_query}\n"
            f"books={[{k: b.get(k) for k in ('name', 'description', 'keywords', 'mood', 'personal_reason')} for b in items or []]}\n"
            "출력 형식:\n"
            "- 책 개수와 동일한 줄 수로 출력\n"
            "- 각 줄은 해당 책의 요약 1줄(1~2문장)\n"
            "- 책 이름/번호/불릿/마크다운/HTML 태그 절대 금지\n"
            "- 줄바꿈으로만 구분\n"
            "규칙:\n"
            "1) 첫 줄은 첫 번째 책, 둘째 줄은 두 번째 책 ... 순서를 정확히 유지.\n"
            "2) 먼저 user_query와 각 책의 연관성을 분류한다. 너무 엄격하게 분류하지 않고, 터무니 없이 내용이 다른 책만 관련 없다고 판단한다.\n"
            "3) 관련 있는 책만 해당 줄에 한국어 요약 문장을 작성한다.\n"
            "4) 관련 없는 책은 해당 줄을 비워두고 즉시 개행한다(아무 텍스트도 쓰지 않음).\n"
            "5) 총 줄 수는 반드시 책 개수와 동일해야 한다.\n"
        )
    else:
        system_prompt = (
            "You rewrite assistant text for final user display. "
            "Keep the same meaning, keep it concise, natural, and user-friendly. "
            "Do not add unsupported facts. Preserve markdown style when useful."
        )
        user_prompt = (
            f"locale={locale}\n"
            f"output_type={output_type}\n"
            f"user_query={user_query}\n"
            f"base_text={base_text}\n"
        )
    started = time.perf_counter()
    first_token_ms: float | None = None
    token_count = 0
    whitespace_streak = 0
    max_whitespace_streak = 12
    async for chunk in llm.astream(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ):
        text = _chunk_to_text(getattr(chunk, "content", ""))
        if not text:
            continue

        if text.strip() == "":
            whitespace_streak += 1
            if whitespace_streak >= max_whitespace_streak:
                logger.warning(
                    "final_refine_stream terminated due to whitespace_streak=%d output_type=%s",
                    whitespace_streak,
                    output_type,
                )
                break
            continue

        whitespace_streak = 0
        token_count += 1
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - started) * 1000
        yield text
    total_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "final_refine_stream total_ms=%.2f first_token_ms=%.2f token_chunks=%d output_type=%s locale=%s",
        total_ms,
        first_token_ms or -1,
        token_count,
        output_type,
        locale,
    )


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    # Stable per-browser id (localStorage) so memory survives reconnects/sessions.
    user_id = (websocket.query_params.get("user_id") or "").strip()[:64] or f"anon-{session_id}"
    history: list[dict[str, Any]] = []
    await _send(
        websocket,
        {
            "status": "CONNECTED",
            "node": "websocket",
            "message": "웹소켓 연결 완료",
            "data": {"session_id": session_id, "user_id": user_id},
        }
    )

    def _notify(event: dict[str, Any]) -> None:
        # Background jobs (e.g. thin-record upgrades) finish after the turn; push them live.
        asyncio.ensure_future(_send(websocket, {**event, "session_id": session_id, "background": True}))

    register_notifier(user_id, _notify)

    try:
        while True:
            req_started = time.perf_counter()
            payload = await websocket.receive_json()
            req_id = str(uuid.uuid4())

            # Explicit 👍/👎 on a book card -> straight into long-term memory.
            if payload.get("type") == "feedback":
                rating = payload.get("rating")
                if rating not in ("like", "dislike") or not payload.get("title"):
                    continue
                profile = await asyncio.to_thread(get_profile, user_id)
                profile = apply_feedback(
                    profile,
                    book_id=payload.get("book_id"),
                    title=payload["title"],
                    rating=rating,
                    reason=payload.get("reason"),
                )
                await asyncio.to_thread(save_profile, profile)
                await _send(
                    websocket,
                    {
                        "status": "MEMORY",
                        "node": "feedback",
                        "message": f"'{payload['title']}'을(를) {'좋아하신 책' if rating == 'like' else '별로였던 책'}으로 기억해둘게요.",
                        "data": {"changed": True, "profile": _public_profile(profile)},
                    },
                )
                continue

            text = (payload.get("text") or "").strip()
            image_data_url = payload.get("image_data_url")
            locale = payload.get("locale", "ko")
            logger.info(
                "ws_request_start req_id=%s text_len=%d has_image=%s locale=%s",
                req_id,
                len(text),
                bool(image_data_url),
                locale,
            )

            if not text and not image_data_url:
                await _send(
                    websocket,
                    {
                        "status": "ERROR",
                        "node": "input",
                        "message": "text 또는 image_data_url 중 하나는 필요합니다.",
                        "data": {"request_id": req_id},
                    }
                )
                continue

            history.append({"role": "user", "text": text})
            await _send(
                websocket,
                {
                    "status": "USER_MESSAGE",
                    "node": "input",
                    "message": "사용자 메시지 수신",
                    "data": {"request_id": req_id, "text": text, "has_image": bool(image_data_url)},
                }
            )

            try:
                decode_started = time.perf_counter()
                image_b64 = _safe_decode_image_data_url(image_data_url)
                logger.info(
                    "ws_image_decode req_id=%s decode_ms=%.2f has_image_b64=%s",
                    req_id,
                    (time.perf_counter() - decode_started) * 1000,
                    bool(image_b64),
                )
            except Exception:
                await _send(
                    websocket,
                    {
                        "status": "ERROR",
                        "node": "input",
                        "message": "이미지 데이터 파싱에 실패했습니다.",
                        "data": {"request_id": req_id},
                    }
                )
                continue

            request_text = text
            history_context = _history_to_context(history)
            if history_context:
                request_text = f"{text}\n\n[conversation_context]\n{history_context}"

            await _send(
                websocket,
                {
                    "status": "THINKING",
                    "node": "pipeline",
                    "message": "요청을 처리 중입니다.",
                    "data": {"request_id": req_id},
                }
            )

            final_event: dict[str, Any] | None = None
            graph_started = time.perf_counter()
            graph_events = 0
            async for event in stream_events(
                request_text=request_text,
                image_b64=image_b64,
                locale=locale,
                user_id=user_id,
            ):
                graph_events += 1
                enriched: dict[str, Any] = {
                    "request_id": req_id,
                    "session_id": session_id,
                    **event,
                }
                is_final = event.get("status") == "FINAL"
                final_event = enriched if is_final else final_event

                if is_final:
                    logger.info(
                        "ws_final_event req_id=%s graph_elapsed_ms=%.2f graph_events=%d",
                        req_id,
                        (time.perf_counter() - graph_started) * 1000,
                        graph_events,
                    )
                    final_data = dict(enriched.get("data", {}))
                    text_output = final_data.get("text", "") or ""
                    html_output = final_data.get("html")

                    await _send(
                        websocket,
                        {
                            "status": "FINAL_START",
                            "node": "formatter",
                            "message": "최종 응답 스트리밍 시작",
                            "data": {
                                "request_id": req_id,
                                "session_id": session_id,
                                "output_type": final_data.get("output_type") or final_data.get("type"),
                                "items": final_data.get("items") or [],
                                "personal_intro": final_data.get("personal_intro"),
                                "enrichment": final_data.get("enrichment"),
                            },
                        }
                    )
                    output_type = final_data.get("output_type") or final_data.get("type")
                    if output_type == "READING_ROADMAP":
                        # Structured output: no line-refinement streaming needed.
                        await _send(
                            websocket,
                            {
                                "status": "FINAL",
                                "node": "formatter",
                                "message": "최종 응답을 구성했어요.",
                                "data": final_data,
                                "request_id": req_id,
                                "session_id": session_id,
                            },
                        )
                        continue

                    refined_text = ""
                    idx = 0
                    stream_started = time.perf_counter()
                    try:
                        async for token in _stream_refined_text_from_openai(
                            base_text=text_output,
                            user_query=text,
                            locale=locale,
                            output_type=final_data.get("output_type") or final_data.get("type"),
                            items=final_data.get("items"),
                        ):
                            refined_text += token
                            await _send(
                                websocket,
                                {
                                    "status": "FINAL_TOKEN",
                                    "node": "formatter",
                                    "message": "최종 응답 토큰",
                                    "data": {
                                        "request_id": req_id,
                                        "session_id": session_id,
                                        "token": token,
                                        "index": idx,
                                    },
                                }
                            )
                            idx += 1
                        logger.info(
                            "ws_final_tokens req_id=%s token_chunks=%d stream_ms=%.2f",
                            req_id,
                            idx,
                            (time.perf_counter() - stream_started) * 1000,
                        )
                    except Exception:
                        # Fallback to original text if streaming refinement fails.
                        refined_text = text_output
                        logger.exception("ws_final_stream_fallback req_id=%s", req_id)

                    await _send(
                        websocket,
                        {
                            "status": "FINAL",
                            "node": "formatter",
                            "message": "최종 응답을 구성했어요.",
                            "data": {
                                **final_data,
                                "text": refined_text or text_output,
                                "html": html_output,
                            },
                            "request_id": req_id,
                            "session_id": session_id,
                        }
                    )
                else:
                    await _send(websocket, enriched)
                await asyncio.sleep(0)

            if final_event:
                data = final_event.get("data", {})
                assistant_text = data.get("text", "")
                titles = [item.get("name") for item in data.get("items") or [] if item.get("name")]
                if titles:
                    # keep titles in the short-term context so "첫 번째 책 별로였어" can be resolved
                    assistant_text = f"{assistant_text}\n추천한 책: {', '.join(titles)}"
                history.append({"role": "assistant", "text": assistant_text})

            logger.info(
                "ws_request_done req_id=%s total_ms=%.2f",
                req_id,
                (time.perf_counter() - req_started) * 1000,
            )
            await _send(
                websocket,
                {
                    "status": "DONE",
                    "node": "pipeline",
                    "message": "요청 처리 완료",
                    "data": {"request_id": req_id},
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        unregister_notifier(user_id, _notify)
