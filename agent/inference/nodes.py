from __future__ import annotations

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.base.events import make_event
from agent.base.states import AgentState
from agent.inference.llm import get_chat_model

logger = logging.getLogger(__name__)


class IntentionDecision(BaseModel):
    user_visible_text: str = Field(description="Detailed Korean explanation in 2-4 sentences.")
    intent_label: str = Field(description="Intent identifier in snake_case.")
    route: Literal["vector_db", "chat_completion"]
    query_text: str
    reason: str


_NO_DB_HINTS = [
    "데이터베이스에서 검색하지 말",
    "db에서 검색하지 말",
    "db 검색하지 말",
    "검색하지 말고 직접",
    "검색하지 말아",
    "직접 추천",
]

_DB_SEARCH_HINTS = [
    "추천",
    "비슷한 책",
    "유사한 책",
    "찾아줘",
    "검색",
    "recommend",
    "similar book",
]

_DIRECT_QA_HINTS = [
    "이 책은",
    "무슨 내용",
    "내용이 뭐",
    "줄거리",
    "요약해",
    "설명해",
    "무엇에 대한",
    "what is this book about",
    "summarize",
]


def _latest_user_text(request_text: str) -> str:
    marker = "\n\n[conversation_context]\n"
    return request_text.split(marker, 1)[0].strip()


def _should_skip_db(latest_text: str) -> bool:
    lowered = latest_text.lower()
    return any(hint in lowered for hint in _NO_DB_HINTS)


def _has_db_search_intent(latest_text: str) -> bool:
    lowered = latest_text.lower()
    return any(hint in lowered for hint in _DB_SEARCH_HINTS)


def _has_direct_qa_intent(latest_text: str) -> bool:
    lowered = latest_text.lower()
    if "?" in lowered:
        return True
    return any(hint in lowered for hint in _DIRECT_QA_HINTS)


async def analyze_image_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    llm = get_chat_model(temperature=0)
    user_content: list[dict[str, Any]] = [{"type": "text", "text": "책 표지/책 관련 정보를 분석해줘."}]
    if state.get("image_url"):
        user_content.append({"type": "image_url", "image_url": {"url": state["image_url"]}})
    else:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{state.get('image_b64', '')}"},
            }
        )

    schema = {
        "title": "ImageBookAnalysis",
        "type": "object",
        "properties": {
            "has_book": {"type": "boolean"},
            "title": {"type": ["string", "null"]},
            "author": {"type": ["string", "null"]},
            "book_like_confidence": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": ["has_book", "title", "author", "book_like_confidence", "summary"],
    }
    analyzer = llm.with_structured_output(schema)
    result = await analyzer.ainvoke(
        [
            {
                "role": "system",
                "content": "You analyze an image and extract whether a book exists and what book it is.",
            },
            {"role": "user", "content": user_content},
        ]
    )
    logger.info(
        "node=analyze_image elapsed_ms=%.2f has_image_url=%s has_image_b64=%s",
        (time.perf_counter() - started) * 1000,
        bool(state.get("image_url")),
        bool(state.get("image_b64")),
    )
    return {
        "image_analysis": result,
        "events": [
            make_event(
                "TOOL_DONE",
                "analyze_image",
                "이미지 분석이 끝났어요. 책 존재 여부와 책 정보를 추출했어요.",
                result,
            )
        ],
    }


async def intention_checker_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    llm = get_chat_model(temperature=0)
    structured = llm.with_structured_output(IntentionDecision)
    decision = await structured.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "입력 정보를 바탕으로 사용자 의도를 자세히 분석해라.\n"
                    "출력 규칙:\n"
                    "1) user_visible_text는 2~4문장 한국어.\n"
                    "2) 반드시 근거, 의도 해석, 다음 동작 포함.\n"
                    "3) route는 vector_db 또는 chat_completion.\n"
                    "4) 이미지에 책이 없으면 chat_completion.\n"
                    "5) 검색 의도가 있으면 vector_db.\n"
                    "6) route=vector_db이면 마지막 문장을 '데이터베이스에서 검색해볼게요.'로 마감.\n"
                    "7) route=chat_completion이면 마지막 문장을 '제가 바로 답변해드릴게요.'로 마감.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"request_text={state.get('request_text')}\n"
                    f"input_type={state.get('input_type')}\n"
                    f"image_analysis={state.get('image_analysis')}\n"
                ),
            },
        ]
    )
    dumped = decision.model_dump()
    route = dumped["route"]
    latest_text = _latest_user_text(state.get("request_text") or "")
    image_analysis = state.get("image_analysis") or {}
    has_image_without_book = (
        state.get("input_type") == "image" and image_analysis.get("has_book") is False
    )

    # Deterministic routing policy:
    # - If image has no book -> chat_completion
    # - Else if user explicitly says do not search DB -> chat_completion
    # - Else if direct Q&A intent ("이 책은 무슨 내용?") -> chat_completion
    # - Else if explicit recommendation/search intent -> vector_db
    # - Else default to chat_completion
    if has_image_without_book:
        route = "chat_completion"
        dumped["route"] = route
        dumped["intent_label"] = "image_without_book"
        dumped["reason"] = "Image contains no book-like object."
        dumped["query_text"] = ""
        dumped["user_visible_text"] = (
            "사진을 분석해보니 책으로 보이는 대상이 없었어요. "
            "그래서 데이터베이스 검색보다 현재 맥락을 기준으로 답변하는 게 적절해 보여요. "
            "제가 바로 답변해드릴게요."
        )
    elif _should_skip_db(latest_text):
        route = "chat_completion"
        dumped["route"] = route
        dumped["intent_label"] = dumped.get("intent_label") or "direct_chat_question"
        dumped["reason"] = "User explicitly requested no database search."
        dumped["user_visible_text"] = (
            "요청 문장에서 데이터베이스 검색을 원하지 않는 의도를 확인했어요. "
            "그래서 검색 단계는 생략하고, 질문 맥락을 바탕으로 바로 추천을 생성할게요. "
            "제가 바로 답변해드릴게요."
        )
    elif _has_direct_qa_intent(latest_text) and not _has_db_search_intent(latest_text):
        route = "chat_completion"
        dumped["route"] = route
        dumped["intent_label"] = "direct_book_qa"
        dumped["reason"] = "User asked direct Q&A about the given book/content."
        dumped["user_visible_text"] = (
            "입력 내용을 보면 특정 책에 대해 직접 설명을 요청하는 질문 의도가 보여요. "
            "이 경우에는 데이터베이스 검색보다 현재 책 정보 기반의 직접 답변이 더 적합해요. "
            "제가 바로 답변해드릴게요."
        )
    elif _has_db_search_intent(latest_text):
        route = "vector_db"
        dumped["route"] = route
        dumped["query_text"] = dumped.get("query_text") or latest_text
        dumped["user_visible_text"] = (
            "요청 내용을 보면 도서 추천/탐색 의도가 분명해 보여요. "
            "별도의 검색 제외 지시가 없어서 먼저 데이터베이스에서 후보를 찾는 흐름으로 진행할게요. "
            "데이터베이스에서 검색해볼게요."
        )
    else:
        route = "chat_completion"
        dumped["route"] = route
        dumped["intent_label"] = dumped.get("intent_label") or "direct_chat_question"
        dumped["reason"] = "No explicit recommendation/search intent detected."
        dumped["user_visible_text"] = (
            "현재 요청은 추천/검색보다는 설명형 대화 의도에 가까워 보여요. "
            "그래서 데이터베이스 조회 없이 바로 이해를 돕는 답변으로 진행할게요. "
            "제가 바로 답변해드릴게요."
        )
    logger.info(
        "node=intention_checker elapsed_ms=%.2f route=%s intent_label=%s",
        (time.perf_counter() - started) * 1000,
        route,
        dumped.get("intent_label"),
    )

    return {
        "intention": dumped,
        "intention_tool_type": route,
        "events": [
            make_event(
                "THINKING",
                "intention_checker",
                dumped["user_visible_text"],
                {
                    "intent_label": dumped["intent_label"],
                    "route": dumped["route"],
                    "query_text": dumped["query_text"],
                    "reason": dumped["reason"],
                },
            )
        ],
    }


async def chat_completion_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    llm = get_chat_model(temperature=0.4)
    intention = state.get("intention") or {}
    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": "당신은 친절한 도서 추천 비서다. 한국어로 핵심 위주로 답해라.",
            },
            {
                "role": "user",
                "content": (
                    "아래 의도 분석을 반영해서 답변해줘.\n"
                    f"intention={intention}\n"
                    f"request_text={state.get('request_text')}\n"
                    f"image_analysis={state.get('image_analysis')}\n"
                ),
            },
        ]
    )
    answer = response.content if isinstance(response.content, str) else str(response.content)
    logger.info(
        "node=chat_completion elapsed_ms=%.2f answer_len=%d",
        (time.perf_counter() - started) * 1000,
        len(answer),
    )
    return {
        "chat_result": answer,
        "events": [
            make_event(
                "THINKING",
                "chat_completion",
                (
                    f"{intention.get('user_visible_text', '의도 분석을 완료했어요.')} "
                    "데이터베이스 검색 대신 직접 답변을 생성하고 있어요..."
                ),
                {"route": state.get("intention_tool_type")},
            ),
            make_event("TOOL_DONE", "chat_completion", "직접 답변을 생성했어요.", {"text": answer}),
        ],
    }


async def formatter_node(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    db_items = state.get("db_result") or []
    chat_result = state.get("chat_result") or ""
    locale = state.get("locale", "ko")

    # No summarization: return DB fields as-is for fastest response.
    if db_items:
        cards = []
        normalized_items: list[dict[str, Any]] = []
        for item in db_items:
            item_id = str(item["id"])
            description = item.get("description", "")
            normalized_item = {
                **item,
                "id": item_id,
                "description": description,
            }
            normalized_items.append(normalized_item)
            cards.append(
                (
                    f'<article class="book-card" data-book-id="{item_id}">'
                    f"<h4>{item['name']}</h4>"
                    f"<p>{description}</p>"
                    "</article>"
                )
            )
        payload = {
            "output_type": "BOOK_RECOMMENDATIONS",
            "text": (
            "추천 도서를 찾았어요. 아래 목록을 확인해 주세요."
            if locale == "ko"
            else "I found recommended books. Please check the list below."
            ),
            "html": '<div class="book-reommendations">' + "".join(cards) + "</div>",
            "items": normalized_items,
        }
    else:
        payload = {
            "output_type": "CHAT",
            "text": chat_result or ("요청을 처리했지만 표시할 내용이 없어요." if locale == "ko" else "No content to show."),
            "html": None,
            "items": [],
        }

    # Keep old key compatibility for existing clients.
    payload["type"] = payload["output_type"]
    logger.info(
        "node=formatter elapsed_ms=%.2f output_type=%s items=%d",
        (time.perf_counter() - started) * 1000,
        payload.get("output_type"),
        len(payload.get("items", [])),
    )

    return {
        "formatted": payload,
        "events": [make_event("FINAL", "formatter", "최종 응답을 구성했어요.", payload)],
    }
