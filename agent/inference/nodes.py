from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.base.events import make_event
from agent.base.states import AgentState
from agent.inference.llm import get_chat_model
from agent.memory.profile_view import profile_to_prompt

logger = logging.getLogger(__name__)


class MentionedBook(BaseModel):
    title: str = Field(description="Title exactly as the user wrote it.")
    title_en: str | None = Field(default=None, description="English/original title if you know it.")
    author: str | None = None


class IntentionDecision(BaseModel):
    user_visible_text: str = Field(description="Detailed Korean explanation in 2-4 sentences.")
    intent_label: str = Field(description="Intent identifier in snake_case.")
    route: Literal["vector_db", "roadmap", "chat_completion"]
    query_text: str = Field(
        description=(
            "ENGLISH semantic-search query describing the wanted books (genre, themes, mood, pace). "
            "The vector DB is English, so always write it in English."
        )
    )
    mentioned_books: list[MentionedBook] = Field(
        default_factory=list,
        description="Specific books the user named as a reference/anchor (e.g. '데미안 같은 책' -> 데미안).",
    )
    reason: str


_NO_DB_HINTS = [
    "데이터베이스에서 검색하지 말",
    "db에서 검색하지 말",
    "db 검색하지 말",
    "검색하지 말고 직접",
    "검색하지 말아",
    "직접 추천",
]

# "추천해줘/추천 좀/추천받고" = request. A bare "추천" (e.g. "추천해준 책은 지루했어") is not.
_RECOMMEND_REQUEST_RE = re.compile(r"추천\s*(해\s*줘|해\s*주|해\s*달|좀|받|부탁|해봐|해 봐|할\s*만한|하는\s*책|\?|$)")
_FEEDBACK_HINTS = [
    "재밌었", "재미있었", "재미없었", "지루했", "별로였", "좋았어", "좋았다", "최고였", "실망",
    "좋아해", "싫어해", "좋아하는 편", "싫어하는 편", "선호해", "싫더라", "좋더라",
]
_REFERENCE_RE = re.compile(r"[『「'\"“‘]([^』」'\"”’]{1,40})[』」'\"”’]|([가-힣A-Za-z0-9]{2,20})\s*(?:같은|처럼|와 비슷한|과 비슷한|이랑 비슷한|랑 비슷한)")

_DB_SEARCH_HINTS = [
    "비슷한 책",
    "유사한 책",
    "찾아줘",
    "검색",
    "recommend",
    "similar book",
]

_ROADMAP_HINTS = [
    "로드맵",
    "입문",
    "단계별",
    "순서대로",
    "공부하고 싶",
    "공부하려",
    "탐구",
    "깊이 알고",
    "깊게 알고",
    "처음부터",
    "기초부터",
    "독서 경로",
    "roadmap",
    "reading path",
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
    return bool(_RECOMMEND_REQUEST_RE.search(lowered)) or any(hint in lowered for hint in _DB_SEARCH_HINTS)


def _is_feedback_only(latest_text: str) -> bool:
    """'A는 지루했어, 빠른 전개가 좋아' -> memory update + short ack, not a new search."""
    lowered = latest_text.lower()
    return any(h in lowered for h in _FEEDBACK_HINTS) and not _has_db_search_intent(latest_text)


def _regex_reference_books(latest_text: str) -> list[dict[str, Any]]:
    refs = []
    for m in _REFERENCE_RE.finditer(latest_text):
        title = (m.group(1) or m.group(2) or "").strip()
        if title and title not in ("이런", "그런", "저런", "이것", "그것", "지난번", "요즘", "전개"):
            refs.append({"title": title, "title_en": None, "author": None})
    return refs


def _has_roadmap_intent(latest_text: str) -> bool:
    lowered = latest_text.lower()
    return any(hint in lowered for hint in _ROADMAP_HINTS)


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
                    "3) route는 vector_db, roadmap, chat_completion 중 하나.\n"
                    "4) 이미지에 책이 없으면 chat_completion.\n"
                    "5) 검색 의도가 있으면 vector_db. 특정 주제를 단계적으로 공부/탐구하려 하면 roadmap.\n"
                    "6) route=vector_db이면 마지막 문장을 '데이터베이스에서 검색해볼게요.'로 마감.\n"
                    "7) route=chat_completion이면 마지막 문장을 '제가 바로 답변해드릴게요.'로 마감.\n"
                    "8) query_text는 반드시 영어. 사용자의 장기기억(선호 스타일/관심사)이 이번 요청과 관련되면 반영하되, "
                    "이번 요청이 우선이다. 기피 소재는 query_text에 넣지 마라.\n"
                    "9) 사용자가 기준으로 언급한 구체적 책 제목은 mentioned_books에 넣어라(원제 title_en도 채워라). 이미지에서 인식한 책도 포함.\n"
                    "   예) '데미안 같은 성장소설 추천해줘' -> mentioned_books=[{title:'데미안', title_en:'Demian', author:'Hermann Hesse'}]\n"
                    "   예) '해리포터처럼 마법 학교 나오는 책' -> [{title:'해리포터', title_en:\"Harry Potter and the Philosopher's Stone\", author:'J.K. Rowling'}]\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"request_text={state.get('request_text')}\n"
                    f"input_type={state.get('input_type')}\n"
                    f"image_analysis={state.get('image_analysis')}\n"
                    f"[user_long_term_memory]\n{profile_to_prompt(state.get('user_profile'))}\n"
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
    elif _is_feedback_only(latest_text):
        route = "chat_completion"
        dumped["route"] = route
        dumped["intent_label"] = "reading_feedback"
        dumped["reason"] = "User shared feedback/preferences without asking for a new search."
        dumped["user_visible_text"] = (
            "새 추천 요청이라기보다 읽은 책에 대한 감상과 취향을 알려주신 것 같아요. "
            "이 내용은 장기기억에 반영해두고, 다음 추천부터 활용할게요. "
            "제가 바로 답변해드릴게요."
        )
    elif _has_roadmap_intent(latest_text) or (
        route == "roadmap" and not _has_direct_qa_intent(latest_text)
    ):
        route = "roadmap"
        dumped["route"] = route
        dumped["intent_label"] = "reading_roadmap"
        dumped["user_visible_text"] = (
            "특정 주제를 단계적으로 탐구하려는 의도가 보여요. "
            "한 권이 아니라 입문서 → 대중서 → 심화서로 이어지는 독서 경로를 설계하는 게 적합해 보여요. "
            "현재 수준과 상황에 맞춰 로드맵을 짜볼게요."
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
    if route == "vector_db" and not dumped.get("mentioned_books"):
        dumped["mentioned_books"] = _regex_reference_books(latest_text)
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
                    "mentioned_books": dumped.get("mentioned_books"),
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
                "content": (
                    "당신은 친절한 도서 추천 비서다. 한국어로 핵심 위주로 답해라. "
                    "사용자의 장기기억이 주어지면 자연스럽게 활용하고(예: 지난번 좋아한 책 언급), "
                    "사용자가 책에 대한 감상/피드백을 말하면(intent_label=reading_feedback) 기억해두겠다고 짧게 확인하고, "
                    "새 책 목록은 제시하지 말고 이 취향으로 추천받고 싶은지 물어봐라."
                ),
            },
            {
                "role": "user",
                "content": (
                    "아래 의도 분석을 반영해서 답변해줘.\n"
                    f"intention={intention}\n"
                    f"request_text={state.get('request_text')}\n"
                    f"image_analysis={state.get('image_analysis')}\n"
                    f"[user_long_term_memory]\n{profile_to_prompt(state.get('user_profile'))}\n"
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

    roadmap = state.get("roadmap")
    personal_intro = state.get("personal_intro")
    enrichment = state.get("enrichment") or {}

    if roadmap:
        payload = {
            "output_type": "READING_ROADMAP",
            "text": roadmap.get("intro") or "",
            "html": None,
            "items": [book for stage in roadmap.get("stages", []) for book in stage.get("books", [])],
            "roadmap": roadmap,
        }
    # No summarization: return DB fields as-is for fastest response.
    elif db_items:
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
            "personal_intro": personal_intro,
            "enrichment": enrichment,
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
