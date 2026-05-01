from __future__ import annotations

from agent.base.events import make_event
from agent.base.states import AgentState


async def input_type_checker_node(state: AgentState) -> dict:
    has_image = bool(state.get("image_url") or state.get("image_b64"))
    has_text = bool((state.get("request_text") or "").strip())

    if has_image:
        input_type = "image"
        message = "이미지 입력으로 판단했어요. 이미지 분석을 시작할게요."
    elif has_text:
        input_type = "text"
        message = "텍스트 입력으로 판단했어요. 의도를 분석할게요."
    else:
        input_type = "unknown"
        message = "입력을 해석하기 어려워요. 대화형 응답으로 진행할게요."

    return {
        "input_type": input_type,
        "events": [make_event("THINKING", "input_type_checker", message, {"input_type": input_type})],
    }
