from __future__ import annotations

from typing import Any


def profile_to_prompt(profile: dict[str, Any] | None) -> str:
    """Compact, LLM-readable summary of the long-term memory."""
    if not profile:
        return "(기억된 정보 없음)"
    lines = []
    if profile.get("liked"):
        lines.append(
            "좋아한 책: "
            + "; ".join(f"{e['title']}({e.get('reason') or '이유 미기재'})" for e in profile["liked"][-8:])
        )
    if profile.get("disliked"):
        lines.append(
            "싫어한 책: "
            + "; ".join(f"{e['title']}({e.get('reason') or '이유 미기재'})" for e in profile["disliked"][-8:])
        )
    if profile.get("interests"):
        lines.append("관심사: " + ", ".join(profile["interests"][-10:]))
    if profile.get("avoid"):
        lines.append("기피 소재: " + ", ".join(profile["avoid"][-10:]))
    if profile.get("preferences"):
        lines.append("선호 스타일: " + ", ".join(profile["preferences"][-10:]))
    if profile.get("knowledge_levels"):
        lines.append("주제별 지식수준: " + ", ".join(f"{k}={v}" for k, v in profile["knowledge_levels"].items()))
    if profile.get("recommended"):
        lines.append("이전에 추천한 책: " + ", ".join(e["title"] for e in profile["recommended"][-10:] if e.get("title")))
    return "\n".join(lines) or "(기억된 정보 없음)"
