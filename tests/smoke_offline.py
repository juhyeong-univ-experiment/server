"""Offline smoke test: real Mongo/Milvus/external APIs, stubbed LLM.

Verifies the plumbing of the three follow-up features without spending LLM tokens:
  [1] enrichment grows the pool, [2] roadmap output, [3] memory is persisted & reused.
Run: python tests/smoke_offline.py
"""
from __future__ import annotations

import ast
import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.enrichment.pipeline as pipeline_mod  # noqa: E402
import agent.inference.nodes as inference_mod  # noqa: E402
import agent.memory.nodes as memory_mod  # noqa: E402
import agent.roadmap.nodes as roadmap_mod  # noqa: E402
from agent.core.graph import stream_events  # noqa: E402
from database import count_books  # noqa: E402
from database.profile_store import delete_profile, get_profile  # noqa: E402


def _text(messages) -> str:
    return "\n".join(str(m.get("content")) for m in messages)


def _ids(text: str) -> list[str]:
    return re.findall(r"'id': '([0-9a-f]{24})'", text)


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, messages):
        t = _text(messages)
        name = getattr(self.schema, "__name__", "")
        S = self.schema
        if name == "IntentionDecision":
            mentioned = [{"title": "데미안", "title_en": "Demian", "author": "Hermann Hesse"}] if "데미안" in t else []
            return S(user_visible_text="x", intent_label="book_search", route="vector_db",
                     query_text="coming-of-age novel about self discovery", mentioned_books=mentioned, reason="x")
        if name == "ExtractedBook":
            title = re.search(r"요청 도서: (.+?) /", t).group(1)
            return S(matched_sources=["openlibrary", "wikipedia-en", "wikipedia-ko"], is_real_book=True, title=title, title_en=title, author="Stub Author", year=1919,
                     summary_ko=f"{title}의 요약입니다.", keywords=["성장", "자아"], mood="사색적인",
                     emotional_arc="혼란에서 각성으로", pace="slow", difficulty="popular", topics=["성장소설"],
                     category=["fiction", "classics"],
                     embedding_text_en=f"{title}. A coming-of-age classic about self discovery and identity.")
        if name == "SuggestedTitles":
            return S(titles=[{"title": "Thinking, Fast and Slow", "title_en": "Thinking, Fast and Slow", "author": "Daniel Kahneman"}])
        if name == "Personalization":
            ids = _ids(t)
            return S(intro="지난번에 좋아하신 책처럼 빠른 전개의 책을 먼저 골랐어요.",
                     books=[{"id": i, "keep": True, "conflict_evidence": "", "personal_reason": "빠른 전개 선호와 맞아요"} for i in ids])
        if name == "MemoryUpdate":
            if "빠른" in t.split("최근 추천했던")[0]:
                recent = ast.literal_eval(re.search(r"최근 추천했던 책 제목: (\[.*?\])", t).group(1))
                fb = [{"title": recent[0], "rating": "dislike", "reason": "지루해서"}] if recent else []
                return S(feedback=fb, interests=["성장소설"], avoid=["잔인한 묘사"], preferences=["빠른 전개"],
                         knowledge_levels=[], learned="빠른 전개를 좋아하고 잔인한 묘사는 피하고 싶어하시는 걸 기억했어요.")
            return S(learned="")
        if name == "RoadmapPlan":
            stages = [
                {"stage": s, "query_en": q, "canonical_books": [{"title": b, "title_en": b, "author": "x"}]}
                for s, q, b in [
                    ("intro", "introductory behavioral economics", "Nudge"),
                    ("popular", "popular psychology decision making", "Thinking, Fast and Slow"),
                    ("advanced", "judgment under uncertainty heuristics", "Misbehaving"),
                ]
            ]
            return S(topic="행동경제학", user_level="beginner", level_evidence="처음 공부", emotional_state="지침", stages=stages)
        if name == "CuratedRoadmap":
            view = ast.literal_eval(t.split("[candidates_by_stage]\n", 1)[1])
            stages = [{"stage": s, "goal": "g", "books": [{"id": c[0]["id"], "reason": "r"}], "bridge": "b"}
                      for s, c in view.items() if c]
            return S(intro="부담 없는 입문서부터 시작해요.", stages=stages, first_book_id=stages[0]["books"][0]["id"])
        raise AssertionError(f"unexpected schema {name}")


class FakeLLM:
    def with_structured_output(self, schema):
        return FakeStructured(schema)

    async def ainvoke(self, messages):
        return SimpleNamespace(content="stub answer")


def fake_get_chat_model(temperature: float = 0.0):
    return FakeLLM()


for mod in (pipeline_mod, inference_mod, memory_mod, roadmap_mod):
    mod.get_chat_model = fake_get_chat_model


async def run(text: str, user_id: str) -> tuple[list[dict], dict]:
    events = [e async for e in stream_events(request_text=text, user_id=user_id)]
    final = next(e for e in events if e["status"] == "FINAL")["data"]
    return events, final


async def main() -> None:
    uid = "smoke-test-user"
    delete_profile(uid)
    before = count_books()

    # [1] mentioned book missing from DB -> enriched & used as anchor
    events, final = await run("데미안 같은 성장소설 추천해줘", uid)
    enrich_evt = next(e for e in events if e["node"] == "enrichment" and e["status"] == "TOOL_DONE")
    progress = [e["message"] for e in events if e["status"] == "ENRICH_PROGRESS"]
    print("[1] progress:", *progress, sep="\n    ")
    print("[1] enrichment:", enrich_evt["data"]["reasons"], enrich_evt["data"]["anchors"])
    assert final["output_type"] == "BOOK_RECOMMENDATIONS" and final["items"]
    assert all(i["name"] != "데미안" for i in final["items"]), "anchor must not be recommended back"
    after = count_books()
    print("[1] pool:", before, "->", after)

    # [3] memory extraction + persisted
    events, final = await run("나는 전개가 빠른 책이 좋아. 잔인한 건 싫고. 첫 번째 책은 지루했어.", uid)
    profile = get_profile(uid)
    print("[3] profile:", {k: profile[k] for k in ("preferences", "avoid", "interests")}, "disliked=", [d["title"] for d in profile["disliked"]])
    assert "빠른 전개" in profile["preferences"] and profile["disliked"] and profile["disliked"][0]["book_id"]

    # [3] memory reused: personalized intro + disliked/recommended excluded
    events, final = await run("판타지 소설 추천해줘", uid)
    rec_ids = {e["book_id"] for e in profile["recommended"]}
    print("[3] intro:", final.get("personal_intro"))
    assert final.get("personal_intro")
    assert not rec_ids & {i["id"] for i in final["items"]}, "already recommended books must be excluded"

    # [2] roadmap
    events, final = await run("행동경제학을 처음부터 공부하고 싶어", uid)
    rm = final["roadmap"]
    print("[2] roadmap:", [(s["label"], [b["name"] for b in s["books"]]) for s in rm["stages"]], "start=", rm["start_stage"])
    assert final["output_type"] == "READING_ROADMAP" and rm["stages"]
    print("OK", count_books())
    delete_profile(uid)


if __name__ == "__main__":
    asyncio.run(main())
