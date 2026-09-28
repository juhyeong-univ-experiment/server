"""End-to-end scenario against a running server (real LLM).

Run: python tests/e2e_ws.py [ws://localhost:8000/ws/chat]
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/chat"
SCENARIO = [
    "데미안 같은 성장소설 추천해줘.",
    "나는 전개가 빠른 책을 좋아하고, 잔인한 묘사는 싫어해. 첫 번째 추천 책은 너무 지루했어.",
    "판타지 소설 추천해줘.",
    "행동경제학을 처음부터 공부하고 싶어. 요즘 좀 지쳐 있어서 너무 어려운 건 부담스러워.",
]


async def turn(ws, text: str) -> dict:
    await ws.send(json.dumps({"text": text, "locale": "ko"}))
    final = {}
    while True:
        msg = json.loads(await ws.recv())
        status, node = msg["status"], msg.get("node")
        if status in ("THINKING", "TOOL_DONE", "ENRICH_PROGRESS", "ROADMAP_PROGRESS", "MEMORY") and msg.get("message"):
            print(f"  [{status}:{node}] {msg['message']}")
        if status == "FINAL":
            final = msg["data"]
        if status == "DONE":
            return final


async def main() -> None:
    user_id = f"e2e-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(f"{URL}?user_id={user_id}", max_size=None) as ws:
        await ws.recv()  # CONNECTED
        for text in SCENARIO:
            print(f"\n>>> {text}")
            final = await turn(ws, text)
            if final.get("personal_intro"):
                print("  intro:", final["personal_intro"])
            if final.get("output_type") == "READING_ROADMAP":
                rm = final["roadmap"]
                print("  roadmap:", rm["intro"])
                for st in rm["stages"]:
                    print(f"   - {st['label']}: " + ", ".join(b["name"] for b in st["books"]))
            elif final.get("items"):
                for item in final["items"][:5]:
                    tag = " [NEW]" if item.get("enriched") else ""
                    print(f"   - {item['name']}{tag}")
            else:
                print("  text:", (final.get("text") or "")[:300])
    print("\nuser_id:", user_id)


if __name__ == "__main__":
    asyncio.run(main())
