from __future__ import annotations

from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph

from agent.base.states import AgentState
from agent.database.nodes import database_node, vector_db_node
from agent.enrichment.nodes import enrichment_node
from agent.inference.nodes import (
    analyze_image_node,
    chat_completion_node,
    formatter_node,
    intention_checker_node,
)
from agent.memory.nodes import load_memory_node, memory_update_node, personalize_node
from agent.roadmap.nodes import roadmap_node
from agent.validation.nodes import input_type_checker_node


def _route_after_input_type(state: AgentState) -> str:
    return "analyze_image" if state.get("input_type") == "image" else "load_memory"


def _route_after_intention(state: AgentState) -> str:
    route = state.get("intention_tool_type")
    return route if route in ("vector_db", "roadmap") else "chat_completion"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input_type_checker", input_type_checker_node)
    graph.add_node("analyze_image", analyze_image_node)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("intention_checker", intention_checker_node)
    graph.add_node("vector_db", vector_db_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("database", database_node)
    graph.add_node("personalize", personalize_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("chat_completion", chat_completion_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("memory_update", memory_update_node)

    graph.add_edge(START, "input_type_checker")
    graph.add_conditional_edges(
        "input_type_checker",
        _route_after_input_type,
        {"analyze_image": "analyze_image", "load_memory": "load_memory"},
    )
    graph.add_edge("analyze_image", "load_memory")
    graph.add_edge("load_memory", "intention_checker")
    graph.add_conditional_edges(
        "intention_checker",
        _route_after_intention,
        {"vector_db": "vector_db", "roadmap": "roadmap", "chat_completion": "chat_completion"},
    )
    graph.add_edge("vector_db", "enrichment")
    graph.add_edge("enrichment", "database")
    graph.add_edge("database", "personalize")
    graph.add_edge("personalize", "formatter")
    graph.add_edge("roadmap", "formatter")
    graph.add_edge("chat_completion", "formatter")
    graph.add_edge("formatter", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile()


_APP = None


def get_graph():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


async def stream_events(
    request_text: str,
    image_url: str | None = None,
    image_b64: str | None = None,
    locale: str = "ko",
    user_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    app = get_graph()
    initial_state: AgentState = {
        "request_text": request_text,
        "locale": locale,
        "user_id": user_id,
        "user_profile": None,
        "memory_update": None,
        "input_type": None,
        "input_data": image_b64 or image_url,
        "image_url": image_url,
        "image_b64": image_b64,
        "image_analysis": None,
        "intention": None,
        "intention_tool_type": None,
        "vdb_result": None,
        "enrichment": None,
        "db_result": None,
        "book_recommendation": None,
        "personal_intro": None,
        "roadmap": None,
        "chat_result": None,
        "formatted": None,
        "events": [],
    }
    # "custom" carries live progress (enrichment/roadmap) emitted from inside long nodes.
    async for mode, chunk in app.astream(initial_state, stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield chunk
            continue
        for _, payload in chunk.items():
            for event in (payload or {}).get("events", []):
                yield event
