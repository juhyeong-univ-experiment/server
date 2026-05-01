from __future__ import annotations

from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph

from agent.base.states import AgentState
from agent.database.nodes import database_node, vector_db_node
from agent.inference.nodes import (
    analyze_image_node,
    chat_completion_node,
    formatter_node,
    intention_checker_node,
)
from agent.validation.nodes import input_type_checker_node


def _route_after_input_type(state: AgentState) -> str:
    return "analyze_image" if state.get("input_type") == "image" else "intention_checker"


def _route_after_intention(state: AgentState) -> str:
    return "vector_db" if state.get("intention_tool_type") == "vector_db" else "chat_completion"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("input_type_checker", input_type_checker_node)
    graph.add_node("analyze_image", analyze_image_node)
    graph.add_node("intention_checker", intention_checker_node)
    graph.add_node("vector_db", vector_db_node)
    graph.add_node("database", database_node)
    graph.add_node("chat_completion", chat_completion_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "input_type_checker")
    graph.add_conditional_edges(
        "input_type_checker",
        _route_after_input_type,
        {"analyze_image": "analyze_image", "intention_checker": "intention_checker"},
    )
    graph.add_edge("analyze_image", "intention_checker")
    graph.add_conditional_edges(
        "intention_checker",
        _route_after_intention,
        {"vector_db": "vector_db", "chat_completion": "chat_completion"},
    )
    graph.add_edge("vector_db", "database")
    graph.add_edge("database", "formatter")
    graph.add_edge("chat_completion", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


async def stream_events(
    request_text: str,
    image_url: str | None = None,
    image_b64: str | None = None,
    locale: str = "ko",
) -> AsyncGenerator[dict[str, Any], None]:
    app = build_graph()
    initial_state: AgentState = {
        "request_text": request_text,
        "locale": locale,
        "input_type": None,
        "input_data": image_b64 or image_url,
        "image_url": image_url,
        "image_b64": image_b64,
        "image_analysis": None,
        "intention": None,
        "intention_tool_type": None,
        "vdb_result": None,
        "db_result": None,
        "book_recommendation": None,
        "chat_result": None,
        "formatted": None,
        "events": [],
    }
    async for chunk in app.astream(initial_state, stream_mode="updates"):
        for _, payload in chunk.items():
            for event in payload.get("events", []):
                yield event
