from typing import TypedDict, Annotated, Literal, Any
import operator


class AgentState(TypedDict):
    request_text: str
    locale: str

    # Backward-compatible input field + explicit image fields.
    input_type: Literal["image", "text", "unknown"] | None
    input_data: str | None
    image_url: str | None
    image_b64: str | None

    image_analysis: dict[str, Any] | None

    intention: dict[str, Any] | None
    intention_tool_type: Literal["vector_db", "chat_completion"] | None

    vdb_result: dict[str, Any] | None
    db_result: list[dict[str, Any]] | None
    book_recommendation: list[dict[str, Any]] | None
    chat_result: Annotated[str, operator.add] | None

    formatted: dict[str, Any] | None
    events: Annotated[list[dict[str, Any]], operator.add]
    