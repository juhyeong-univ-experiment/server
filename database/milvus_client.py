from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from database.config import get_env
from database.embeddings import encode_query

_connected = False
DEFAULT_MILVUS_COLLECTION = "book_embeddings"
logger = logging.getLogger(__name__)


def _resolve_id_field(collection: Collection, preferred: str) -> str:
    """
    Resolve ID field safely from collection schema.
    Priority:
    1) preferred field if exists
    2) primary key field
    3) common fallback names
    """
    field_names = [field.name for field in collection.schema.fields]
    if preferred in field_names:
        return preferred

    for field in collection.schema.fields:
        if getattr(field, "is_primary", False):
            return field.name

    for candidate in ("id", "book_id", "item_id", "pk"):
        if candidate in field_names:
            return candidate

    raise ValueError(
        f"Unable to resolve Milvus id field. available_fields={field_names}, preferred={preferred}"
    )


def _resolve_vector_dim(collection: Collection, vector_field: str) -> int:
    for field in collection.schema.fields:
        if field.name == vector_field:
            dim = field.params.get("dim")
            if dim is None:
                raise ValueError(f"Vector field '{vector_field}' has no dim in schema.")
            return int(dim)
    raise ValueError(f"Vector field '{vector_field}' not found in collection schema.")


def connect_milvus() -> None:
    global _connected
    if _connected:
        return
    host = get_env("MILVUS_HOST", "localhost")
    port = get_env("MILVUS_PORT", "19530")
    alias = get_env("MILVUS_ALIAS", "default")
    if not host or not port:
        raise ValueError("MILVUS_HOST and MILVUS_PORT must be set in .env")
    connections.connect(alias=alias, host=host, port=port)
    utility.list_collections()
    logger.info("Milvus connection check passed")
    _connected = True


def get_or_create_milvus_collection(collection_name: str, vector_dim: int) -> Collection:
    connect_milvus()
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        schema_dim = next(
            (
                field.params.get("dim")
                for field in collection.schema.fields
                if field.name == "embedding"
            ),
            None,
        )
        if schema_dim != vector_dim:
            raise ValueError(
                f"Existing Milvus collection dim is {schema_dim}, but embedding dim is {vector_dim}."
            )
        return collection

    fields = [
        FieldSchema(
            name="mongo_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=24,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]
    schema = CollectionSchema(fields=fields, description="Book embedding vectors")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
    )
    logger.info("Created Milvus collection '%s' with dim=%s", collection_name, vector_dim)
    return collection


def _embed_query(query_text: str, target_dim: int | None = None) -> list[float]:
    vector = encode_query(query_text)
    if target_dim is not None and len(vector) != target_dim:
        raise ValueError(
            f"Embedding dimension mismatch: model={len(vector)} collection={target_dim}. "
            "Check EMBEDDING_MODEL_NAME used for ingest/query."
        )
    return vector


def search_similar_items(query_text: str, top_k: int = 5) -> dict[str, Any]:
    connect_milvus()
    collection_name = get_env("MILVUS_COLLECTION", DEFAULT_MILVUS_COLLECTION)
    vector_field = get_env("MILVUS_VECTOR_FIELD", "embedding")
    id_field_preferred = get_env("MILVUS_ID_FIELD", "mongo_id")
    score_metric = get_env("MILVUS_METRIC_TYPE", "COSINE")

    collection = Collection(collection_name)
    collection.load()
    id_field = _resolve_id_field(collection, id_field_preferred)
    vector_dim = _resolve_vector_dim(collection, vector_field)
    vector = _embed_query(query_text, target_dim=vector_dim)

    results = collection.search(
        data=[vector],
        anns_field=vector_field,
        param={"metric_type": score_metric, "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=[id_field],
    )

    items: list[dict[str, Any]] = []
    if results and len(results) > 0:
        for hit in results[0]:
            item_id = None
            entity = getattr(hit, "entity", None)
            if entity is not None:
                item_id = entity.get(id_field)
            if item_id is None:
                item_id = getattr(hit, "id", None)
            items.append({"id": str(item_id), "score": float(hit.score)})

    return {
        "query": query_text,
        "items": items,
        "id_field": id_field,
        "vector_dim": vector_dim,
    }
