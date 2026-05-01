from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from database.config import get_env

_mongo_client: MongoClient | None = None
DEFAULT_MONGO_COLLECTION = "book_records"


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        uri = get_env("DATABASE_URL")
        _mongo_client = MongoClient(uri)
    return _mongo_client


def get_books_collection():
    client = get_mongo_client()
    db_name = get_env("MONGO_DB_NAME", "univ_experiment")
    collection_name = get_env("MONGO_BOOKS_COLLECTION", DEFAULT_MONGO_COLLECTION)
    return client[db_name][collection_name]


def fetch_books_by_ids(item_ids: list[str]) -> list[dict[str, Any]]:
    if not item_ids:
        return []
    collection = get_books_collection()
    object_ids = []
    for raw_id in item_ids:
        try:
            object_ids.append(ObjectId(raw_id))
        except (InvalidId, TypeError):
            continue

    docs = list(
        collection.find(
            {"$or": [{"id": {"$in": item_ids}}, {"mongo_id": {"$in": item_ids}}, {"_id": {"$in": object_ids}}]},
            {"_id": 1, "id": 1, "mongo_id": 1, "name": 1, "title": 1, "description": 1, "author": 1, "genre": 1},
        )
    )
    by_id: dict[str, dict[str, Any]] = {}
    for doc in docs:
        keys = [
            doc.get("id"),
            doc.get("mongo_id"),
            str(doc.get("_id")) if doc.get("_id") else None,
        ]
        for key in keys:
            if key:
                by_id[str(key)] = doc
    ordered: list[dict[str, Any]] = []
    for item_id in item_ids:
        doc = by_id.get(str(item_id))
        if not doc:
            continue
        resolved_id = (
            doc.get("id")
            or doc.get("mongo_id")
            or (str(doc.get("_id")) if doc.get("_id") else item_id)
        )
        ordered.append(
            {
                "id": str(resolved_id),
                "name": doc.get("name") or doc.get("title") or f"Unknown-{item_id}",
                "description": doc.get("description")
                or f"{doc.get('author', 'Unknown')} / {doc.get('genre', 'Unknown')}",
            }
        )
    return ordered


def prepare_mongo_document(raw: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "title": raw.get("title", ""),
        "author": raw.get("author", ""),
        "year": raw.get("year"),
        "pages": raw.get("pages"),
        "category": raw.get("category", []),
        "description": raw.get("description", ""),
        "summary": raw.get("summary", ""),
        "embedding_text": raw.get("embedding_text", ""),
        "createdAt": now,
        "updatedAt": now,
    }
