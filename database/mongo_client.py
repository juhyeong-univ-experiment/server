from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

from database.config import get_env

_mongo_client: MongoClient | None = None
DEFAULT_MONGO_COLLECTION = "book_records"

# Fields returned to the agent. Enriched books carry keywords/mood/difficulty etc.
BOOK_PROJECTION = {
    "_id": 1,
    "id": 1,
    "mongo_id": 1,
    "name": 1,
    "title": 1,
    "description": 1,
    "summary": 1,
    "author": 1,
    "genre": 1,
    "category": 1,
    "year": 1,
    "keywords": 1,
    "mood": 1,
    "emotional_arc": 1,
    "pace": 1,
    "difficulty": 1,
    "topics": 1,
    "source": 1,
    "source_url": 1,
    "enriched_at": 1,
}


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        uri = get_env("DATABASE_URL")
        _mongo_client = MongoClient(uri)
    return _mongo_client


def get_database():
    return get_mongo_client()[get_env("MONGO_DB_NAME", "univ_experiment")]


def get_books_collection():
    collection_name = get_env("MONGO_BOOKS_COLLECTION", DEFAULT_MONGO_COLLECTION)
    return get_database()[collection_name]


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace and subtitle so '데미안: 에밀 싱클레어' == '데미안'."""
    base = re.split(r"[:(\[]", title or "", maxsplit=1)[0]
    return re.sub(r"[\W_]+", "", base.lower())


def _to_public_book(doc: dict[str, Any], fallback_id: str | None = None) -> dict[str, Any]:
    resolved_id = (
        doc.get("id")
        or doc.get("mongo_id")
        or (str(doc.get("_id")) if doc.get("_id") else fallback_id)
    )
    return {
        "id": str(resolved_id),
        "name": doc.get("name") or doc.get("title") or f"Unknown-{fallback_id}",
        "author": doc.get("author"),
        "year": doc.get("year"),
        "category": doc.get("category") or [],
        "description": doc.get("description")
        or f"{doc.get('author', 'Unknown')} / {doc.get('genre', 'Unknown')}",
        "keywords": doc.get("keywords") or [],
        "mood": doc.get("mood"),
        "emotional_arc": doc.get("emotional_arc"),
        "pace": doc.get("pace"),
        "difficulty": doc.get("difficulty"),
        "topics": doc.get("topics") or [],
        "source": doc.get("source") or "seed",
        "source_url": doc.get("source_url"),
        "enriched": bool(doc.get("enriched_at")),
    }


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
            BOOK_PROJECTION,
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
        ordered.append(_to_public_book(doc, fallback_id=item_id))
    return ordered


def find_book_by_title(title: str) -> dict[str, Any] | None:
    """Exact normalized-title lookup (used to detect books missing from the DB)."""
    key = normalize_title(title)
    if not key:
        return None
    collection = get_books_collection()
    doc = collection.find_one({"title_key": key}, BOOK_PROJECTION)
    if doc is None:
        # Seed documents may predate title_key; fall back to a case-insensitive exact match.
        doc = collection.find_one(
            {"title": {"$regex": f"^{re.escape(title.strip())}$", "$options": "i"}}, BOOK_PROJECTION
        )
    return _to_public_book(doc) if doc else None


def get_raw_book(book_id: str) -> dict[str, Any] | None:
    try:
        return get_books_collection().find_one({"_id": ObjectId(book_id)})
    except (InvalidId, TypeError):
        return None


def insert_book(doc: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {**doc, "title_key": normalize_title(doc.get("title", "")), "createdAt": now, "updatedAt": now}
    result = get_books_collection().insert_one(payload)
    return str(result.inserted_id)


def update_book(book_id: str, fields: dict[str, Any]) -> None:
    get_books_collection().update_one(
        {"_id": ObjectId(book_id)},
        {"$set": {**fields, "updatedAt": datetime.now(timezone.utc)}},
    )


def list_enriched_books(limit: int = 50) -> list[dict[str, Any]]:
    docs = (
        get_books_collection()
        .find({"enriched_at": {"$exists": True}}, BOOK_PROJECTION)
        .sort("enriched_at", -1)
        .limit(limit)
    )
    return [_to_public_book(doc) | {"enriched_at": doc.get("enriched_at")} for doc in docs]


def count_books() -> dict[str, int]:
    collection = get_books_collection()
    return {
        "total": collection.estimated_document_count(),
        "enriched": collection.count_documents({"enriched_at": {"$exists": True}}),
    }


def prepare_mongo_document(raw: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "title": raw.get("title", ""),
        "title_key": normalize_title(raw.get("title", "")),
        "author": raw.get("author", ""),
        "year": raw.get("year"),
        "pages": raw.get("pages"),
        "category": raw.get("category", []),
        "description": raw.get("description", ""),
        "summary": raw.get("summary", ""),
        "embedding_text": raw.get("embedding_text", ""),
        "source": raw.get("source", "seed"),
        "createdAt": now,
        "updatedAt": now,
    }
