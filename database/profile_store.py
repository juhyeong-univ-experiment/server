"""Long-term user memory (profile) stored in MongoDB."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from database.config import get_env
from database.mongo_client import get_database

MAX_LIST = 30
MAX_HISTORY = 50


def _profiles():
    return get_database()[get_env("MONGO_PROFILES_COLLECTION", "user_profiles")]


def empty_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "liked": [],  # [{book_id, title, reason, at}]
        "disliked": [],  # [{book_id, title, reason, at}]
        "interests": [],  # ["심리학", "우주"]
        "avoid": [],  # ["잔인한 묘사"]
        "preferences": [],  # ["빠른 전개", "짧은 분량"]
        "knowledge_levels": {},  # {"심리학": "beginner"}
        "recommended": [],  # [{book_id, title, at}]
        "turns": 0,
    }


def get_profile(user_id: str) -> dict[str, Any]:
    if not user_id:
        return empty_profile("")
    doc = _profiles().find_one({"_id": user_id})
    if not doc:
        return empty_profile(user_id)
    doc.pop("_id", None)
    return {**empty_profile(user_id), **doc}


def save_profile(profile: dict[str, Any]) -> None:
    user_id = profile.get("user_id")
    if not user_id:
        return
    data = {k: v for k, v in profile.items() if k != "user_id"}
    data["updated_at"] = datetime.now(timezone.utc)
    _profiles().update_one({"_id": user_id}, {"$set": data}, upsert=True)


def delete_profile(user_id: str) -> None:
    _profiles().delete_one({"_id": user_id})


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    seen = {s.strip().lower() for s in existing}
    merged = list(existing)
    for item in new:
        item = (item or "").strip()
        if item and item.lower() not in seen:
            merged.append(item)
            seen.add(item.lower())
    return merged[-MAX_LIST:]


def apply_feedback(
    profile: dict[str, Any],
    *,
    book_id: str | None,
    title: str,
    rating: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Register like/dislike. The same book can live in only one of the two lists."""
    now = datetime.now(timezone.utc).isoformat()
    key = (book_id or "", (title or "").strip().lower())

    def _same(entry: dict[str, Any]) -> bool:
        return (book_id and entry.get("book_id") == book_id) or (
            (entry.get("title") or "").strip().lower() == key[1]
        )

    profile["liked"] = [e for e in profile["liked"] if not _same(e)]
    profile["disliked"] = [e for e in profile["disliked"] if not _same(e)]
    entry = {"book_id": book_id, "title": title, "reason": reason, "at": now}
    target = "liked" if rating == "like" else "disliked"
    profile[target] = (profile[target] + [entry])[-MAX_LIST:]
    return profile


def apply_memory_update(profile: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    profile["interests"] = _merge_unique(profile["interests"], update.get("interests") or [])
    profile["avoid"] = _merge_unique(profile["avoid"], update.get("avoid") or [])
    profile["preferences"] = _merge_unique(profile["preferences"], update.get("preferences") or [])
    # A new stated interest overrides an old avoid entry and vice versa.
    avoid_lower = {a.lower() for a in update.get("avoid") or []}
    profile["interests"] = [i for i in profile["interests"] if i.lower() not in avoid_lower]
    levels = dict(profile.get("knowledge_levels") or {})
    for item in update.get("knowledge_levels") or []:
        if item.get("topic") and item.get("level"):
            levels[item["topic"]] = item["level"]
    profile["knowledge_levels"] = levels
    for fb in update.get("feedback") or []:
        if fb.get("title") and fb.get("rating") in ("like", "dislike"):
            apply_feedback(
                profile,
                book_id=fb.get("book_id"),
                title=fb["title"],
                rating=fb["rating"],
                reason=fb.get("reason"),
            )
    return profile


def add_recommended(profile: dict[str, Any], books: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    known = {e.get("book_id") for e in profile["recommended"]}
    for book in books:
        if book.get("id") and book["id"] not in known:
            profile["recommended"].append({"book_id": book["id"], "title": book.get("name"), "at": now})
            known.add(book["id"])
    profile["recommended"] = profile["recommended"][-MAX_HISTORY:]
    return profile


def has_memory(profile: dict[str, Any]) -> bool:
    return any(profile.get(k) for k in ("liked", "disliked", "interests", "avoid", "preferences"))
