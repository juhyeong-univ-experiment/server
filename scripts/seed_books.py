"""Seed a small test subset (Goodreads refined sample) into MongoDB + Milvus.

Usage: python -m scripts.seed_books [--reset] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymilvus import utility  # noqa: E402

from database import connect_milvus, ensure_book_collection, upsert_book_embeddings  # noqa: E402
from database.milvus_client import _collection_name  # noqa: E402
from database.mongo_client import get_books_collection, get_database, prepare_mongo_document  # noqa: E402

SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "seed_books.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop books, profiles and vectors first")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    connect_milvus()
    books = get_books_collection()
    if args.reset:
        books.drop()
        get_database()["user_profiles"].drop()
        if utility.has_collection(_collection_name()):
            utility.drop_collection(_collection_name())
        print("reset done")
    ensure_book_collection()
    books.create_index("title_key")

    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))[: args.limit]
    docs = [prepare_mongo_document(item) for item in raw]
    result = books.insert_many(docs)
    ids = [str(_id) for _id in result.inserted_ids]
    upsert_book_embeddings(ids, [d["embedding_text"] for d in docs])
    print(f"seeded {len(ids)} books")


if __name__ == "__main__":
    main()
