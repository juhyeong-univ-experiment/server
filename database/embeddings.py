from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from database.config import get_env

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_name = get_env("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    return SentenceTransformer(model_name)


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError("Failed to infer embedding dimension from SentenceTransformer model.")
    return int(dim)


def encode_query(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector.tolist()


def encode_texts(texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(
        list(texts),
        batch_size=min(batch_size, len(texts) if texts else 1),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()
