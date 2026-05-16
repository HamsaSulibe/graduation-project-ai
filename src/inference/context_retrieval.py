"""Retrieval helpers for FAISS-backed context lookup."""
from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import CONFIDENCE_THRESHOLD


def get_best_retrieval_score(retrieved: list[dict]) -> float:
    """Return the highest ``score`` from retrieved results (0.0 if empty)."""
    if not retrieved:
        return 0.0
    try:
        return max(r.get("score", 0.0) for r in retrieved)
    except (ValueError, TypeError):
        return 0.0

def is_retrieval_strong(
    retrieved: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """True when at least one result has score >= *threshold*."""
    if not retrieved:
        return False
    return get_best_retrieval_score(retrieved) >= threshold

def retrieve_relevant_context(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    cards: list[str],
    top_k: int = 3,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Encode *query* and return top-k plant cards above *threshold*."""
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        if score < threshold:
            continue
        results.append({"card": cards[idx], "score": round(float(score), 4)})
    return results
