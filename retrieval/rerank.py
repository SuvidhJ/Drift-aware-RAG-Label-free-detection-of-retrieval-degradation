"""Maximal Marginal Relevance re-ranking - the query-drift remediation (a2).

Returns positions into the candidate array rather than reordered document dicts, so
callers can carry their own row indices and scores alongside without the re-ranker
needing to know their shape.
"""
from __future__ import annotations

import numpy as np

from configs.settings import settings


def mmr_rerank(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    top_k: int | None = None,
    mmr_lambda: float | None = None,
) -> np.ndarray:
    """Select positions maximising relevance while penalising redundancy.

    Args:
        query_emb: (d,) query vector
        candidate_embs: (n, d) candidate vectors, in retrieval order
        top_k: how many to return
        mmr_lambda: 1.0 is pure relevance, 0.0 pure diversity

    Returns:
        Positions into ``candidate_embs``, best first.
    """
    cfg = settings.retrieval
    k = top_k or cfg.top_k
    lam = cfg.mmr_lambda if mmr_lambda is None else mmr_lambda

    C = np.asarray(candidate_embs, dtype=np.float32)
    if len(C) == 0:
        return np.empty(0, dtype=np.int64)

    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    q = np.asarray(query_emb, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)

    relevance = C @ q
    k = min(k, len(C))

    selected: list[int] = []
    remaining = np.ones(len(C), dtype=bool)
    # running max similarity of each candidate to anything already selected
    redundancy = np.full(len(C), -np.inf, dtype=np.float32)

    for _ in range(k):
        if not selected:
            scores = relevance.copy()
        else:
            scores = lam * relevance - (1.0 - lam) * redundancy
        scores[~remaining] = -np.inf

        pick = int(np.argmax(scores))
        selected.append(pick)
        remaining[pick] = False
        redundancy = np.maximum(redundancy, C @ C[pick])

    return np.asarray(selected, dtype=np.int64)
