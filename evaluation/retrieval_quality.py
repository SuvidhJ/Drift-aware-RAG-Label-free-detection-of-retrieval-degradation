"""The primary endpoint: does the index still return the right document?

Why not RAGAS faithfulness
--------------------------
Faithfulness scores whether an answer is entailed by the context it was given.  A
stale-but-plausible context scores *well* on it -- the model faithfully summarises
the outdated document it was handed.  The oracle was therefore blind to the exact
failure mode this project is about (FINDINGS.md F17), quite apart from being called
with a schema three versions old and having its exception swallowed, so faithfulness
came back empty for every window while the run still reported success (F10, F23).

Recall@k against a known gold document has none of those problems: it is defined
without an LLM, it moves when the index goes stale, and it is exactly the quantity a
selective re-index is trying to protect.  RAGAS remains available in
evaluation/ragas_harness.py for anyone who wants the secondary signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs.settings import settings


@dataclass
class RetrievalQuality:
    recall_at_k: float
    mrr: float
    k: int
    n_queries: int

    def as_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "k": self.k,
            "n_queries": self.n_queries,
        }


def evaluate(index, query_embs: np.ndarray, gold_rows: np.ndarray, k: int | None = None) -> RetrievalQuality:
    """Recall@k and MRR of ``index`` for queries whose gold document row is known."""
    k = k or settings.retrieval.top_k
    gold_rows = np.asarray(gold_rows, dtype=np.int64)

    if len(query_embs) != len(gold_rows):
        raise ValueError(f"{len(query_embs)} queries but {len(gold_rows)} gold rows")
    if len(gold_rows) == 0:
        return RetrievalQuality(recall_at_k=float("nan"), mrr=float("nan"), k=k, n_queries=0)

    result = index.search(query_embs, top_k=k)
    hits = result.indices == gold_rows[:, None]

    recall = float(hits.any(axis=1).mean())
    ranks = np.argmax(hits, axis=1)
    rr = np.where(hits.any(axis=1), 1.0 / (ranks + 1.0), 0.0)
    return RetrievalQuality(recall_at_k=recall, mrr=float(rr.mean()), k=k, n_queries=len(gold_rows))


def headroom(stale_quality: RetrievalQuality, fresh_quality: RetrievalQuality) -> float:
    """Recall a perfect re-index would recover. Selective re-index is vacuous at ~0."""
    return float(fresh_quality.recall_at_k - stale_quality.recall_at_k)


def share_of_gain(
    achieved: float, stale: float, fresh: float
) -> float:
    """Fraction of the available recall gain a partial re-index captured.

    This is the quantity the bake-off compares across selectors; reporting raw recall
    hides how much of it was already there before any refresh.
    """
    gain = fresh - stale
    if abs(gain) < 1e-9:
        return float("nan")
    return float((achieved - stale) / gain)
