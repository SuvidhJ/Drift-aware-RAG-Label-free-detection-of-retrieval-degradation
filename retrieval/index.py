"""Vector index with genuine incremental re-index support.

``NumpyIndex`` is the default: exact cosine search by one matmul.  At the corpus
sizes this project works with (10^3-10^4 documents) an approximate index buys
nothing and costs a heavyweight dependency, and exactness removes recall@k noise that
would otherwise be confounded with the drift effect being measured.

The important capability is ``reindex_rows``.  The previous pipeline called
``build_index(..., force=True)`` once per window, which deletes and rebuilds the whole
collection; there was no way to refresh part of an index, so the ``a3_reindex`` action
had nothing to call and existed only in an enum (FINDINGS.md F16).  Selective refresh
is the action the policy engine actually needs.

A Chroma backend is available behind the ``heavy`` extra for parity with the original.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@dataclass
class SearchResult:
    indices: np.ndarray  # (n_queries, k)
    scores: np.ndarray  # (n_queries, k)


class NumpyIndex:
    """Exact cosine index over a fixed document set."""

    def __init__(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str] | None = None,
        texts: list[str] | None = None,
        name: str = "corpus",
    ) -> None:
        self.embeddings = _normalize(embeddings)
        self.n = len(self.embeddings)
        self.doc_ids = list(doc_ids) if doc_ids is not None else [str(i) for i in range(self.n)]
        self.texts = list(texts) if texts is not None else [""] * self.n
        self.name = name

        if len(self.doc_ids) != self.n or len(self.texts) != self.n:
            raise ValueError(
                f"index {name!r}: {self.n} embeddings but {len(self.doc_ids)} ids and "
                f"{len(self.texts)} texts"
            )
        self.reindexed_rows: set[int] = set()

    @classmethod
    def from_frame(cls, df: pd.DataFrame, embeddings: np.ndarray, name: str = "corpus") -> "NumpyIndex":
        if len(df) != len(embeddings):
            raise ValueError(f"frame has {len(df)} rows but {len(embeddings)} embeddings")
        return cls(
            embeddings,
            doc_ids=df["doc_id"].astype(str).tolist() if "doc_id" in df else None,
            texts=df["text"].fillna("").astype(str).tolist() if "text" in df else None,
            name=name,
        )

    def search(self, queries: np.ndarray, top_k: int | None = None) -> SearchResult:
        k = min(top_k or settings.retrieval.top_k, self.n)
        Q = _normalize(queries)

        idx_out = np.empty((len(Q), k), dtype=np.int64)
        score_out = np.empty((len(Q), k), dtype=np.float32)

        for start in range(0, len(Q), 512):
            block = Q[start : start + 512] @ self.embeddings.T
            part = np.argpartition(-block, k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(block, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            idx_out[start : start + len(block)] = np.take_along_axis(part, order, axis=1)
            score_out[start : start + len(block)] = np.take_along_axis(part_scores, order, axis=1)

        return SearchResult(indices=idx_out, scores=score_out)

    def reindex_rows(self, rows: np.ndarray, fresh_embeddings: np.ndarray) -> int:
        """Replace the vectors for ``rows`` in place. This is action a3.

        Args:
            rows: row positions to refresh
            fresh_embeddings: either (len(rows), d) aligned to ``rows``, or the full
                (n, d) fresh matrix, from which the given rows are taken.

        Returns:
            Number of rows refreshed.
        """
        rows = np.asarray(rows, dtype=np.int64).ravel()
        if len(rows) == 0:
            return 0
        if rows.min() < 0 or rows.max() >= self.n:
            raise IndexError(f"row out of range for index of size {self.n}")

        fresh = np.asarray(fresh_embeddings, dtype=np.float32)
        block = fresh[rows] if len(fresh) == self.n else fresh
        if len(block) != len(rows):
            raise ValueError(
                f"expected {len(rows)} fresh vectors (or the full {self.n}-row matrix), got {len(fresh)}"
            )

        self.embeddings[rows] = _normalize(block)
        self.reindexed_rows.update(rows.tolist())
        logger.debug(f"index {self.name!r}: refreshed {len(rows)} of {self.n} rows")
        return len(rows)

    def copy(self) -> "NumpyIndex":
        dup = NumpyIndex(self.embeddings.copy(), self.doc_ids, self.texts, self.name)
        dup.reindexed_rows = set(self.reindexed_rows)
        return dup

    def documents(self, rows: np.ndarray) -> list[dict]:
        return [
            {"row": int(r), "id": self.doc_ids[int(r)], "text": self.texts[int(r)]}
            for r in np.asarray(rows).ravel()
        ]


class ChromaIndex:
    """Chroma-backed index. Requires the ``heavy`` extra."""

    def __init__(
        self,
        embeddings: np.ndarray,
        doc_ids: list[str],
        texts: list[str],
        name: str = "corpus",
        chroma_path: Path | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "the chroma retrieval backend needs the heavy extra:\n"
                "  uv pip install -e .[heavy]\n"
                f"(original error: {exc})"
            ) from exc

        self._client = chromadb.PersistentClient(
            path=str(chroma_path or settings.retrieval.chroma_path)
        )
        try:
            self._client.delete_collection(name)
        except Exception:  # collection simply did not exist
            pass
        self._col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self.name = name
        self.n = len(embeddings)
        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self._upsert(np.arange(self.n), _normalize(embeddings))

    def _upsert(self, rows: np.ndarray, embs: np.ndarray) -> None:
        for start in range(0, len(rows), 512):
            sl = slice(start, start + 512)
            block_rows = rows[sl]
            self._col.upsert(
                ids=[self.doc_ids[int(r)] for r in block_rows],
                documents=[self.texts[int(r)] for r in block_rows],
                embeddings=embs[sl].tolist(),
                metadatas=[{"row": int(r)} for r in block_rows],
            )

    def search(self, queries: np.ndarray, top_k: int | None = None) -> SearchResult:
        k = min(top_k or settings.retrieval.top_k, self.n)
        Q = _normalize(queries)
        res = self._col.query(query_embeddings=Q.tolist(), n_results=k, include=["metadatas", "distances"])
        idx = np.array([[int(m["row"]) for m in row] for row in res["metadatas"]], dtype=np.int64)
        scores = 1.0 - np.array(res["distances"], dtype=np.float32)
        return SearchResult(indices=idx, scores=scores)

    def reindex_rows(self, rows: np.ndarray, fresh_embeddings: np.ndarray) -> int:
        rows = np.asarray(rows, dtype=np.int64).ravel()
        if len(rows) == 0:
            return 0
        fresh = np.asarray(fresh_embeddings, dtype=np.float32)
        block = fresh[rows] if len(fresh) == self.n else fresh
        self._upsert(rows, _normalize(block))
        return len(rows)


def build_index(
    corpus_df: pd.DataFrame,
    embeddings: np.ndarray,
    name: str = "corpus",
    backend: str | None = None,
) -> NumpyIndex | ChromaIndex:
    backend = backend or settings.retrieval.backend
    if backend == "numpy":
        return NumpyIndex.from_frame(corpus_df, embeddings, name=name)
    if backend == "chroma":
        return ChromaIndex(
            embeddings,
            corpus_df["doc_id"].astype(str).tolist(),
            corpus_df["text"].fillna("").astype(str).tolist(),
            name=name,
        )
    raise ValueError(f"unknown retrieval backend {backend!r}; expected 'numpy' or 'chroma'")
