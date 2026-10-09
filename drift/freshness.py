"""Event-level supersession: serve the current version, never the stale one.

Why this exists
---------------
The distributional detectors in this package answer "has the corpus moved?" with a
z-score.  That question turns out to be the wrong one for staleness: counting the
documents that were edited predicts retrieval damage at R^2 0.92, while the full
embedding geometry adds a partial correlation of ~0.02 on top (p = 0.80).  The
geometry is dominated by the edit *event*, so the instrument should be the event.

This module records those events and acts on them.  It is deliberately not
statistical: every decision here is a content hash, an identity match, or a
near-duplicate test with a stated threshold, and every decision is logged with the
reason it was taken.

The failure mode it removes
---------------------------
A revised document is ingested while the previous revision is still indexed.  Both
versions are now searchable, they are near-identical in embedding space, and which
one a query lands on is arbitrary.  Retrieval silently serves text that no longer
exists.  `active_rows()` is the fix: superseded rows leave the searchable set at the
moment the replacement is ingested.

Four event kinds
----------------
``new``         previously unseen content and unseen identity
``edit``        same document identity, different content -> supersedes its prior row
``duplicate``   byte-identical content already indexed -> ignored, nothing to do
``republish``   new identity, but near-identical content to an indexed document
                (same text re-posted under a new id) -> supersedes it

``republish`` is the one that needs both tests.  Embedding similarity alone will
happily call two different papers on the same topic "the same document", so it is
gated on character-shingle Jaccard as well; a pair must look the same to both.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

_WS = re.compile(r"\s+")


def canonical(text: str) -> str:
    """Whitespace- and case-normalised text, so trivial reformatting is not an edit."""
    return _WS.sub(" ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles of the canonical text."""
    c = canonical(text)
    if len(c) <= k:
        return {c} if c else set()
    return {c[i : i + k] for i in range(len(c) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class IngestEvent:
    """One ingestion decision, with the evidence that produced it."""

    row: int
    doc_id: str
    kind: str                      # new | edit | duplicate | republish
    timestamp: str | None = None
    supersedes_row: int | None = None
    supersedes_doc: str | None = None
    similarity: float | None = None
    text_overlap: float | None = None
    version: int = 1

    def describe(self) -> str:
        if self.kind == "new":
            return f"new       {self.doc_id}"
        if self.kind == "duplicate":
            return f"duplicate {self.doc_id}  (identical content already indexed)"
        if self.kind == "edit":
            return (
                f"edit      {self.doc_id}  v{self.version}  "
                f"supersedes row {self.supersedes_row}  "
                f"(content moved {1.0 - (self.similarity or 0.0):.3f})"
            )
        return (
            f"republish {self.doc_id}  supersedes {self.supersedes_doc} "
            f"(cos {self.similarity:.3f}, text overlap {self.text_overlap:.2f})"
        )


@dataclass
class FreshnessRegistry:
    """Tracks which indexed rows are current, and why the others are not.

    Args:
        near_dup_cosine: cosine at or above which two rows are *candidates* for
            being the same document under different ids
        near_dup_overlap: character-shingle Jaccard that a candidate must also clear
        shingle_k: shingle width for the overlap test
    """

    near_dup_cosine: float = 0.97
    near_dup_overlap: float = 0.60
    shingle_k: int = 5

    _emb: list[np.ndarray] = field(default_factory=list, init=False)
    _rows: list[dict] = field(default_factory=list, init=False)
    _by_hash: dict[str, int] = field(default_factory=dict, init=False)
    _by_doc: dict[str, int] = field(default_factory=dict, init=False)
    _events: list[IngestEvent] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        doc_id: str,
        text: str,
        embedding: np.ndarray,
        timestamp: str | None = None,
    ) -> IngestEvent:
        """Add one document and decide what it does to the searchable set."""
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        h = content_hash(text)

        # 1. this exact document, re-sent unchanged: a true no-op.
        seen = self._by_hash.get(h)
        if seen is not None and self._rows[seen]["doc_id"] == doc_id:
            ev = IngestEvent(
                row=seen, doc_id=doc_id, kind="duplicate", timestamp=timestamp
            )
            self._events.append(ev)
            return ev

        # 1b. identical content under a DIFFERENT id.  It still needs its own row:
        # treating it as a no-op loses the identity, and every later revision of it
        # then arrives as an unknown document rather than as an edit.  The older
        # identical row is superseded, because serving both is pure redundancy.
        #
        # This branch has never fired on the three tracked corpora -- arXiv's 102
        # shared abstracts all sit under a repeated pageid, which is case 1 -- so it
        # is guarded by tests only and remains unverified against real data.
        if seen is not None:
            return self._append(
                doc_id, text, emb, h, timestamp,
                kind="republish", supersedes=seen, similarity=1.0, text_overlap=1.0,
                version=self._rows[seen]["version"] + 1,
            )

        prior = self._by_doc.get(doc_id)

        # 2. same identity, new content: a revision.
        if prior is not None:
            sim = float(emb @ self._emb[prior])
            ev = self._append(
                doc_id, text, emb, h, timestamp,
                kind="edit", supersedes=prior, similarity=sim,
                version=self._rows[prior]["version"] + 1,
            )
            return ev

        # 3. new identity: is this the same content re-posted under a new id?
        cand, sim = self._nearest_active(emb)
        if cand is not None and sim >= self.near_dup_cosine:
            overlap = jaccard(
                shingles(text, self.shingle_k), self._rows[cand]["shingles"]
            )
            if overlap >= self.near_dup_overlap:
                return self._append(
                    doc_id, text, emb, h, timestamp,
                    kind="republish", supersedes=cand, similarity=sim,
                    text_overlap=overlap, version=self._rows[cand]["version"] + 1,
                )

        # 4. genuinely new.
        return self._append(doc_id, text, emb, h, timestamp, kind="new")

    def _nearest_active(self, emb: np.ndarray) -> tuple[int | None, float]:
        active = [i for i, r in enumerate(self._rows) if r["active"]]
        if not active:
            return None, 0.0
        M = np.stack([self._emb[i] for i in active])
        sims = M @ emb
        j = int(np.argmax(sims))
        return active[j], float(sims[j])

    def _append(
        self, doc_id, text, emb, h, timestamp, *, kind,
        supersedes=None, similarity=None, text_overlap=None, version=1,
    ) -> IngestEvent:
        row = len(self._rows)
        self._rows.append(
            {
                "doc_id": doc_id,
                "hash": h,
                "active": True,
                "version": version,
                "timestamp": timestamp,
                "shingles": shingles(text, self.shingle_k),
            }
        )
        self._emb.append(emb)
        self._by_hash[h] = row
        self._by_doc[doc_id] = row

        superseded_doc = None
        if supersedes is not None:
            self._rows[supersedes]["active"] = False
            superseded_doc = self._rows[supersedes]["doc_id"]

        ev = IngestEvent(
            row=row, doc_id=doc_id, kind=kind, timestamp=timestamp,
            supersedes_row=supersedes, supersedes_doc=superseded_doc,
            similarity=similarity, text_overlap=text_overlap, version=version,
        )
        self._events.append(ev)
        return ev

    # ------------------------------------------------------------------- views
    def active_rows(self) -> np.ndarray:
        """Rows that should be searchable. Superseded revisions are excluded."""
        return np.array([i for i, r in enumerate(self._rows) if r["active"]], dtype=np.int64)

    def stale_rows(self) -> np.ndarray:
        return np.array([i for i, r in enumerate(self._rows) if not r["active"]], dtype=np.int64)

    def embeddings(self) -> np.ndarray:
        return np.stack(self._emb) if self._emb else np.empty((0, 0))

    def events(self, kind: str | None = None) -> list[IngestEvent]:
        return [e for e in self._events if kind is None or e.kind == kind]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return {
            "ingested": len(self._events),
            "indexed_rows": len(self._rows),
            "active": int(len(self.active_rows())),
            "superseded": int(len(self.stale_rows())),
            **{f"event_{k}": v for k, v in sorted(counts.items())},
        }

    def report(self, limit: int = 12) -> str:
        s = self.summary()
        lines = [
            "FRESHNESS REGISTRY",
            f"  ingested {s['ingested']}  ->  {s['active']} searchable, "
            f"{s['superseded']} superseded and withheld",
        ]
        for k in ("new", "edit", "republish", "duplicate"):
            if s.get(f"event_{k}"):
                lines.append(f"    {k:<10} {s[f'event_{k}']}")
        interesting = [e for e in self._events if e.kind in ("edit", "republish")]
        if interesting:
            lines.append("  most recent supersessions:")
            for e in interesting[-limit:]:
                lines.append("    " + e.describe())
        return "\n".join(lines)


def stale_serve_rate(
    registry: FreshnessRegistry,
    queries: np.ndarray,
    top_k: int = 10,
) -> dict:
    """How often retrieval hands back text that no longer exists.

    This exists because **recall@k cannot see this failure**.  A superseded revision
    and its replacement carry the same document id, so a gold-document metric scores
    a hit either way: measured on arXiv, recall@10 is 0.9576 whether or not the old
    rows were withdrawn, while 17.4% of queries were being answered from a row that
    had already been replaced.  That is the same shape of defect as F10, where RAGAS
    faithfulness scored a stale-but-plausible context well.

    Searching over every row (rather than ``search_current``) is the point: this
    measures the damage done by *not* applying supersession.

    Returns:
        ``top1`` -- share of queries whose best match is superseded
        ``any_in_topk`` -- share with at least one superseded row in the top k
        ``mean_stale_in_topk`` -- average count of superseded rows per result page
    """
    E = registry.embeddings()
    if not len(E):
        return {"top1": 0.0, "any_in_topk": 0.0, "mean_stale_in_topk": 0.0}
    active = np.array([r["active"] for r in registry._rows])
    Q = np.asarray(queries, dtype=np.float64)
    k = min(top_k, len(E))
    t1 = any_k = cnt = 0
    for s in range(0, len(Q), 256):
        S = Q[s : s + 256] @ E.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), axis=1
        )
        stale = ~active[top]
        t1 += int(stale[:, 0].sum())
        any_k += int(stale.any(axis=1).sum())
        cnt += int(stale.sum())
    m = max(len(Q), 1)
    return {
        "top1": t1 / m,
        "any_in_topk": any_k / m,
        "mean_stale_in_topk": cnt / m,
    }


def search_current(
    registry: FreshnessRegistry,
    queries: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Top-k over the *current* rows only, returned as registry row indices.

    The whole point of the module: a superseded revision can never be returned,
    however well it matches.
    """
    active = registry.active_rows()
    if len(active) == 0:
        return np.empty((len(queries), 0), dtype=np.int64)
    M = registry.embeddings()[active]
    Q = np.asarray(queries, dtype=np.float64)
    k = min(top_k, len(active))
    out = np.empty((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 256):
        S = Q[s : s + 256] @ M.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), axis=1
        )
        out[s : s + 256] = active[ordered]
    return out
