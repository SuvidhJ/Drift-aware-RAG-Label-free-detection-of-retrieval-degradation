"""Deterministic query pools for measuring what staleness costs.

Every query is a two-sentence window taken from a document, so its gold document is
known.  What differs between the pools is *which* text the window comes from, and that
decides whether a stale index can possibly answer it:

``new``        text present only in the revised version.  A stale index does not
               contain it.  This is the maximally staleness-sensitive pool, and the
               one every number in FINDINGS.md was measured on.
``retained``   text present in BOTH versions of a revised document.  A stale index
               contains it verbatim; a refreshed one contains it diluted by whatever
               was added.  This is the pool that exposes the cost of re-indexing.
``unchanged``  text from documents that were never revised.  Staleness cannot touch it.

A realistic query stream is a mixture of all three in the corpus's own proportions
(see ``measure.realistic_mix``).  Measuring on ``new`` alone, as earlier work here
did, reports an upper bound and calls it the effect.

The RNG call pattern below is reproduced exactly from the scripts that produced the
cached embeddings (draws happen only when a candidate pool is non-empty), so the pools
are bit-for-bit reproducible and ``verify_against_cache`` can check them.
"""
from __future__ import annotations

import re

import numpy as np

from drift.freshness import canonical

SENT = re.compile(r"(?<=[.!?])\s+")
MIN_SENT = 40      # characters; shorter fragments are headings and list debris
MIN_QUERY = 60     # characters; a query must carry enough text to be specific


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text) if len(s.strip()) > MIN_SENT]


def _window(pool: list[str], rng: np.random.Generator) -> str:
    j = int(rng.integers(0, len(pool)))
    return " ".join(pool[j : j + 2])


def new_text(docs: list[dict], seed: int = 7) -> tuple[list[str], np.ndarray]:
    """Windows of text that exists only in the revised version."""
    rng = np.random.default_rng(seed)
    qs, gold = [], []
    for i, d in enumerate(docs):
        s1 = sentences(d["text1"])
        if not s1:
            continue
        s0 = set(sentences(d["text0"]))
        pool = [s for s in s1 if s not in s0]
        if not pool:
            continue
        w = _window(pool, rng)
        if len(w) >= MIN_QUERY:
            qs.append(w)
            gold.append(i)
    return qs, np.array(gold, dtype=np.int64)


def retained_text(docs: list[dict], seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Windows of text present in both versions of a revised document."""
    rng = np.random.default_rng(seed)
    qs, gold = [], []
    for i, d in enumerate(docs):
        if canonical(d["text0"]) == canonical(d["text1"]):
            continue
        s0 = set(sentences(d["text0"]))
        shared = [x for x in sentences(d["text1"]) if x in s0]
        if not shared:
            continue
        w = _window(shared, rng)
        if len(w) >= MIN_QUERY:
            qs.append(w)
            gold.append(i)
    return qs, np.array(gold, dtype=np.int64)


def unchanged_text(
    docs: list[dict], seed: int = 0, limit: int = 1400
) -> tuple[list[str], np.ndarray]:
    """Windows from documents that were never revised."""
    rng = np.random.default_rng(seed)
    unch = [i for i, d in enumerate(docs) if d["text0"] == d["text1"]]
    qs, gold = [], []
    if not unch:
        return qs, np.array(gold, dtype=np.int64)
    for i in rng.choice(unch, min(limit, len(unch)), replace=False):
        s = sentences(docs[i]["text1"])
        if not s:
            continue
        w = _window(s, rng)
        if len(w) >= MIN_QUERY:
            qs.append(w)
            gold.append(int(i))
    return qs, np.array(gold, dtype=np.int64)


POOLS = {"new": new_text, "retained": retained_text, "unchanged": unchanged_text}
