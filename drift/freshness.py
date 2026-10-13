"""Text normalisation and character-shingle overlap.

These define what counts as a revision (``canonical``) and how closely two sentences
match (``shingles`` + ``jaccard``) throughout ``research/staleness``.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def canonical(text: str) -> str:
    """Whitespace- and case-normalised text, so trivial reformatting is not an edit."""
    return _WS.sub(" ", text).strip().lower()


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
