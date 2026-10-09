"""Supersession: the index must never serve a revision that has been replaced.

Several of these encode a specific defect rather than a generic property:

  * ``test_recall_cannot_see_the_bug`` pins the reason this module needs its own
    metric -- a gold-document metric scores a superseded revision as a hit, so
    recall@k is identical with and without the fix while users are being served
    text that no longer exists.
  * ``test_reformatting_is_not_an_edit`` guards the canonicaliser; without it every
    whitespace change would burn a supersession and a version number.
  * ``test_republish_needs_both_tests`` guards against embedding similarity alone
    collapsing two genuinely different documents into one.
"""
from __future__ import annotations

import numpy as np
import pytest

from drift.freshness import (
    FreshnessRegistry,
    canonical,
    content_hash,
    jaccard,
    search_current,
    shingles,
    stale_serve_rate,
)


def _vec(seed: int, d: int = 32) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=d)
    return v / np.linalg.norm(v)


def _near(v: np.ndarray, eps: float = 0.01, seed: int = 99) -> np.ndarray:
    n = np.random.default_rng(seed).normal(size=len(v))
    w = v + eps * n / np.linalg.norm(n)
    return w / np.linalg.norm(w)


# ------------------------------------------------------------------ primitives
def test_canonical_collapses_whitespace_and_case():
    assert canonical("  The   CAT \n sat ") == "the cat sat"


def test_hash_is_stable_under_reformatting():
    assert content_hash("A  b\nc") == content_hash("a b c")


def test_jaccard_bounds():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert 0.0 < jaccard({"a", "b"}, {"b", "c"}) < 1.0


def test_shingles_short_text():
    assert shingles("ab", k=5) == {"ab"}
    assert shingles("", k=5) == set()


# ----------------------------------------------------------------- event kinds
def test_new_document():
    r = FreshnessRegistry()
    ev = r.ingest("d1", "first document about otters", _vec(1))
    assert ev.kind == "new"
    assert list(r.active_rows()) == [0]


def test_identical_content_is_a_duplicate_and_adds_no_row():
    r = FreshnessRegistry()
    r.ingest("d1", "same text", _vec(1))
    ev = r.ingest("d1", "same   TEXT", _vec(1))
    assert ev.kind == "duplicate"
    assert len(r.active_rows()) == 1


def test_reformatting_is_not_an_edit():
    """Whitespace and case changes must not consume a version."""
    r = FreshnessRegistry()
    r.ingest("d1", "Hello   World", _vec(1))
    ev = r.ingest("d1", "hello world", _vec(1))
    assert ev.kind == "duplicate"
    assert r.summary()["superseded"] == 0


def test_edit_supersedes_prior_revision():
    r = FreshnessRegistry()
    r.ingest("d1", "version one of the text", _vec(1))
    ev = r.ingest("d1", "version two of the text, revised", _near(_vec(1)))
    assert ev.kind == "edit"
    assert ev.version == 2
    assert ev.supersedes_row == 0
    assert list(r.active_rows()) == [1]
    assert list(r.stale_rows()) == [0]


def test_republish_under_a_new_id_supersedes():
    r = FreshnessRegistry()
    text = "the quick brown fox jumps over the lazy dog repeatedly"
    r.ingest("d1", text, _vec(1))
    ev = r.ingest("d2", text + " today", _near(_vec(1), eps=0.005))
    assert ev.kind == "republish"
    assert ev.supersedes_doc == "d1"
    assert list(r.active_rows()) == [1]


def test_republish_needs_both_tests():
    """Near-identical embedding but unrelated text must stay a separate document."""
    r = FreshnessRegistry()
    r.ingest("d1", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", _vec(1))
    ev = r.ingest("d2", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", _near(_vec(1), eps=0.001))
    assert ev.kind == "new"
    assert len(r.active_rows()) == 2


def test_unrelated_document_is_new():
    r = FreshnessRegistry()
    r.ingest("d1", "otters and rivers", _vec(1))
    ev = r.ingest("d2", "compiler optimisation passes", _vec(2))
    assert ev.kind == "new"
    assert len(r.active_rows()) == 2


def test_three_revisions_leave_only_the_last():
    r = FreshnessRegistry()
    for i, t in enumerate(["draft one", "draft two", "draft three"]):
        r.ingest("d1", t, _near(_vec(1), eps=0.01 * (i + 1), seed=i))
    assert list(r.active_rows()) == [2]
    assert r.summary()["superseded"] == 2


# -------------------------------------------------------------------- retrieval
def test_search_never_returns_a_superseded_row():
    r = FreshnessRegistry()
    v = _vec(1)
    r.ingest("d1", "the original text of the document", v)
    r.ingest("d1", "the revised text of the document", _near(v, eps=0.02))
    stale = r.stale_rows()
    rows = search_current(r, np.stack([v, _vec(2)]), top_k=2)
    assert not set(rows.ravel()) & set(stale.tolist())


def test_search_current_handles_empty_registry():
    r = FreshnessRegistry()
    assert search_current(r, np.zeros((2, 8))).shape[1] == 0


def test_recall_cannot_see_the_bug():
    """A gold-document metric is identical with and without supersession.

    Both revisions carry doc id ``d1``, so either scores as a hit. The version-aware
    metric is what separates them. This is why stale_serve_rate exists.
    """
    r = FreshnessRegistry()
    v = _vec(1)
    r.ingest("d1", "the original text of the document", v)
    r.ingest("d1", "the revised text of the document", _near(v, eps=0.02))
    q = v[None, :]

    all_rows = np.argsort(-(q @ r.embeddings().T), axis=1)[:, :2]
    doc_ids = {r._rows[i]["doc_id"] for i in all_rows.ravel()}
    current = {r._rows[i]["doc_id"] for i in search_current(r, q, top_k=1).ravel()}
    assert doc_ids == current == {"d1"}          # recall@k: identical

    rate = stale_serve_rate(r, q, top_k=2)
    assert rate["any_in_topk"] == 1.0            # version-aware: catches it


def test_stale_serve_rate_zero_when_nothing_superseded():
    r = FreshnessRegistry()
    r.ingest("d1", "alpha text here", _vec(1))
    r.ingest("d2", "beta text here", _vec(2))
    rate = stale_serve_rate(r, np.stack([_vec(1)]), top_k=2)
    assert rate["top1"] == 0.0 and rate["any_in_topk"] == 0.0


def test_summary_and_report_are_consistent():
    r = FreshnessRegistry()
    r.ingest("d1", "one text", _vec(1))
    r.ingest("d1", "two text revised", _near(_vec(1)))
    r.ingest("d2", "unrelated material", _vec(5))
    s = r.summary()
    assert s["active"] + s["superseded"] == s["indexed_rows"] == 3
    assert "FRESHNESS REGISTRY" in r.report()


def test_cross_listed_document_edited_once_is_edited_once():
    """The same document ingested under several listings must edit once, not N times.

    arXiv cross-lists a paper into several categories, so one pageid occupies several
    corpus rows. Counting those rows as separate documents is what made edit recall
    look like 93.8% when every distinct document had in fact been detected.
    """
    r = FreshnessRegistry()
    v = _vec(1)
    r.ingest("p1", "the original abstract", v)          # listing A
    r.ingest("p1", "the original abstract", v)          # listing B, same paper
    assert r.summary()["event_duplicate"] == 1

    rev = _near(v, eps=0.02)
    assert r.ingest("p1", "the revised abstract", rev).kind == "edit"
    assert r.ingest("p1", "the revised abstract", rev).kind == "duplicate"
    assert len(r.events("edit")) == 1
    assert list(r.active_rows()) == [1]


def test_identical_content_under_a_new_id_keeps_its_identity():
    """A second document sharing text byte-for-byte must still be addressable.

    Swallowing it as a no-op would lose its id, and every later revision of it would
    then arrive as an unknown document instead of as an edit.
    """
    r = FreshnessRegistry()
    text = "an abstract that two different submissions happen to share exactly"
    r.ingest("p1", text, _vec(1))
    ev = r.ingest("p2", text, _vec(1))
    assert ev.kind == "republish" and ev.supersedes_doc == "p1"
    assert r.ingest("p2", text + " now revised", _near(_vec(1))).kind == "edit"
