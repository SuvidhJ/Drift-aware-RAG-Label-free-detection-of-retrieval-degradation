"""Vector index and partial re-index.

``reindex_rows`` is the capability F16 found missing: three of five remediation
actions existed only in an enum because there was no way to refresh part of an index,
only ``build_index(force=True)`` which rebuilt the whole collection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.retrieval_quality import evaluate, share_of_gain
from retrieval.index import NumpyIndex


@pytest.fixture
def index_pair():
    rng = np.random.default_rng(9)
    n, d = 200, 16
    stale = rng.standard_normal((n, d))
    stale /= np.linalg.norm(stale, axis=1, keepdims=True)

    fresh = stale.copy()
    moved = np.arange(0, n, 4)  # a quarter of the corpus moves a long way
    step = rng.standard_normal((len(moved), d))
    fresh[moved] = step / np.linalg.norm(step, axis=1, keepdims=True)

    frame = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "text": [f"text {i}" for i in range(n)]})
    return frame, stale, fresh, moved


def test_search_finds_the_exact_vector(index_pair):
    frame, stale, _, _ = index_pair
    idx = NumpyIndex.from_frame(frame, stale)
    res = idx.search(stale[:20], top_k=1)
    assert np.array_equal(res.indices[:, 0], np.arange(20))
    assert np.allclose(res.scores[:, 0], 1.0, atol=1e-5)


def test_results_are_ordered_by_descending_score(index_pair):
    frame, stale, _, _ = index_pair
    res = NumpyIndex.from_frame(frame, stale).search(stale[:5], top_k=10)
    assert np.all(np.diff(res.scores, axis=1) <= 1e-6)


def test_reindex_updates_only_the_named_rows(index_pair):
    frame, stale, fresh, moved = index_pair
    idx = NumpyIndex.from_frame(frame, stale)
    subset = moved[:10]

    assert idx.reindex_rows(subset, fresh) == 10
    assert np.allclose(idx.embeddings[subset], fresh[subset], atol=1e-6)

    untouched = np.setdiff1d(np.arange(len(stale)), subset)
    assert np.allclose(idx.embeddings[untouched], stale[untouched], atol=1e-6)
    assert idx.reindexed_rows == set(subset.tolist())


def test_reindex_accepts_an_aligned_block(index_pair):
    frame, stale, fresh, moved = index_pair
    idx = NumpyIndex.from_frame(frame, stale)
    rows = moved[:5]
    idx.reindex_rows(rows, fresh[rows])
    assert np.allclose(idx.embeddings[rows], fresh[rows], atol=1e-6)


def test_reindex_rejects_bad_input(index_pair):
    frame, stale, fresh, _ = index_pair
    idx = NumpyIndex.from_frame(frame, stale)
    assert idx.reindex_rows(np.array([], dtype=int), fresh) == 0
    with pytest.raises(IndexError):
        idx.reindex_rows(np.array([9999]), fresh)
    with pytest.raises(ValueError):
        idx.reindex_rows(np.array([1, 2, 3]), fresh[:2])


def test_partial_reindex_recovers_part_of_the_recall_gap(index_pair):
    """The end-to-end property a3 exists to provide."""
    frame, stale, fresh, moved = index_pair
    queries, gold = fresh[moved], moved

    stale_recall = evaluate(NumpyIndex.from_frame(frame, stale), queries, gold).recall_at_k
    fresh_recall = evaluate(NumpyIndex.from_frame(frame, fresh), queries, gold).recall_at_k
    assert fresh_recall > stale_recall, "fixture has no headroom to recover"

    partial = NumpyIndex.from_frame(frame, stale)
    partial.reindex_rows(moved[: len(moved) // 2], fresh)
    got = evaluate(partial, queries, gold).recall_at_k

    assert stale_recall < got < fresh_recall
    assert 0.0 < share_of_gain(got, stale_recall, fresh_recall) < 1.0


def test_copy_is_independent(index_pair):
    frame, stale, fresh, moved = index_pair
    original = NumpyIndex.from_frame(frame, stale)
    duplicate = original.copy()
    duplicate.reindex_rows(moved[:5], fresh)
    assert np.allclose(original.embeddings[moved[:5]], stale[moved[:5]], atol=1e-6)


def test_frame_and_embedding_length_must_agree(index_pair):
    frame, stale, _, _ = index_pair
    with pytest.raises(ValueError):
        NumpyIndex.from_frame(frame, stale[:10])


def test_evaluate_validates_its_inputs(index_pair):
    frame, stale, _, _ = index_pair
    idx = NumpyIndex.from_frame(frame, stale)
    with pytest.raises(ValueError):
        evaluate(idx, stale[:5], np.arange(3))
    empty = evaluate(idx, stale[:0], np.arange(0))
    assert empty.n_queries == 0


def test_share_of_gain_is_undefined_without_headroom():
    assert np.isnan(share_of_gain(0.5, 0.5, 0.5))
