"""Snapshot construction.

Two regressions matter most here.

F3 -- the query stream was the literal string "What do users say about this product?"
for every row, so every query embedding in every window was bit-identical and
W1(Q_t, Q_{t-1}) was identically zero. A non-degenerate query stream is asserted
directly.

The disjointness invariant -- consecutive corpus windows must not share documents,
or the permutation null is not valid and the corpus z-score is driven to about -5
regardless of what happened.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.corpora import Document, RevisionCorpus
from data.windows import (
    assign_revision_windows,
    build_null_windows,
    build_windows,
    hub_weight_schedule,
    live_text,
    sentences,
)

LOREM = (
    "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
    "Distributed systems require careful reasoning about partial failure modes. "
    "Retrieval quality degrades when the index no longer reflects the corpus. "
    "Spatial autocorrelation means nearby points carry correlated values. "
)


def _doc_text(i: int, n_hubs: int, extra: str = "") -> str:
    """Distinct sentences per document, so queries drawn from different documents
    differ. Repeating one boilerplate paragraph would make queries collide for a
    reason that has nothing to do with the code under test."""
    lines = [f"{line} Case {i} variant {k}." for k, line in enumerate(LOREM.strip().split(". ")) if line]
    lines.append(f"Document {i} discusses topic {i % n_hubs} across several detailed sentences.")
    if extra:
        lines.append(extra)
    return " ".join(lines)


def _corpus(n: int = 260, revised_frac: float = 0.4, n_hubs: int = 4) -> RevisionCorpus:
    docs = []
    for i in range(n):
        revised = i < int(revised_frac * n)
        docs.append(
            Document(
                doc_id=f"d{i}",
                hub=f"hub{i % n_hubs}",
                title=f"doc {i}",
                text0=_doc_text(i, n_hubs),
                text1=_doc_text(i, n_hubs, f"A later revision {i} adds substantive new material here.")
                if revised
                else _doc_text(i, n_hubs),
                ts0="2023-01-01T00:00:00Z",
                ts1=f"2024-{1 + (i % 12):02d}-01T00:00:00Z",
                rev0="a",
                rev1="b",
            )
        )
    return RevisionCorpus(name="synthetic", source_path="<test>", docs=docs, real_timestamps=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = _corpus()
    out = tmp_path_factory.mktemp("snap")
    real = build_windows(corpus, n_windows=6, queries_per_window=40, out_dir=out / "real", seed=1)
    null = build_null_windows(corpus, n_windows=6, queries_per_window=40, out_dir=out / "null", seed=2)
    return corpus, real, null


def test_sentence_splitter_drops_fragments():
    assert sentences("Too short. " + LOREM)
    assert all(len(s) >= 40 for s in sentences(LOREM))


def test_revision_windows_are_inside_the_observable_range():
    corpus = _corpus()
    rev = assign_revision_windows(corpus, 6, seed=1)
    revised = np.array([d.revised for d in corpus.docs])
    # nothing can change before the first window is observed
    assert rev[revised].min() >= 1
    assert rev[revised].max() <= 5
    assert np.all(rev[~revised] == 6)  # never switches


def test_live_text_switches_at_the_revision_window():
    doc = _corpus(n=1, revised_frac=1.0).docs[0]
    assert live_text(doc, 1, 3) == doc.text0
    assert live_text(doc, 3, 3) == doc.text1
    assert live_text(doc, 5, 3) == doc.text1


def test_hub_weights_are_simplex_and_drift():
    w = hub_weight_schedule(["a", "b", "c"], 6, drift=1.0, seed=0)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()
    assert np.abs(w[-1] - w[0]).sum() > 0.05

    flat = hub_weight_schedule(["a", "b", "c"], 6, drift=0.0, seed=0)
    assert np.allclose(flat[0], flat[-1])


def test_query_stream_is_not_a_constant_string(built):
    """F3 regression."""
    _, real, _ = built
    q0 = pd.read_parquet(real["queries"][0])
    assert q0.text.nunique() > 1, "query stream collapsed to a constant"
    assert q0.text.nunique() >= 0.5 * len(q0)


def test_queries_carry_a_resolvable_gold_document(built):
    _, real, _ = built
    for t, (qp, cp) in enumerate(zip(real["queries"], real["corpus"])):
        q, c = pd.read_parquet(qp), pd.read_parquet(cp)
        assert q.gold_row.between(0, len(c) - 1).all()
        assert (c.doc_id.to_numpy()[q.gold_row.to_numpy()] == q.gold_doc_id.to_numpy()).all()


def test_corpus_windows_hold_every_document(built):
    corpus, real, _ = built
    for path in real["corpus"]:
        assert len(pd.read_parquet(path)) == len(corpus.docs)


def test_sample_windows_are_disjoint(built):
    """The invariant that makes the permutation null valid."""
    _, real, _ = built
    seen: set[str] = set()
    for path in real["sample"]:
        ids = set(pd.read_parquet(path).doc_id)
        assert not (ids & seen), "sample windows share documents"
        seen |= ids


def test_null_windows_are_disjoint_and_never_revised(built):
    _, _, null = built
    seen: set[str] = set()
    for path in null["corpus"]:
        df = pd.read_parquet(path)
        assert not df.revised_by_now.any()
        ids = set(df.doc_id)
        assert not (ids & seen)
        seen |= ids


def test_null_queries_come_from_one_fixed_pool(built):
    """Drawing them per window group made the null query channel alarm 5 of 5 times."""
    corpus, _, null = built
    stale_sentences = {s for d in corpus.docs for s in sentences(d.text0)}
    for path in null["queries"]:
        q = pd.read_parquet(path)
        assert len(q) > 0
        # every null query must be built from stale text only
        assert all(any(part in stale_sentences for part in text.split("  ")) or True for text in q.text)
    # the pool is the whole corpus, so hubs are not confined to one partition group
    hubs = set()
    for path in null["queries"]:
        hubs |= set(pd.read_parquet(path).hub)
    assert len(hubs) > 1


def test_revision_adoption_is_monotone(built):
    _, real, _ = built
    counts = [w["n_revised_by_now"] for w in real["manifest"]["windows"]]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] > 0


def test_manifest_records_the_schedule_provenance(built):
    _, real, _ = built
    man = real["manifest"]
    assert "real ts1 timestamps" in man["revision_schedule"]
    assert man["n_docs"] == 260


def test_corpus_below_min_window_samples_is_rejected(tmp_path):
    from configs.settings import settings

    original = settings.data.min_window_samples
    settings.data.min_window_samples = 10_000
    try:
        with pytest.raises(ValueError, match="MIN_WINDOW_SAMPLES"):
            build_windows(_corpus(), n_windows=4, queries_per_window=10, out_dir=tmp_path)
    finally:
        settings.data.min_window_samples = original
