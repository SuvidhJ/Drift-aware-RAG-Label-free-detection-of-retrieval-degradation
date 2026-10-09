"""Query topic drift reported as composition rather than as a distance.

``test_knn_beats_centroid_on_overlapping_topics`` encodes the measurement that set
the default: on the arXiv corpus, nearest-centroid assignment is 67.1% correct
against the documents' own categories and kNN at k=5 is 73.7%.  Overlapping topics
put centroids close enough together that the nearest one is often wrong, and the
error surfaces as traffic appearing in topics that received none.
"""
from __future__ import annotations

import numpy as np
import pytest

from drift.topics import TopicModel, analyse, coverage_gap


def _world(d=24, per=40, seed=0):
    """Three well-separated topics, plus queries drawn from each."""
    rng = np.random.default_rng(seed)
    axes = np.eye(3, d)
    X, labels = [], []
    for i, nm in enumerate(["alpha", "beta", "gamma"]):
        pts = axes[i] * 3 + rng.normal(size=(per, d)) * 0.25
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        X.append(pts)
        labels += [nm] * per
    return np.vstack(X), labels, axes


def _queries(axes, i, m, d=24, seed=1):
    rng = np.random.default_rng(seed)
    q = axes[i] * 3 + rng.normal(size=(m, d)) * 0.25
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ------------------------------------------------------------------ the model
def test_from_labels_builds_one_anchor_per_topic():
    X, labels, _ = _world()
    m = TopicModel.from_labels(X, labels)
    assert m.names == ["alpha", "beta", "gamma"]
    assert m.centroids.shape == (3, X.shape[1])
    assert list(m.doc_counts) == [40, 40, 40]


def test_centroids_are_unit_norm():
    X, labels, _ = _world()
    m = TopicModel.from_labels(X, labels)
    assert np.allclose(np.linalg.norm(m.centroids, axis=1), 1.0)


def test_mixture_sums_to_one():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    shares, coverage = m.mixture(_queries(axes, 0, 20))
    assert shares.sum() == pytest.approx(1.0)
    assert coverage == 1.0


def test_assignment_is_correct_on_separable_topics():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    for i in range(3):
        assert m.validate(_queries(axes, i, 25), [m.names[i]] * 25) == 1.0


def test_knn_beats_centroid_on_overlapping_topics():
    """The measurement behind the kNN default."""
    rng = np.random.default_rng(3)
    d, per = 16, 60
    base = rng.normal(size=(3, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # topics 0 and 1 deliberately overlap; 2 sits apart
    base[1] = base[0] * 0.93 + base[1] * 0.07
    X, labels = [], []
    for i, nm in enumerate("abc"):
        pts = base[i] * 2.2 + rng.normal(size=(per, d)) * 0.5
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        X.append(pts); labels += [nm] * per
    X = np.vstack(X)
    held, truth = [], []
    for i, nm in enumerate("abc"):
        pts = base[i] * 2.2 + rng.normal(size=(30, d)) * 0.5
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        held.append(pts); truth += [nm] * 30
    held = np.vstack(held)

    knn = TopicModel.from_labels(X, labels, k=5)
    cen = TopicModel.from_labels(X, labels)
    cen.assign_mode = "centroid"
    assert knn.validate(held, truth) >= cen.validate(held, truth)


def test_centroid_mode_still_works_without_documents():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    m.doc_embeddings = None
    assert set(m.assign(_queries(axes, 1, 10)).tolist()) == {1}


# ----------------------------------------------------------------- the report
def _drifting_windows(axes, n=8):
    """Traffic migrates from topic 0 to topic 2."""
    out = []
    for w in range(n):
        t = w / (n - 1)
        a = _queries(axes, 0, max(int(40 * (1 - t)), 1), seed=100 + w)
        c = _queries(axes, 2, max(int(40 * t), 1), seed=200 + w)
        out.append(np.vstack([a, c]))
    return out


def test_analyse_names_the_riser_and_the_faller():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    rep = analyse(m, _drifting_windows(axes))
    assert rep.risers[0].name == "gamma"
    assert rep.fallers[0].name == "alpha"
    assert rep.risers[0].delta > 0 and rep.fallers[0].delta < 0


def test_mixtures_shape_matches_windows():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    w = _drifting_windows(axes, n=6)
    assert analyse(m, w).mixtures.shape == (6, 3)


def test_crossover_is_detected():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    rep = analyse(m, _drifting_windows(axes))
    assert any(up == "gamma" and down == "alpha" for _, up, down in rep.crossovers)


def test_coverage_flags_the_overloaded_topic():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    rep = analyse(m, _drifting_windows(axes))
    # the index is uniform across three topics; traffic ends concentrated on gamma
    assert "gamma" in coverage_gap(rep)
    assert "alpha" not in coverage_gap(rep)


def test_render_is_readable_text():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    text = analyse(m, _drifting_windows(axes)).render()
    for expected in ("RISING", "FALLING", "COVERAGE", "gamma"):
        assert expected in text


def test_single_window_is_rejected():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    with pytest.raises(ValueError):
        analyse(m, [_queries(axes, 0, 10)])


# --------------------------------------------------- confidence and regrouping
def test_assign_confident_returns_vote_share():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels, k=5)
    idx, conf = m.assign_confident(_queries(axes, 0, 20))
    assert set(idx.tolist()) == {0}
    assert ((conf > 0) & (conf <= 1.0)).all()


def test_mixture_reports_coverage_and_abstains():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels, k=5)
    shares, cov = m.mixture(_queries(axes, 0, 20), min_confidence=0.0)
    assert shares.sum() == pytest.approx(1.0) and cov == 1.0
    # an impossible bar abstains on everything
    shares, cov = m.mixture(_queries(axes, 0, 20), min_confidence=1.01)
    assert cov == 0.0 and shares.sum() == 0.0


def test_regroup_merges_topics_and_keeps_documents():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels)
    g = m.regroup({"alpha": "greek", "beta": "greek"})
    assert g.names == ["gamma", "greek"]
    assert int(g.doc_counts[g.names.index("greek")]) == 80
    assert g.doc_embeddings is not None


def test_regroup_leaves_unmapped_topics_alone():
    X, labels, axes = _world()
    g = TopicModel.from_labels(X, labels).regroup({"alpha": "greek"})
    assert "gamma" in g.names and "beta" in g.names


def test_regroup_requires_documents():
    X, labels, _ = _world()
    m = TopicModel.from_labels(X, labels)
    m.doc_embeddings = None
    with pytest.raises(ValueError):
        m.regroup({"alpha": "greek"})


def test_report_records_the_confidence_gate():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels, k=5)
    rep = analyse(m, _drifting_windows(axes), min_confidence=0.6)
    assert rep.min_confidence == 0.6
    assert len(rep.assigned_share) == len(rep.mixtures)
    assert "abstaining" in rep.render()


def test_render_has_one_rising_header():
    X, labels, axes = _world()
    m = TopicModel.from_labels(X, labels, k=5)
    for conf in (0.0, 0.6):
        text = analyse(m, _drifting_windows(axes), min_confidence=conf).render()
        assert text.count("RISING") == 1
