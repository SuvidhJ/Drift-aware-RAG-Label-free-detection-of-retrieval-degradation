"""Selective re-index: selectors, and the structure that makes them possible.

Also pins the position-vs-magnitude asymmetry (FINDINGS 12.3), which is the one
structural claim in this project that held in 6 of 6 conditions: interpolating the
displacement *magnitude* field works, interpolating the *vector* field is worse than
doing nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.select import (
    displacement,
    draw_probe,
    knn_graph,
    morans_i,
    morans_i_test,
    predict_displacement,
    predict_displacement_vector,
    select,
)


@pytest.fixture(scope="module")
def clustered():
    """A corpus with spatially concentrated churn: 6 clusters, 2 of them hot.

    Displacement is regionally structured, which is the precondition every
    probe-based selector relies on.
    """
    rng = np.random.default_rng(4)
    n_clusters, per = 6, 60
    centres = rng.standard_normal((n_clusters, 24)) * 3.0
    stale = np.vstack([centres[c] + rng.standard_normal((per, 24)) * 0.3 for c in range(n_clusters)])
    stale /= np.linalg.norm(stale, axis=1, keepdims=True)

    labels = np.repeat(np.arange(n_clusters), per)
    hot = np.isin(labels, [1, 4])
    step = rng.standard_normal(stale.shape)
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    fresh = stale + step * np.where(hot, 0.6, 0.01)[:, None]
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    return stale, fresh, hot


def test_displacement_matches_direct_computation(clustered):
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    rows = np.array([3, 17, 100])
    assert np.allclose(displacement(stale, fresh, rows), m[rows])


def test_probe_is_a_fraction_with_a_floor():
    assert len(draw_probe(10_000, 0.02)) == 200
    assert len(draw_probe(50, 0.02)) == 20  # floor, not 1
    assert len(draw_probe(10, 0.5)) == 10  # never more than the corpus


def test_displacement_is_spatially_autocorrelated(clustered):
    """Moran's I over the kNN graph of the stale index; the basis of probe-kNN."""
    stale, fresh, _ = clustered
    res = morans_i_test(displacement(stale, fresh), knn_graph(stale, 10), n_perm=199, seed=0)
    assert res["morans_i"] > 0.2
    assert res["z"] > 3.0
    assert res["p_value"] < 0.05


def test_morans_i_is_near_zero_for_an_unstructured_field(clustered):
    stale, _, _ = clustered
    rng = np.random.default_rng(0)
    res = morans_i_test(rng.standard_normal(len(stale)), knn_graph(stale, 10), n_perm=199, seed=1)
    assert abs(res["morans_i"]) < 0.2
    assert res["p_value"] > 0.01


def test_predicted_magnitude_correlates_with_the_truth(clustered):
    from scipy.stats import spearmanr

    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    probe = draw_probe(len(stale), 0.10, np.random.default_rng(1))
    predicted = predict_displacement(stale, probe, m[probe])
    rest = np.setdiff1d(np.arange(len(stale)), probe)
    assert spearmanr(predicted[rest], m[rest]).statistic > 0.3


def test_probed_rows_keep_their_measured_value(clustered):
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    probe = draw_probe(len(stale), 0.10, np.random.default_rng(1))
    assert np.allclose(predict_displacement(stale, probe, m[probe])[probe], m[probe])


def test_position_is_not_predictable_even_though_magnitude_is(clustered):
    """FINDINGS 12.3: moving documents to interpolated positions loses to doing nothing."""
    stale, fresh, _ = clustered
    probe = draw_probe(len(stale), 0.10, np.random.default_rng(2))
    delta = (fresh - stale).astype(np.float32)

    predicted = stale + predict_displacement_vector(stale, probe, delta[probe])
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-12)

    rest = np.setdiff1d(np.arange(len(stale)), probe)
    cos_predicted = np.einsum("ij,ij->i", predicted[rest], fresh[rest]).mean()
    cos_do_nothing = np.einsum("ij,ij->i", stale[rest], fresh[rest]).mean()
    assert cos_predicted <= cos_do_nothing + 1e-3


def test_selectors_respect_the_budget(clustered):
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    n = len(stale)
    for name in ("random", "probe_knn", "cluster_threshold", "oracle", "changelog"):
        sel = select(
            stale, 0.2, lambda r: m[r], selector=name,
            true_magnitude=m, edited=m > 1e-6, seed=0,
        )
        assert len(sel.rows) <= int(0.2 * n) + len(sel.probe_rows)
        assert len(np.unique(sel.rows)) == len(sel.rows), f"{name} returned duplicates"


def test_probe_knn_includes_its_probe(clustered):
    """Probes are already re-embedded, so they are spent budget and must be kept."""
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    sel = select(stale, 0.2, lambda r: m[r], selector="probe_knn", seed=0)
    assert set(sel.probe_rows.tolist()) <= set(sel.rows.tolist())


def test_probe_knn_captures_more_movement_than_random(clustered):
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    captured = {}
    for name in ("random", "probe_knn", "oracle"):
        totals = []
        for seed in range(5):
            sel = select(stale, 0.2, lambda r: m[r], selector=name, true_magnitude=m, seed=seed)
            totals.append(m[sel.rows].sum())
        captured[name] = float(np.mean(totals))
    assert captured["random"] < captured["probe_knn"] <= captured["oracle"] * 1.02


def test_unknown_selector_and_missing_inputs_are_rejected(clustered):
    stale, fresh, _ = clustered
    m = displacement(stale, fresh)
    with pytest.raises(ValueError, match="unknown selector"):
        select(stale, 0.2, lambda r: m[r], selector="nope")
    with pytest.raises(ValueError, match="oracle"):
        select(stale, 0.2, lambda r: m[r], selector="oracle")
    with pytest.raises(ValueError, match="changelog"):
        select(stale, 0.2, lambda r: m[r], selector="changelog")
    with pytest.raises(ValueError, match="budget"):
        select(stale, 0.0, lambda r: m[r], selector="random")
