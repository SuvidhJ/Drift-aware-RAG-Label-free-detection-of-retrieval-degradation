"""Which documents to re-embed when the budget is smaller than the corpus.

This is the machinery behind action ``a3_reindex``.  A full re-index costs one
forward pass per document; the question is what to do when you can afford only a
fraction B of that.

All selectors share one production-shaped contract: you may re-embed a small *probe*
sample for free-ish (2% by default), observe how far those documents actually moved,
and must then choose the remaining budget without observing anything else.

    probe_knn          interpolate probe displacement MAGNITUDE over the kNN graph of
                       the stale index, then take the highest predicted movers
    cluster_threshold  k-means the stale index, score each cluster by the mean probe
                       displacement inside it, refresh whole clusters by rank
    random             uniform sample -- the floor
    oracle             true displacement, not implementable, an upper bound
    changelog          documents a change log says were edited, in random order

Read this before choosing probe_knn
-----------------------------------
probe_knn beats random by 1.26-1.58x in share of recoverable recall at n = 7346
(all p < 0.001), but given the SAME probe it ties or loses to cluster_threshold at
every budget tested -- 10%: -0.0010 (p=0.09), 20%: -0.0008 (p=0.18), 35%: -0.0003
(p=0.57), 30 seeds (FINDINGS.md 12.5).  Cluster averaging extracts as much from a
probe as kNN interpolation does.  Neither comes close to a change log
(0.9003/0.9158/0.9250), so if you have one, use it.

Two structural results do hold and are what make any of this work at all:

  * displacement is spatially autocorrelated -- Moran's I = +0.1184 (Python docs,
    z=+15.4), +0.0557 (arXiv, z=+15.0), +0.0311 (Wikipedia, z=+1.8), so a probe
    carries information about its neighbourhood;
  * magnitude is predictable but POSITION is not -- interpolating the displacement
    *vector* field and moving documents there is worse than leaving them alone, in
    6 of 6 conditions (FINDINGS.md 12.3).  ``predict_displacement_vector`` exists
    only so that test can be reproduced; do not use it to update an index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from configs.settings import settings


@dataclass
class Selection:
    rows: np.ndarray
    probe_rows: np.ndarray
    selector: str
    budget_rows: int
    predicted: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.rows)


def displacement(stale: np.ndarray, fresh: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """L2 distance each document moved between two index states."""
    if rows is None:
        return np.linalg.norm(fresh - stale, axis=1)
    rows = np.asarray(rows, dtype=np.int64)
    return np.linalg.norm(fresh[rows] - stale[rows], axis=1)


def draw_probe(n: int, probe_frac: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    frac = settings.retrieval.probe_frac if probe_frac is None else probe_frac
    rng = rng or np.random.default_rng(0)
    size = int(np.clip(int(frac * n), 20, n))
    return rng.choice(n, size, replace=False)


def predict_displacement(
    stale: np.ndarray,
    probe_rows: np.ndarray,
    probe_magnitude: np.ndarray,
    k: int | None = None,
    power: int | None = None,
) -> np.ndarray:
    """Interpolate displacement magnitude from probes over the stale kNN graph.

    Weights are cosine^power over the k nearest probes, which sharpens the kernel so
    distant probes stop contributing. Probed rows keep their measured value.
    """
    cfg = settings.retrieval
    k = cfg.knn_k if k is None else k
    power = cfg.knn_power if power is None else power

    n = len(stale)
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    P = stale[probe_rows]
    mp = np.asarray(probe_magnitude, dtype=np.float64)
    if len(mp) != len(probe_rows):
        raise ValueError(f"{len(probe_rows)} probe rows but {len(mp)} magnitudes")

    out = np.zeros(n, dtype=np.float64)
    kk = min(k, len(probe_rows))
    for start in range(0, n, 512):
        sims = stale[start : start + 512] @ P.T
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        w = np.maximum(np.take_along_axis(sims, top, axis=1), 0.0) ** power
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        out[start : start + sims.shape[0]] = (w * mp[top]).sum(axis=1)

    out[probe_rows] = mp
    return out


def predict_displacement_vector(
    stale: np.ndarray,
    probe_rows: np.ndarray,
    probe_delta: np.ndarray,
    k: int | None = None,
    power: int | None = None,
) -> np.ndarray:
    """Interpolate the displacement VECTOR field.

    Reproduces the position-vs-magnitude test only. Applying this to an index makes
    retrieval worse than doing nothing, in every corpus at every probe size tested.
    """
    cfg = settings.retrieval
    k = cfg.knn_k if k is None else k
    power = cfg.knn_power if power is None else power

    n, d = stale.shape
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    P = stale[probe_rows]
    Dp = np.asarray(probe_delta, dtype=np.float32)

    out = np.zeros((n, d), dtype=np.float32)
    kk = min(k, len(probe_rows))
    for start in range(0, n, 512):
        sims = stale[start : start + 512] @ P.T
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        w = np.maximum(np.take_along_axis(sims, top, axis=1), 0.0) ** power
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        out[start : start + sims.shape[0]] = np.einsum("bk,bkd->bd", w, Dp[top])
    return out


# ------------------------------------------------------------------ selectors


def select_random(n: int, budget_rows: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, min(budget_rows, n), replace=False)


def select_oracle(true_magnitude: np.ndarray, budget_rows: int) -> np.ndarray:
    return np.argsort(-np.asarray(true_magnitude))[:budget_rows]


def select_changelog(edited: np.ndarray, budget_rows: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.where(np.asarray(edited))[0]
    rng.shuffle(rows)
    return rows[:budget_rows]


def select_probe_knn(
    stale: np.ndarray,
    probe_rows: np.ndarray,
    probe_magnitude: np.ndarray,
    budget_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Probes are already refreshed, so they are spent budget and always included."""
    n = len(stale)
    predicted = predict_displacement(stale, probe_rows, probe_magnitude)
    rest = np.setdiff1d(np.arange(n), probe_rows)
    remaining = max(budget_rows - len(probe_rows), 0)
    picked = rest[np.argsort(-predicted[rest])[:remaining]]
    return np.concatenate([probe_rows, picked]), predicted


def select_cluster_threshold(
    stale: np.ndarray,
    probe_rows: np.ndarray,
    probe_magnitude: np.ndarray,
    budget_rows: int,
    n_clusters: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Closest deployed prior art: refresh whole clusters, hottest first."""
    from sklearn.cluster import KMeans

    n = len(stale)
    n_clusters = settings.retrieval.n_clusters if n_clusters is None else n_clusters
    n_clusters = int(min(n_clusters, max(n // 20, 2)))

    labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(stale).labels_
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    mp = np.asarray(probe_magnitude, dtype=np.float64)

    score: dict[int, float] = {}
    for c in np.unique(labels):
        inside = mp[labels[probe_rows] == c]
        score[int(c)] = float(inside.mean()) if len(inside) else 0.0

    chosen: list[int] = []
    for c in sorted(score, key=lambda c: -score[c]):
        chosen.extend(np.where(labels == c)[0].tolist())
        if len(chosen) >= budget_rows:
            break
    return np.array(chosen[:budget_rows], dtype=np.int64)


def select(
    stale: np.ndarray,
    budget: float,
    probe_magnitude_fn,
    selector: str | None = None,
    true_magnitude: np.ndarray | None = None,
    edited: np.ndarray | None = None,
    seed: int = 0,
) -> Selection:
    """Choose rows to re-embed under a fractional ``budget``.

    Args:
        stale: (n, d) current index vectors
        budget: fraction of the corpus that may be re-embedded, in (0, 1]
        probe_magnitude_fn: rows -> measured displacement for those rows. In
            production this re-embeds them and compares; in an experiment it reads
            the known fresh matrix.
        selector: name, defaults to ``RETRIEVAL_SELECTOR``
        true_magnitude: required by 'oracle' only
        edited: required by 'changelog' only
    """
    selector = selector or settings.retrieval.selector
    n = len(stale)
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    budget_rows = max(int(budget * n), 1)
    rng = np.random.default_rng(seed)

    probe_rows = draw_probe(n, rng=rng)
    predicted = None

    if selector == "random":
        rows = select_random(n, budget_rows, rng)
    elif selector == "oracle":
        if true_magnitude is None:
            raise ValueError("selector 'oracle' needs true_magnitude (it is an upper bound, not deployable)")
        rows = select_oracle(true_magnitude, budget_rows)
    elif selector == "changelog":
        if edited is None:
            raise ValueError("selector 'changelog' needs the edited mask")
        rows = select_changelog(edited, budget_rows, rng)
    elif selector == "probe_knn":
        rows, predicted = select_probe_knn(
            stale, probe_rows, np.asarray(probe_magnitude_fn(probe_rows)), budget_rows
        )
    elif selector == "cluster_threshold":
        rows = select_cluster_threshold(
            stale, probe_rows, np.asarray(probe_magnitude_fn(probe_rows)), budget_rows, seed=seed
        )
    else:
        raise ValueError(
            f"unknown selector {selector!r}; expected one of "
            "probe_knn, cluster_threshold, random, oracle, changelog"
        )

    logger.debug(f"selector {selector}: {len(rows)} of {n} rows at budget {budget:.0%}")
    return Selection(
        rows=np.asarray(rows, dtype=np.int64),
        probe_rows=probe_rows,
        selector=selector,
        budget_rows=budget_rows,
        predicted=predicted,
    )


def knn_graph(embs: np.ndarray, k: int = 15) -> np.ndarray:
    """Indices of the k nearest neighbours by cosine, excluding self."""
    n = len(embs)
    k = min(k, n - 1)
    out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, 512):
        sims = embs[start : start + 512] @ embs.T
        for r in range(sims.shape[0]):
            sims[r, start + r] = -np.inf
        out[start : start + sims.shape[0]] = np.argpartition(-sims, k, axis=1)[:, :k]
    return out


def morans_i(values: np.ndarray, neighbours: np.ndarray) -> float:
    """Moran's I over a row-standardised kNN graph (Diaz, SIGIR'07 formulation).

    Row standardisation makes the weight matrix sum to n, so the n/sum(W) prefactor
    is 1 and this reduces to the ratio of the neighbour cross-product to the variance.
    """
    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    return float((z @ z[neighbours].mean(axis=1)) / (z @ z))


def morans_i_test(
    values: np.ndarray, neighbours: np.ndarray, n_perm: int = 999, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    observed = morans_i(values, neighbours)
    null = np.array([morans_i(rng.permutation(values), neighbours) for _ in range(n_perm)])
    sd = null.std(ddof=1)
    return {
        "morans_i": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(sd),
        "z": float((observed - null.mean()) / sd) if sd > 1e-15 else 0.0,
        "p_value": float((1 + int((null >= observed).sum())) / (n_perm + 1)),
        "n_perm": n_perm,
    }
