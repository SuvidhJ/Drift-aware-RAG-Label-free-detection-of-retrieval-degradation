"""Detection power swept down through the false-alarm floor.

The second surviving negative result (FINDINGS.md 12.9), as a re-runnable command.

At a correctly calibrated 5% false-alarm rate, power at the effect sizes real corpora
actually exhibit sits close to the false-alarm floor: measured here at n=1500, arXiv
lands around 0.13-0.15 and Python docs around 0.20, against a 5% floor.  That is
stronger than "power degrades near a floor": the realistic operating regime *is* the
floor.

**Mind the displacement convention.**  Perturbing a fraction f of documents by a step
of size s gives a mean displacement of s over the moved documents and f*s over all of
them.  FINDINGS 12.9 swept the per-moved quantity -- its "0.765" is the displacement
of the documents that moved -- but compared corpora against it using their per-ALL
means (arXiv 0.080, Python docs 0.164, Wikipedia 0.263), which are smaller by exactly
the revised fraction.  The two must not be mixed.  Corrected to a common convention
the conclusion is unchanged, because both readings land in the same low-power region,
but the sweep now reports both columns so the comparison cannot be made wrongly.

The obvious objection is that power grows with window size, so one should just use
more data.  That is not available here: enlarging the window changes the null
distribution and invalidates the threshold, which is exactly the failure documented
in evaluation/transfer.py.  The two results compound rather than cancel.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from configs.settings import settings
from drift.mmd import centroid_shift, mmd_rbf_fast, median_heuristic_sigma
from drift.wasserstein import sliced_w1_fast


def perturb(
    X: np.ndarray,
    scale: float,
    revised_fraction: float,
    rng: np.random.Generator,
    return_rows: bool = False,
):
    """Move a fraction of the rows by an isotropic step of size ``scale``.

    Renormalised afterwards, because the index holds unit vectors and a perturbation
    that changed vector norms would be detectable for the wrong reason.

    ``return_rows`` gives back the rows that were actually moved. Callers need that
    rather than thresholding the displacement: renormalisation nudges *every* row by
    the float error in its original norm (~1e-7 for vectors stored as float32 and
    promoted to float64), which is larger than any sensible "did it move" epsilon, so
    a threshold marks the whole corpus as moved and the per-moved mean silently
    collapses onto the per-all mean.
    """
    Y = X.copy()
    n = len(Y)
    k = int(revised_fraction * n)
    rows = np.empty(0, dtype=np.int64)
    if k > 0 and scale > 0:
        rows = rng.choice(n, k, replace=False)
        step = rng.standard_normal((k, Y.shape[1]))
        step /= np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-12)
        Y[rows] = Y[rows] + scale * step
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return (Y, rows) if return_rows else Y


def run_power_sweep(
    pool: np.ndarray,
    scales: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.60, 1.00),
    window: int = 800,
    revised_fraction: float = 0.30,
    trials: int = 60,
    alpha: float | None = None,
    seed: int = 0,
) -> dict:
    """Power at a threshold calibrated to ``alpha`` on the null, per perturbation scale."""
    alpha = settings.drift.alpha_level if alpha is None else alpha
    pool = np.asarray(pool, dtype=np.float64)

    if len(pool) < 2 * window:
        raise ValueError(
            f"need {2 * window} vectors for two disjoint windows of {window}; pool has {len(pool)}"
        )

    rng = np.random.default_rng(seed)
    sigma = median_heuristic_sigma(pool[:500], pool[500:1000] if len(pool) > 1000 else pool[:500])
    stats = {
        "sliced_w1": lambda a, b: sliced_w1_fast(a, b, n_projections=64, seed=0),
        "mmd": lambda a, b: mmd_rbf_fast(a, b, sigma),
        "centroid": centroid_shift,
    }

    # thresholds from the null: both windows unperturbed
    null_scores = {k: np.empty(trials) for k in stats}
    for i in range(trials):
        idx = rng.choice(len(pool), 2 * window, replace=False)
        A, B = pool[idx[:window]], pool[idx[window:]]
        for k, fn in stats.items():
            null_scores[k][i] = fn(A, B)
    thresholds = {k: float(np.quantile(v, 1 - alpha)) for k, v in null_scores.items()}
    logger.info(f"thresholds at alpha={alpha}, n={window}: " + ", ".join(f"{k}={v:.5f}" for k, v in thresholds.items()))

    rows: list[dict] = []
    for scale in scales:
        hits = {k: 0 for k in stats}
        induced_all, induced_moved = [], []
        for _ in range(trials):
            idx = rng.choice(len(pool), 2 * window, replace=False)
            A = pool[idx[:window]]
            base = pool[idx[window:]]
            B, moved_rows = perturb(base, scale, revised_fraction, rng, return_rows=True)
            step = np.linalg.norm(B - base, axis=1)
            induced_all.append(float(step.mean()))
            induced_moved.append(float(step[moved_rows].mean()) if len(moved_rows) else 0.0)
            for k, fn in stats.items():
                if fn(A, B) > thresholds[k]:
                    hits[k] += 1
        row = {
            "scale": scale,
            # Both conventions, deliberately. They differ by the revised fraction, and
            # mixing them is a live hazard: FINDINGS 12.9 swept the per-moved quantity
            # but compared real corpora against it using their per-all means, which are
            # smaller by exactly that factor. Reading a corpus off this sweep requires
            # the matching column.
            "mean_displacement_all_docs": float(np.mean(induced_all)),
            "mean_displacement_moved_docs": float(np.mean(induced_moved)),
            **{f"power_{k}": hits[k] / trials for k in stats},
        }
        rows.append(row)
        logger.info(
            f"scale {scale:.2f} (displacement: all docs {row['mean_displacement_all_docs']:.3f}, "
            f"moved only {row['mean_displacement_moved_docs']:.3f}): "
            + ", ".join(f"{k}={row[f'power_{k}']:.3f}" for k in stats)
        )

    return {
        "window": window,
        "revised_fraction": revised_fraction,
        "trials": trials,
        "alpha": alpha,
        "thresholds": thresholds,
        "sweep": rows,
        "note": (
            "Power is measured against a threshold calibrated on the null at this same "
            "window size, so it is not confounded by the transfer failure. To read a "
            "corpus off this sweep, match the convention: corpus_operating_point's "
            "mean_displacement is over ALL documents, so compare it against "
            "mean_displacement_all_docs, not against mean_displacement_moved_docs. "
            "The two differ by the revised fraction."
        ),
    }


def corpus_operating_point(stale: np.ndarray, fresh: np.ndarray) -> dict:
    """Where a real corpus sits on the sweep: its actual mean displacement."""
    m = np.linalg.norm(np.asarray(fresh) - np.asarray(stale), axis=1)
    # A real corpus has genuinely unchanged documents, whose displacement is exactly
    # zero, so a threshold is safe here (unlike in the synthetic sweep, where global
    # renormalisation perturbs every row by float error).
    moved = m > 1e-6
    return {
        # averaged over ALL documents: compare against mean_displacement_all_docs
        "mean_displacement": float(m.mean()),
        # averaged over documents that actually moved: compare against
        # mean_displacement_moved_docs
        "mean_displacement_moved": float(m[moved].mean()) if moved.any() else 0.0,
        "median_displacement": float(np.median(m)),
        "p90_displacement": float(np.quantile(m, 0.9)),
        "fraction_moved": float((m > 1e-6).mean()),
    }
