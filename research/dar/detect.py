"""
Detectors under test.

  D_original    : faithful re-implementation of the repo's D(t) and alpha/beta
                  regime classifier (level-based centroid term, raw sliced-W1,
                  ratio decomposition).  This is the thing we are trying to beat.

  D_calibrated  : the fix.  Each component is standardised against a per-window
                  permutation null (S1); centroid term is a *difference* (B3);
                  D is the sum of z-scores, so components are commensurable and
                  tau is a significance level, not a magic constant.

  regime_two_test : 2x2 from two INDEPENDENT calibrated tests (query side,
                    corpus side).  Fixes S2: the repo's alpha+beta=1 makes the
                    "quadrant" a line, and its "joint" = "two similar magnitudes"
                    = least informative case triggering the 110x action.

  baseline_*    : top-1 cosine, centroid shift, C2ST, PCA-KS, plain sliced-W1,
                  plain MMD.  Each returns a scalar "how drifted is window t vs
                  t-1", higher = more drift, for a common AUROC comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .stats import (
    Calibrated,
    centroid_gap,
    centroid_gap_delta,
    centroid_shift,
    mmd2_rbf,
    permutation_null,
    sliced_w1_max,
    sliced_w1_mean,
)

# repo defaults (configs/settings.py)
ALPHA, BETA, GAMMA = 1.0, 1.0, 0.5
TAU_ORIG = 0.15
TAU_ALPHA = TAU_BETA = 0.4


# --------------------------------------------------------------------------- #
# ORIGINAL D(t)  (repo reproduction)
# --------------------------------------------------------------------------- #
@dataclass
class OrigPoint:
    t: int
    w1_q: float
    w1_c: float
    cgap: float
    d: float
    alpha_hat: float
    beta_hat: float
    raw_regime: str


def _classify_alpha_beta(a: float, b: float, d: float) -> str:
    if d < TAU_ORIG:
        return "stable"
    if a >= TAU_ALPHA and b >= TAU_BETA:
        return "joint_drift"
    if a >= TAU_ALPHA:
        return "query_drift"
    if b >= TAU_BETA:
        return "corpus_drift"
    return "stable"


def d_original_series(windows, n_proj: int = 128, seed: int = 0,
                      ref: int | None = None, start: int = 1):
    """ref=None  -> adjacent-window comparison (the repo's lag=1 change-point view)
       ref=int   -> fixed-reference comparison (level view: 'drifted vs baseline?').
    A persistent shift stays flagged in every post-event window under fixed-ref;
    under adjacent-window it shows only a one-window spike at the change point."""
    rng = np.random.default_rng(seed)
    pts: list[OrigPoint] = []
    lo = start if ref is None else max(start, ref + 1)
    for t in range(lo, len(windows)):
        p = t - 1 if ref is None else ref
        Ct, Cp = windows[t].corpus, windows[p].corpus
        Qt, Qp = windows[t].queries, windows[p].queries
        w1q = sliced_w1_mean(Qt, Qp, n_proj, rng)
        w1c = sliced_w1_mean(Ct, Cp, n_proj, rng)
        cg = centroid_gap(Qt, Ct)                       # LEVEL (the bug)
        d = ALPHA * w1q + BETA * w1c + GAMMA * cg
        tot = w1q + w1c + 1e-10
        a, b = w1q / tot, w1c / tot
        pts.append(OrigPoint(t, w1q, w1c, cg, d, a, b, _classify_alpha_beta(a, b, d)))
    return pts


# --------------------------------------------------------------------------- #
# CALIBRATED D(t)  (S1 + B3)
# --------------------------------------------------------------------------- #
@dataclass
class CalPoint:
    t: int
    q: Calibrated                 # query-side sliced-W1, calibrated
    c: Calibrated                 # corpus-side sliced-W1, calibrated
    cgap_delta: float
    cgap_delta_z: float
    d_z: float                    # sum of the standardised components
    p_combined: float             # Fisher-combined p across the two sides
    q_sig: bool
    c_sig: bool
    regime: str


def _cgap_delta_null(Qt, Ct, Qp, Cp, B, rng, cap=2500):
    """Null for |gap_t - gap_{t-1}|: permute the (Q,C) label within each window."""
    def _s(A):
        return A if A.shape[0] <= cap else A[rng.choice(A.shape[0], cap, replace=False)]
    Qt, Ct, Qp, Cp = _s(Qt), _s(Ct), _s(Qp), _s(Cp)
    obs = centroid_gap_delta(Qt, Ct, Qp, Cp)
    poolt = np.vstack([Qt, Ct]); nqt = Qt.shape[0]
    poolp = np.vstack([Qp, Cp]); nqp = Qp.shape[0]
    vals = np.empty(B)
    for b in range(B):
        pt = rng.permutation(poolt.shape[0]); pp = rng.permutation(poolp.shape[0])
        gt = np.linalg.norm(poolt[pt[:nqt]].mean(0) - poolt[pt[nqt:]].mean(0))
        gp = np.linalg.norm(poolp[pp[:nqp]].mean(0) - poolp[pp[nqp:]].mean(0))
        vals[b] = abs(gt - gp)
    mu, sd = vals.mean(), vals.std() + 1e-12
    p = (1.0 + np.sum(vals >= obs)) / (B + 1.0)
    return obs, (obs - mu) / sd, float(p)


def d_calibrated_series(windows, B: int = 90, subsample: int = 1300,
                        alpha_level: float = 0.02, z_min: float = 2.5,
                        n_proj: int = 112, seed: int = 0,
                        do_mmd: bool = False, ref: int | None = None, start: int = 1):
    rng = np.random.default_rng(seed + 999)
    pts: list[CalPoint] = []
    lo = start if ref is None else max(start, ref + 1)
    for t in range(lo, len(windows)):
        p = t - 1 if ref is None else ref
        Ct, Cp = windows[t].corpus, windows[p].corpus
        Qt, Qp = windows[t].queries, windows[p].queries
        qcal = permutation_null(Qt, Qp, B, subsample, n_proj, seed + 10 * t, do_mmd)
        ccal = permutation_null(Ct, Cp, B, subsample, n_proj, seed + 10 * t + 5, do_mmd)
        # use max-slice z as the per-side test statistic (S4)
        q = qcal["w1_max"]; c = ccal["w1_max"]
        cgd, cgd_z, cgd_p = _cgap_delta_null(Qt, Ct, Qp, Cp, min(B, 80), rng)

        d_z = q.z + c.z + max(cgd_z, 0.0)
        # Fisher combine query & corpus p-values
        chi = -2.0 * (np.log(max(q.p, 1e-12)) + np.log(max(c.p, 1e-12)))
        p_comb = float(chi2.sf(chi, df=4))

        # significant = permutation p below alpha AND a minimum effect size
        # (guards against a small-B permutation p being low by luck)
        q_sig = (q.p <= alpha_level) and (q.z >= z_min)
        c_sig = (c.p <= alpha_level) and (c.z >= z_min)
        if q_sig and c_sig:
            reg = "joint_drift"
        elif q_sig:
            reg = "query_drift"
        elif c_sig:
            reg = "corpus_drift"
        else:
            reg = "stable"
        pts.append(CalPoint(t, q, c, cgd, cgd_z, d_z, p_comb, q_sig, c_sig, reg))
    return pts


# --------------------------------------------------------------------------- #
# BASELINES
# Each returns a score array aligned with d_*_series(ref=REF): one value per
# window t in [max(start, ref+1) .. T-1], higher = more drift, comparing W[t]
# against the fixed reference window W[ref].
# --------------------------------------------------------------------------- #
REF = 1          # fixed reference window (post burn-in)
START = 1


def _ts(windows, ref, start):
    lo = start if ref is None else max(start, ref + 1)
    return list(range(lo, len(windows)))


def _cap(A, n, rng):
    return A if A.shape[0] <= n else A[rng.choice(A.shape[0], n, replace=False)]


def _retr_quality(w, rng, subn=2000):
    Q = _cap(w.queries, subn, rng)
    C = _cap(w.corpus, min(subn * 2, w.corpus.shape[0]), rng)
    return (Q @ C.T).max(axis=1)


def baseline_top1_cosine(windows, seed=0, ref=REF, start=START) -> np.ndarray:
    """Reference mean best-match cosine minus this window's.  Positive => retrieval
    quality has fallen vs baseline.  The cheap baseline most likely to beat D(t)."""
    rng = np.random.default_rng(seed)
    m_ref = _retr_quality(windows[ref], rng).mean()
    return np.array([m_ref - _retr_quality(windows[t], rng).mean() for t in _ts(windows, ref, start)])


def baseline_below_floor(windows, floor=0.60, seed=0, ref=REF, start=START) -> np.ndarray:
    """Increase, vs baseline, in the fraction of questions whose best match is weak."""
    rng = np.random.default_rng(seed + 1)
    f_ref = float((_retr_quality(windows[ref], rng) < floor).mean())
    return np.array([float((_retr_quality(windows[t], rng) < floor).mean()) - f_ref
                     for t in _ts(windows, ref, start)])


def baseline_centroid_shift(windows, seed=0, ref=REF, start=START) -> np.ndarray:
    r = windows[ref]
    return np.array([centroid_shift(windows[t].queries, r.queries)
                     + centroid_shift(windows[t].corpus, r.corpus)
                     for t in _ts(windows, ref, start)])


def baseline_sliced_w1(windows, n_proj=128, seed=0, ref=REF, start=START) -> np.ndarray:
    rng = np.random.default_rng(seed + 2)
    r = windows[ref]
    return np.array([sliced_w1_mean(windows[t].queries, r.queries, n_proj, rng)
                     + sliced_w1_mean(windows[t].corpus, r.corpus, n_proj, rng)
                     for t in _ts(windows, ref, start)])


def baseline_mmd(windows, subsample=1200, seed=0, ref=REF, start=START) -> np.ndarray:
    rng = np.random.default_rng(seed + 3)
    r = windows[ref]
    def s(A, Bm):
        return mmd2_rbf(_cap(A, subsample, rng), _cap(Bm, subsample, rng), None, rng)
    return np.array([s(windows[t].queries, r.queries) + s(windows[t].corpus, r.corpus)
                     for t in _ts(windows, ref, start)])


def _c2st_auc(A, Bm, rng, cap=1500):
    A, Bm = _cap(A, cap, rng), _cap(Bm, cap, rng)
    X = np.vstack([A, Bm])
    y = np.r_[np.zeros(A.shape[0]), np.ones(Bm.shape[0])]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(rng.integers(1 << 30)))
    accs = [LogisticRegression(max_iter=200).fit(X[tr], y[tr]).score(X[te], y[te])
            for tr, te in skf.split(X, y)]
    return float(np.mean(accs))          # 0.5 = indistinguishable, ->1 = drifted


def baseline_c2st(windows, seed=0, ref=REF, start=START) -> np.ndarray:
    rng = np.random.default_rng(seed + 4)
    r = windows[ref]
    return np.array([_c2st_auc(windows[t].queries, r.queries, rng)
                     + _c2st_auc(windows[t].corpus, r.corpus, rng)
                     for t in _ts(windows, ref, start)])


def baseline_pca_ks(windows, n_comp=10, seed=0, ref=REF, start=START) -> np.ndarray:
    """KS on the top principal components fitted on the reference window's corpus."""
    rng = np.random.default_rng(seed + 5)
    r = windows[ref]
    base = _cap(r.corpus, 3000, rng)
    mu = base.mean(0)
    _, _, Vt = np.linalg.svd(base - mu, full_matrices=False)
    comps = Vt[:n_comp]
    out = []
    for t in _ts(windows, ref, start):
        score = 0.0
        for A, Bm in ((windows[t].corpus, r.corpus), (windows[t].queries, r.queries)):
            pa = (_cap(A, 3000, rng) - mu) @ comps.T
            pb = (_cap(Bm, 3000, rng) - mu) @ comps.T
            score += max(ks_2samp(pa[:, j], pb[:, j]).statistic for j in range(n_comp))
        out.append(score)
    return np.array(out)
