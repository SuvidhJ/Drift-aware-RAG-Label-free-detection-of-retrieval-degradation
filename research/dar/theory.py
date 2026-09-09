"""
Making the bound real  -- the fix for the "Lipschitz" section.

Repo claims   |E[R(Q_t,C_t)] - E[R(Q_0,C_0)]|  <=  L_hat * D(t)
with L_hat an empirically estimated Lipschitz constant of the *encoder*.  That
constant is (a) the wrong object -- Kantorovich-Rubinstein needs the Lipschitz
constant of the retrieval *functional*, not the encoder -- and (b) estimated as a
max of 500 ratios with a bootstrap that is provably inconsistent for extreme
order statistics.

Fix: choose a functional whose constant is 1 by construction.

    rho_C(q) = max_{c in C} <q, c>            (best available match for q)
    R(Q; C)  = E_{q ~ Q}[ rho_C(q) ]

On the unit sphere rho_C is a pointwise max of linear maps each of operator norm
<= 1, hence 1-Lipschitz in q.  Kantorovich-Rubinstein then gives, exactly:

    | R(Q_t; C) - R(Q_0; C) |  <=  W1(Q_t, Q_0)

No constant to estimate.  This module *empirically checks* that inequality with
    - exact W1 (ot.emd2) on a subsample          -> should hold always
    - mean-sliced W1 (a lower bound on W1)        -> should be VIOLATED sometimes
which is exactly the caveat the report flags: substituting sliced W1 into the RHS
does not preserve the bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot

from .stats import sliced_w1_mean, sliced_w1_max


def rho_C(Q: np.ndarray, C: np.ndarray, block: int = 4096) -> np.ndarray:
    """Per-query best match  max_c <q,c>.  Blocked to bound memory."""
    out = np.empty(Q.shape[0])
    for i in range(0, Q.shape[0], block):
        out[i:i + block] = (Q[i:i + block] @ C.T).max(axis=1)
    return out


def R(Q: np.ndarray, C: np.ndarray) -> float:
    return float(rho_C(Q, C).mean())


def exact_w1(X: np.ndarray, Y: np.ndarray, cap: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if X.shape[0] > cap:
        X = X[rng.choice(X.shape[0], cap, replace=False)]
    if Y.shape[0] > cap:
        Y = Y[rng.choice(Y.shape[0], cap, replace=False)]
    M = ot.dist(np.asarray(X, np.float64), np.asarray(Y, np.float64), metric="euclidean")
    a = np.full(X.shape[0], 1.0 / X.shape[0])
    b = np.full(Y.shape[0], 1.0 / Y.shape[0])
    return float(ot.emd2(a, b, M))


@dataclass
class BoundRow:
    t: int
    lhs: float               # |R(Q_t;C) - R(Q_0;C)|   (C fixed at reference)
    w1_exact: float
    w1_sliced_mean: float
    w1_sliced_max: float
    holds_exact: bool
    holds_sliced_mean: bool


def check_kr_bound(windows, ref: int = 0, cap: int = 1500, seed: int = 0) -> list[BoundRow]:
    """C is held at the reference window's corpus (query-side bound only -- the
    corpus side changes the functional itself and needs a separate argument)."""
    C = windows[ref].corpus
    Q0 = windows[ref].queries
    R0 = R(Q0, C)
    rows: list[BoundRow] = []
    rng = np.random.default_rng(seed)
    for t in range(len(windows)):
        Qt = windows[t].queries
        lhs = abs(R(Qt, C) - R0)
        we = exact_w1(Qt, Q0, cap, seed + t)
        wm = sliced_w1_mean(Qt, Q0, 256, np.random.default_rng(seed + 100 + t))
        wx = sliced_w1_max(Qt, Q0, 256, np.random.default_rng(seed + 200 + t))
        rows.append(BoundRow(
            t=t, lhs=lhs, w1_exact=we, w1_sliced_mean=wm, w1_sliced_max=wx,
            holds_exact=lhs <= we + 1e-9,
            holds_sliced_mean=lhs <= wm + 1e-9,
        ))
    return rows
