"""What does a fully stale index cost, compared with a fully refreshed one?

Stale index = every document at its old revision.  Fresh = every document at its new
one.  Both are scored on the same queries, so the only thing that differs is the
index.

The estimate is STRATIFIED over the three query pools (see queries.py):

    net  =  w_new * gap_new  +  w_retained * gap_retained  +  w_unchanged * gap_unchanged

with each pool's weight taken from the corpus itself:

    w_new       = revision rate x mean fraction of a revised document that is new text
    w_retained  = revision rate x (1 - that fraction)
    w_unchanged = 1 - revision rate

This replaces an earlier estimator that drew a 4,000-query mixture with replacement,
which added sampling noise of its own.  It also makes the mechanism visible: gap_new is
what refreshing gains, gap_retained is what it can cost.

Ranks are exact (1 + number of documents scoring strictly above the gold), so hit@k
and MRR come from the same computation.  MRR matters because hit@10 saturates near 1.0
at fine chunk granularity and then cannot register anything.
"""
from __future__ import annotations

import numpy as np

from drift.freshness import canonical, shingles

from . import data, queries

WHATS = {"new": "q_new", "retained": "q_retained", "unchanged": "q_unchanged"}


def pool_weights(docs: list[dict]) -> dict[str, float]:
    revised = [d for d in docs if canonical(d["text0"]) != canonical(d["text1"])]
    rate = len(revised) / len(docs)
    fracs = []
    for d in revised:
        a, b = shingles(canonical(d["text0"]), 5), shingles(canonical(d["text1"]), 5)
        fracs.append(len(b - a) / max(len(b), 1))
    frac_new = float(np.mean(fracs)) if fracs else 0.0
    return {
        "new": rate * frac_new,
        "retained": rate * (1 - frac_new),
        "unchanged": 1 - rate,
    }


def gold_ranks(Q: np.ndarray, D: np.ndarray, gold: np.ndarray, block: int = 256) -> np.ndarray:
    """Exact rank of each query's gold document (1 = top)."""
    out = np.empty(len(Q), dtype=np.int64)
    for s in range(0, len(Q), block):
        S = Q[s : s + block] @ D.T
        g = S[np.arange(S.shape[0]), gold[s : s + block]]
        out[s : s + block] = (S > g[:, None]).sum(1) + 1
    return out


def metric(ranks: np.ndarray, name: str) -> np.ndarray:
    if name == "mrr":
        return 1.0 / ranks
    k = int(name.split("@")[1])
    return (ranks <= k).astype(float)


def per_pool(model: str, corpus: str) -> dict:
    """Stale and fresh gold ranks for every pool of one corpus."""
    docs = data.load_corpus(corpus)
    D0 = data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs])
    D1 = data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs])
    out = {}
    for pool, fn in queries.POOLS.items():
        texts, gold = fn(docs)
        if len(gold) == 0:
            continue
        Q = data.cached(model, corpus, WHATS[pool], lambda t=texts: t)
        out[pool] = {"stale": gold_ranks(Q, D0, gold), "fresh": gold_ranks(Q, D1, gold)}
    return {"weights": pool_weights(docs), "pools": out, "n_docs": len(docs)}


def net_cost(res: dict, name: str, B: int = 4000, seed: int = 0) -> dict:
    """Stratified net gain of refreshing, with a within-pool bootstrap CI."""
    rng = np.random.default_rng(seed)
    w = res["weights"]
    deltas = {
        p: metric(r["fresh"], name) - metric(r["stale"], name)
        for p, r in res["pools"].items()
    }
    missing = [p for p in w if p not in deltas and w[p] > 0]
    point = sum(w[p] * deltas[p].mean() for p in deltas)
    boots = np.zeros(B)
    for b in range(B):
        boots[b] = sum(
            w[p] * d[rng.integers(0, len(d), len(d))].mean() for p, d in deltas.items()
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pools = {}
    for p, d in deltas.items():
        pb = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        pools[p] = (float(d.mean()), *map(float, np.percentile(pb, [2.5, 97.5])), len(d))
    return {"net": float(point), "ci": (float(lo), float(hi)), "pools": pools,
            "missing": missing}
