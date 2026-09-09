"""
E5 (v3) - calibration probe.  Find a regime where index staleness ACTUALLY costs
retrieval quality, and characterise two distinct kinds of staleness:

  IN-PLACE REWRITE  the document is rewritten but stays in its semantic region.
                    Per-document movement is large; the CORPUS DISTRIBUTION does
                    not move at all.  Distributional drift detection is blind to
                    this by construction.
  MIGRATING REWRITE the document is rewritten and moves to a new region.
                    Both per-document movement and the distribution move.

This distinction matters: D(t) can only ever see the second kind.
"""
import numpy as np

D, N, NCL, NQ, TOPK = 256, 6000, 20, 2000, 10
SPREAD, QNOISE = 0.70, 0.45

def nrm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def build(rng):
    means = nrm(rng.standard_normal((NCL, D)))
    w = 1.0/np.arange(1, NCL+1); w /= w.sum()
    cnt = rng.multinomial(N, w)
    E, CL = [], []
    for k, c in enumerate(cnt):
        if c == 0: continue
        E.append(nrm(means[k] + (SPREAD/np.sqrt(D))*rng.standard_normal((c, D))))
        CL.append(np.full(c, k))
    return np.vstack(E), np.concatenate(CL), means

def stale_world(C0, cl, means, rng, hot_k=5, rewrite_frac=0.60, migrate_frac=0.0):
    sizes = np.bincount(cl, minlength=NCL); order = np.argsort(sizes)
    hot = rng.choice(order[NCL//4:], size=hot_k, replace=False)
    in_hot = np.isin(cl, hot)
    rewritten = in_hot & (rng.random(N) < rewrite_frac)
    C1 = C0.copy()
    idx = np.where(rewritten)[0]
    migrate = rng.random(len(idx)) < migrate_frac
    for j, i in enumerate(idx):
        base = nrm(rng.standard_normal(D)) if migrate[j] else means[cl[i]]
        C1[i] = nrm(base + (SPREAD/np.sqrt(D))*rng.standard_normal(D))
    return C1, rewritten, in_hot

def recall(Q, E, gold, k=TOPK, block=500):
    h = 0
    for s in range(0, Q.shape[0], block):
        sims = Q[s:s+block] @ E.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        h += (top == gold[s:s+block, None]).any(axis=1).sum()
    return h/Q.shape[0]

def _w1(px, py, M=200):
    g = (np.arange(M)+0.5)/M
    return np.abs(np.quantile(px,g,axis=0)-np.quantile(py,g,axis=0)).mean(axis=0)

rng = np.random.default_rng(0)
C0, cl, means = build(rng)
print(f"{'scenario':<34}{'stale':>8}{'full':>8}{'gap':>8}   {'maxW1(C1,C0)':>13}")
for name, mig in [("in-place rewrite  (migrate 0%)", 0.0),
                  ("mixed             (migrate 50%)", 0.5),
                  ("migrating rewrite (migrate 100%)", 1.0)]:
    r = np.random.default_rng(7)
    C1, rew, hot = stale_world(C0, cl, means, r, migrate_frac=mig)
    pick = r.integers(0, N, size=NQ)
    Q = nrm(C1[pick] + (QNOISE/np.sqrt(D))*r.standard_normal((NQ, D)))
    rs, rf = recall(Q, C0, pick), recall(Q, C1, pick)
    th = nrm(r.standard_normal((192, D)))
    w = _w1(C1 @ th.T, C0 @ th.T).max()
    print(f"{name:<34}{rs:>8.4f}{rf:>8.4f}{rf-rs:>8.4f}   {w:>13.5f}")
print(f"\nrewritten docs: {rew.mean():.1%} of corpus   (hot regions hold {hot.mean():.1%})")
