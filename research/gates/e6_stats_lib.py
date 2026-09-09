"""E6 - Open question #1 from FINDINGS.md sec.10, tested on real data.

Is document revision regionally concentrated in embedding space?
Real Wikipedia articles, real revisions 2023-01 -> 2026-09, real MiniLM embeddings.
Nothing here is simulated: every vector is a forward pass on real revision text.

  E  churn heterogeneity by topic hub
  A  Moran's I of the displacement field over the kNN graph of the STALE embeddings
     (formulation follows Diaz, SIGIR'07, who applied the Moran coefficient to
      retrieval scores; here the field is temporal displacement, not score)
  B  variogram - does |m_i - m_j| grow with cosine distance
  C  probe-kNN predictive power + spatial-shuffle null
  F  the crux: position is NOT predictable, magnitude IS
     (Drift-Adapter arXiv:2509.23471 learns a global old->new map for a MODEL swap;
      under CONTENT revision no such map can exist, so F must separate the two)
"""
import json, sys, numpy as np
from scipy.stats import spearmanr
from minilm_np import MiniLM

RNG = np.random.default_rng(20260906)
K_NN, PROBE_FRAC, POW = 15, 0.02, 8


def knn_graph(C, k):
    """Indices of k nearest neighbours by cosine, excluding self."""
    n = len(C)
    idx = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, 512):
        S = C[s:s + 512] @ C.T
        for r in range(S.shape[0]):
            S[r, s + r] = -np.inf
        idx[s:s + 512] = np.argpartition(-S, k, axis=1)[:, :k]
    return idx


def morans_I(m, nb):
    """Row-standardised kNN weights => W sums to n, so n/W = 1."""
    z = m - m.mean()
    return float((z @ z[nb].mean(axis=1)) / (z @ z))


def moran_test(m, nb, nperm=999):
    I = morans_I(m, nb)
    null = np.array([morans_I(RNG.permutation(m), nb) for _ in range(nperm)])
    z = (I - null.mean()) / null.std(ddof=1)
    p = (1 + int((null >= I).sum())) / (nperm + 1)
    return I, float(null.mean()), float(null.std(ddof=1)), float(z), p


def variogram(C, m, ranks=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512), sample=1500):
    """E|m_i - m_j| where j is the r-th nearest neighbour of i.

    Binning by global quantiles of RANDOM pairs is uninformative in high dimensions:
    nearly every random pair is far apart, so the near regime - where the structure
    lives - is never sampled.  Binning by neighbour RANK fixes that.
    """
    n = len(C)
    rmax = min(max(ranks), n - 1)
    idx = RNG.choice(n, min(sample, n), replace=False)
    S = C[idx] @ C.T
    S[np.arange(len(idx)), idx] = -np.inf
    order = np.argpartition(-S, rmax, axis=1)[:, :rmax + 1]
    ordS = np.take_along_axis(S, order, 1)
    order = np.take_along_axis(order, np.argsort(-ordS, axis=1), 1)
    rows = []
    for r in ranks:
        if r > rmax:
            continue
        j = order[:, r - 1]
        cosd = 1.0 - np.einsum("ij,ij->i", C[idx], C[j])
        rows.append((r, float(cosd.mean()), float(np.abs(m[idx] - m[j]).mean())))
    return rows


def probe_knn_predict(C0, m, probe_idx, k=K_NN, power=POW):
    """Predict displacement magnitude for every doc from probe displacements only."""
    n = len(C0)
    P = C0[probe_idx]
    mp = np.asarray(m)[probe_idx]
    mhat = np.zeros(n)
    for s in range(0, n, 512):
        S = C0[s:s + 512] @ P.T
        kk = min(k, S.shape[1])
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.maximum(np.take_along_axis(S, top, 1), 0.0) ** power
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        mhat[s:s + 512] = (w * mp[top]).sum(1)
    mhat[probe_idx] = mp
    return mhat


def probe_knn_predict_vector(C0, D, probe_idx, k=K_NN, power=POW):
    """Same interpolation, but over the displacement VECTOR field (for test F)."""
    n, d = C0.shape
    P = C0[probe_idx]
    Dp = D[probe_idx]
    out = np.zeros((n, d), dtype=np.float32)
    for s in range(0, n, 512):
        S = C0[s:s + 512] @ P.T
        kk = min(k, S.shape[1])
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.maximum(np.take_along_axis(S, top, 1), 0.0) ** power
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        out[s:s + 512] = np.einsum("bk,bkd->bd", w, Dp[top])
    return out


def main(path="wp_corpus.json", tag=""):
    docs = json.load(open(path, encoding="utf-8"))
    print("SOURCE: %s" % path)
    print("corpus n=%d" % len(docs), flush=True)
    model = MiniLM()

    print("embedding stale (2023) ...", flush=True)
    C0 = model.embed([d["text0"] for d in docs], batch=32, verbose=True)
    print("embedding fresh (2026) ...", flush=True)
    C1 = model.embed([d["text1"] for d in docs], batch=32, verbose=True)
    np.save("C0%s.npy" % tag, C0)
    np.save("C1%s.npy" % tag, C1)

    m = np.linalg.norm(C1 - C0, axis=1)
    hubs = np.array([d["hub"] for d in docs])
    print("\n=== displacement summary ===")
    print("  mean %.4f  sd %.4f  min %.4f  p50 %.4f  p90 %.4f  max %.4f"
          % (m.mean(), m.std(), m.min(), np.median(m), np.quantile(m, .9), m.max()))
    print("  text unchanged (m==0): %d / %d" % (int((m < 1e-6).sum()), len(m)))

    print("\n=== E: churn by topic hub ===")
    rows = sorted(((h, float(m[hubs == h].mean()), int((hubs == h).sum()))
                   for h in sorted(set(hubs.tolist()))), key=lambda r: -r[1])
    for h, mu, c in rows:
        print("  %-14s n=%4d  mean m = %.4f" % (h, c, mu))
    grand = m.mean()
    eta2 = sum(c * (mu - grand) ** 2 for _, mu, c in rows) / float(((m - grand) ** 2).sum())
    print("  between-hub variance share (eta^2) = %.4f" % eta2)

    nb = knn_graph(C0, K_NN)
    print("\n=== A: Moran's I of displacement over kNN(stale) graph, k=%d ===" % K_NN)
    I, mu0, sd0, z, p = moran_test(m, nb)
    print("  I = %.4f   perm-null mean %+.4f sd %.4f   z = %.1f   p = %.4f"
          % (I, mu0, sd0, z, p))

    print("\n=== B: variogram over neighbour rank ===")
    print("  %8s %14s %10s" % ("rank", "mean cos-dist", "E|dm|"))
    for r, cd, g in variogram(C0, m):
        print("  %8d %14.4f %10.4f" % (r, cd, g))

    print("\n=== C: probe-kNN predictive power (probe = %.0f%% of corpus) ===" % (100 * PROBE_FRAC))
    n = len(docs)
    npr = max(int(PROBE_FRAC * n), 20)
    res = []
    for trial in range(5):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        mhat = probe_knn_predict(C0, m, pr)
        rho = spearmanr(mhat[rest], m[rest]).statistic
        msh = RNG.permutation(m)
        rho_sh = spearmanr(probe_knn_predict(C0, msh, pr)[rest], msh[rest]).statistic
        kk = int(0.10 * len(rest))
        cap = m[rest[np.argsort(-mhat[rest])[:kk]]].sum() / m[rest[np.argsort(-m[rest])[:kk]]].sum()
        rnd = m[RNG.choice(rest, kk, replace=False)].sum() / m[rest[np.argsort(-m[rest])[:kk]]].sum()
        res.append((rho, rho_sh, cap, rnd))
        print("  trial %d: spearman=%+.3f  (shuffled null %+.3f)   "
              "movement captured @10%% budget: probe-kNN %.3f vs random %.3f"
              % (trial, rho, rho_sh, cap, rnd))
    a = np.array(res)
    print("  MEAN   : spearman=%+.3f (null %+.3f)   captured %.3f vs random %.3f"
          % (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()))

    print("\n=== F: is POSITION predictable, or only MAGNITUDE? ===")
    D = (C1 - C0).astype(np.float32)
    base = np.einsum("ij,ij->i", C0, C1)                    # do nothing: cos(stale, fresh)
    for trial in range(3):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        Dhat = probe_knn_predict_vector(C0, D, pr)
        pred = C0 + Dhat
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        cos_pred = np.einsum("ij,ij->i", pred, C1)
        mhat = probe_knn_predict(C0, m, pr)
        rho = spearmanr(mhat[rest], m[rest]).statistic
        print("  trial %d:  POSITION cos(pred,fresh)=%.4f vs cos(stale,fresh)=%.4f  -> %s"
              % (trial, cos_pred[rest].mean(), base[rest].mean(),
                 "HELPS" if cos_pred[rest].mean() > base[rest].mean() + 1e-3 else "NO GAIN"))
        print("             MAGNITUDE spearman(mhat, m) = %+.3f" % rho)

    json.dump({"n": n, "moran_I": I, "moran_z": z, "moran_p": p, "eta2": float(eta2),
               "spearman_mean": float(a[:, 0].mean()), "spearman_null": float(a[:, 1].mean()),
               "captured_10pct": float(a[:, 2].mean()), "random_10pct": float(a[:, 3].mean()),
               "m_mean": float(m.mean()), "m_sd": float(m.std()),
               "unchanged": int((m < 1e-6).sum())},
              open("e6_summary%s.json" % tag, "w"), indent=1)
    print("\nwrote e6_summary.json")


if __name__ == "__main__":
    pth = sys.argv[1] if len(sys.argv) > 1 else "wp_corpus.json"
    main(pth, "_smoke" if "smoke" in pth else "")
