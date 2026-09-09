"""E6 on the real arXiv corpus, using the cached embeddings."""
import json, sys, numpy as np
from scipy.stats import spearmanr
from e6_real import (knn_graph, moran_test, variogram,
                     probe_knn_predict, probe_knn_predict_vector)

RNG = np.random.default_rng(20260906)
K_NN, POW = 15, 8

docs = json.load(open("ax_corpus.json", encoding="utf-8"))
C0, C1 = np.load("C0_ax.npy"), np.load("C1_ax.npy")
n = len(docs)
assert len(C0) == len(C1) == n
m = np.linalg.norm(C1 - C0, axis=1)
hubs = np.array([d["hub"] for d in docs])
changed = np.array([d["text0"] != d["text1"] for d in docs])

print("=" * 78)
print("E6 - REAL DATA: arXiv abstracts, v1 (2023 H1) vs latest (Sep 2026)")
print("     n=%d documents, %d categories, MiniLM-L6-v2 (numpy), 384-d" % (n, len(set(hubs.tolist()))))
print("=" * 78)

print("\n=== displacement summary ===")
print("  mean %.4f  sd %.4f  p50 %.4f  p90 %.4f  max %.4f"
      % (m.mean(), m.std(), np.median(m), np.quantile(m, .9), m.max()))
print("  abstract unchanged: %d (%.1f%%)   changed: %d (%.1f%%)"
      % ((~changed).sum(), 100 * (~changed).mean(), changed.sum(), 100 * changed.mean()))
print("  mean displacement | unchanged = %.4f   | changed = %.4f"
      % (m[~changed].mean(), m[changed].mean()))

print("\n=== E: churn by arXiv category ===")
rows = sorted(((h, float(m[hubs == h].mean()), int((hubs == h).sum()),
                float(changed[hubs == h].mean())) for h in sorted(set(hubs.tolist()))),
              key=lambda r: -r[1])
for h, mu, c, ch in rows:
    print("  %-20s n=%4d  mean m = %.4f   changed = %.0f%%" % (h, c, mu, 100 * ch))
grand = m.mean()
eta2 = sum(c * (mu - grand) ** 2 for _, mu, c, _ in rows) / float(((m - grand) ** 2).sum())
print("  between-category variance share (eta^2) = %.4f" % eta2)

nb = knn_graph(C0, K_NN)
print("\n=== A: Moran's I of displacement over kNN(stale) graph, k=%d ===" % K_NN)
I, mu0, sd0, z, p = moran_test(m, nb)
print("  I = %+.4f   perm-null mean %+.4f sd %.4f   z = %+.1f   p = %.4f"
      % (I, mu0, sd0, z, p))
Ib, _, _, zb, pb = moran_test(changed.astype(float), nb)
print("  (binary was-it-revised field: I = %+.4f  z = %+.1f  p = %.4f)" % (Ib, zb, pb))

print("\n=== B: variogram over neighbour rank ===")
print("  %8s %14s %10s" % ("rank", "mean cos-dist", "E|dm|"))
for r, cd, g in variogram(C0, m):
    print("  %8d %14.4f %10.4f" % (r, cd, g))

print("\n=== C: probe-kNN predictive power ===")
summary = {}
for frac in [0.02, 0.05, 0.10, 0.20]:
    npr = max(int(frac * n), 20)
    rr, rs, cap, rnd = [], [], [], []
    for t in range(5):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        mhat = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
        rr.append(spearmanr(mhat[rest], m[rest]).statistic)
        msh = RNG.permutation(m)
        rs.append(spearmanr(probe_knn_predict(C0, msh, pr, k=K_NN, power=POW)[rest],
                            msh[rest]).statistic)
        kk = int(0.10 * len(rest))
        oracle = m[rest[np.argsort(-m[rest])[:kk]]].sum()
        cap.append(m[rest[np.argsort(-mhat[rest])[:kk]]].sum() / oracle)
        rnd.append(m[RNG.choice(rest, kk, replace=False)].sum() / oracle)
    print("  probe %5.0f%% (%4d docs): spearman %+.3f (null %+.3f)   "
          "movement captured @10%% budget: %.3f vs random %.3f"
          % (100 * frac, npr, np.mean(rr), np.mean(rs), np.mean(cap), np.mean(rnd)))
    summary["probe_%.2f" % frac] = dict(spearman=float(np.mean(rr)),
                                        null=float(np.mean(rs)),
                                        captured=float(np.mean(cap)),
                                        random=float(np.mean(rnd)))

print("\n=== F: is POSITION predictable, or only MAGNITUDE? ===")
D = (C1 - C0).astype(np.float32)
base = np.einsum("ij,ij->i", C0, C1)
for frac in [0.02, 0.10]:
    npr = max(int(frac * n), 20)
    cp, bs, rho = [], [], []
    for t in range(3):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        pred = C0 + probe_knn_predict_vector(C0, D, pr, k=K_NN, power=POW)
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        cp.append(np.einsum("ij,ij->i", pred, C1)[rest].mean())
        bs.append(base[rest].mean())
        rho.append(spearmanr(probe_knn_predict(C0, m, pr, k=K_NN, power=POW)[rest],
                             m[rest]).statistic)
    print("  probe %4.0f%%:  POSITION cos(pred,fresh)=%.4f  vs  do-nothing cos(stale,fresh)=%.4f"
          "   -> %s" % (100 * frac, np.mean(cp), np.mean(bs),
                        "HELPS" if np.mean(cp) > np.mean(bs) + 1e-3 else "NO GAIN"))
    print("               MAGNITUDE spearman(mhat, m) = %+.3f" % np.mean(rho))
    summary["F_probe_%.2f" % frac] = dict(pos_pred=float(np.mean(cp)),
                                          pos_base=float(np.mean(bs)),
                                          mag_spearman=float(np.mean(rho)))

summary.update(n=n, moran_I=I, moran_z=z, moran_p=p, moran_binary_I=Ib, moran_binary_z=zb,
               eta2=float(eta2), m_mean=float(m.mean()), frac_changed=float(changed.mean()))
json.dump(summary, open("e6_real_summary.json", "w"), indent=1)
print("\nwrote e6_real_summary.json")
