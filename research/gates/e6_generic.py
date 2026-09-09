"""E6 analysis for any corpus: Moran's I, variogram, probe-kNN, position-vs-magnitude.

usage: e6_generic.py <corpus.json> <embedding_tag> <name>
"""
import json, sys, numpy as np
from scipy.stats import spearmanr
from e6_real import (knn_graph, moran_test, variogram,
                     probe_knn_predict, probe_knn_predict_vector)

RNG = np.random.default_rng(20260906)
K_NN, POW = 15, 8

corpus, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(corpus, encoding="utf-8"))
C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
n = min(len(docs), len(C0), len(C1))
docs, C0, C1 = docs[:n], C0[:n], C1[:n]
m = np.linalg.norm(C1 - C0, axis=1)
hubs = np.array([d["hub"] for d in docs])
changed = np.array([d["text0"] != d["text1"] for d in docs])

print("=" * 78)
print("E6 :: %s   n=%d   %d groups" % (name.upper(), n, len(set(hubs.tolist()))))
print("=" * 78)
print("displacement: mean %.4f sd %.4f p50 %.4f p90 %.4f max %.4f"
      % (m.mean(), m.std(), np.median(m), np.quantile(m, .9), m.max()))
print("changed: %d (%.1f%%)   unchanged: %d" % (changed.sum(), 100 * changed.mean(),
                                                (~changed).sum()))
if changed.any() and (~changed).any():
    print("mean m | changed = %.4f   | unchanged = %.4f"
          % (m[changed].mean(), m[~changed].mean()))

print("\n--- churn by group (top/bottom 8) ---")
rows = sorted(((h, float(m[hubs == h].mean()), int((hubs == h).sum()),
                float(changed[hubs == h].mean())) for h in set(hubs.tolist())),
              key=lambda r: -r[1])
for h, mu, c, ch in rows[:8]:
    print("  %-34s n=%4d  m=%.4f  changed=%.0f%%" % (h[:34], c, mu, 100 * ch))
if len(rows) > 16:
    print("  ...")
for h, mu, c, ch in rows[-8:]:
    print("  %-34s n=%4d  m=%.4f  changed=%.0f%%" % (h[:34], c, mu, 100 * ch))
grand = m.mean()
eta2 = (sum(c * (mu - grand) ** 2 for _, mu, c, _ in rows)
        / max(float(((m - grand) ** 2).sum()), 1e-12))
print("  between-group variance share (eta^2) = %.4f" % eta2)

nb = knn_graph(C0, K_NN)
print("\n--- A: Moran's I of displacement over kNN(stale), k=%d ---" % K_NN)
I, mu0, sd0, z, p = moran_test(m, nb)
print("  I = %+.4f   null mean %+.4f sd %.4f   z = %+.1f   p = %.4f" % (I, mu0, sd0, z, p))
Ib, _, _, zb, pb = moran_test(changed.astype(float), nb)
print("  binary revised-or-not field: I = %+.4f  z = %+.1f  p = %.4f" % (Ib, zb, pb))

print("\n--- B: variogram over neighbour rank ---")
vg = variogram(C0, m)
for r, cd, g in vg:
    print("  rank %5d  cos-dist %.4f  E|dm| %.4f" % (r, cd, g))

print("\n--- C: probe-kNN predictive power ---")
summary = {}
for frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
    npr = max(int(frac * n), 20)
    if npr >= n:
        continue
    rr, rs, cap, rnd = [], [], [], []
    for _ in range(5):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        mhat = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
        rr.append(spearmanr(mhat[rest], m[rest]).statistic)
        msh = RNG.permutation(m)
        rs.append(spearmanr(probe_knn_predict(C0, msh, pr, k=K_NN, power=POW)[rest],
                            msh[rest]).statistic)
        kk = max(int(0.10 * len(rest)), 1)
        orc = m[rest[np.argsort(-m[rest])[:kk]]].sum()
        cap.append(m[rest[np.argsort(-mhat[rest])[:kk]]].sum() / max(orc, 1e-12))
        rnd.append(m[RNG.choice(rest, kk, replace=False)].sum() / max(orc, 1e-12))
    print("  probe %5.1f%% (%5d): spearman %+.3f (null %+.3f)   captured@10%% %.3f vs random %.3f"
          % (100 * frac, npr, np.mean(rr), np.mean(rs), np.mean(cap), np.mean(rnd)))
    summary["probe_%.2f" % frac] = dict(spearman=float(np.mean(rr)), null=float(np.mean(rs)),
                                        captured=float(np.mean(cap)), random=float(np.mean(rnd)))

print("\n--- F: position vs magnitude ---")
D = (C1 - C0).astype(np.float32)
base = np.einsum("ij,ij->i", C0, C1)
for frac in [0.02, 0.10]:
    npr = max(int(frac * n), 20)
    if npr >= n:
        continue
    cp, bs, rho = [], [], []
    for _ in range(3):
        pr = RNG.choice(n, npr, replace=False)
        rest = np.setdiff1d(np.arange(n), pr)
        pred = C0 + probe_knn_predict_vector(C0, D, pr, k=K_NN, power=POW)
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        cp.append(np.einsum("ij,ij->i", pred, C1)[rest].mean())
        bs.append(base[rest].mean())
        rho.append(spearmanr(probe_knn_predict(C0, m, pr, k=K_NN, power=POW)[rest],
                             m[rest]).statistic)
    verdict = "HELPS" if np.mean(cp) > np.mean(bs) + 1e-3 else "NO GAIN"
    print("  probe %4.0f%%: POSITION %.4f vs do-nothing %.4f -> %s | MAGNITUDE rho %+.3f"
          % (100 * frac, np.mean(cp), np.mean(bs), verdict, np.mean(rho)))
    summary["F_%.2f" % frac] = dict(pos_pred=float(np.mean(cp)), pos_base=float(np.mean(bs)),
                                    mag_spearman=float(np.mean(rho)))

summary.update(name=name, n=n, moran_I=I, moran_z=z, moran_p=p, moran_binary_I=Ib,
               moran_binary_z=zb, eta2=float(eta2), m_mean=float(m.mean()),
               frac_changed=float(changed.mean()), n_groups=len(set(hubs.tolist())),
               variogram=[list(v) for v in vg])
json.dump(summary, open("results/e6_%s.json" % name, "w"), indent=1)
print("\nwrote results/e6_%s.json" % name)
