"""Validate the E6 statistics on data with KNOWN structure, before trusting them on real data.

Positive control : displacement concentrated in a few embedding-space regions
                   -> Moran's I strongly positive, probe-kNN Spearman strongly positive
Null control     : same displacement magnitudes, randomly assigned to documents
                   -> Moran's I ~ 0, probe-kNN Spearman ~ 0
Anti-control     : displacement anti-correlated with neighbourhood
                   -> Moran's I negative
"""
import numpy as np
from scipy.stats import spearmanr
from e6_real import knn_graph, moran_test, probe_knn_predict, variogram

RNG = np.random.default_rng(11)
N, D, NCL = 1200, 64, 24


def make_corpus():
    cent = RNG.normal(size=(NCL, D))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    lab = RNG.integers(0, NCL, N)
    C = cent[lab] + 0.25 * RNG.normal(size=(N, D))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C, lab


def report(name, C, m, nb):
    I, mu, sd, z, p = moran_test(m, nb, nperm=499)
    npr = int(0.02 * N)
    pr = RNG.choice(N, npr, replace=False)
    rest = np.setdiff1d(np.arange(N), pr)
    rho = spearmanr(probe_knn_predict(C, m, pr)[rest], m[rest]).statistic
    print("  %-28s Moran I = %+.4f  z = %+7.1f  p = %.4f   probe-kNN spearman = %+.3f"
          % (name, I, z, p, rho))
    return I, z, rho


if __name__ == "__main__":
    C, lab = make_corpus()
    nb = knn_graph(C, 15)
    print("synthetic corpus: N=%d D=%d clusters=%d\n" % (N, D, NCL))

    # POSITIVE: 6 of 24 clusters are "hot" and churn hard, the rest barely move
    hot = set(RNG.choice(NCL, 6, replace=False).tolist())
    m_pos = np.where(np.isin(lab, list(hot)),
                     RNG.gamma(4, 0.15, N), RNG.gamma(1.2, 0.02, N))
    I1, z1, r1 = report("POSITIVE (regional churn)", C, m_pos, nb)

    # NULL: identical magnitudes, shuffled across documents
    m_null = RNG.permutation(m_pos)
    I2, z2, r2 = report("NULL (same values, shuffled)", C, m_null, nb)

    # RESOLUTION: same field, but a probe dense enough to resolve 24 regions.
    # test_probe_density.py showed predictive power tracks probes-per-region, not
    # probe-percent; at 2% of N=1200 there is ~1 probe per cluster.
    npr_big = int(0.10 * N)
    pr = RNG.choice(N, npr_big, replace=False)
    rest = np.setdiff1d(np.arange(N), pr)
    r3 = spearmanr(probe_knn_predict(C, m_pos, pr)[rest], m_pos[rest]).statistic
    print("  %-28s probe = %d (%.1f per cluster)          probe-kNN spearman = %+.3f"
          % ("RESOLUTION (10%% probe)", npr_big, npr_big / NCL, r3))

    print("\n  variogram on POSITIVE (E|dm| should RISE with neighbour rank):")
    for r, cd, g in variogram(C, m_pos):
        print("    rank %4d  mean cos-dist %.4f  E|dm| = %.4f" % (r, cd, g))

    print("\n=== assertions ===")
    ok = True
    for cond, msg in [
        (I1 > 0.3,            "positive control has strong positive Moran I"),
        (z1 > 10,             "positive control null-rejects decisively"),
        (r1 > 0.15,           "positive control is predictable even at 1 probe/cluster"),
        (abs(I2) < 0.08,      "null control has Moran I near zero"),
        (abs(z2) < 4,         "null control does not reject"),
        (abs(r2) < 0.15,      "null control is not predictable"),
        (r3 > r1 + 0.10,      "denser probe resolves regions better (r3 >> r1)"),
    ]:
        print("  [%s] %s" % ("PASS" if cond else "FAIL", msg))
        ok &= bool(cond)
    print("\nSTATISTICS VALIDATED" if ok else "\nSOMETHING IS WRONG - do not trust E6")
