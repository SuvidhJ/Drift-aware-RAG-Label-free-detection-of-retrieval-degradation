"""Why did probe-kNN score only +0.227 on a corpus with Moran I = +0.42?

Hypothesis: the probe must resolve the churn REGIONS, not just be a fixed % of N.
At N=1200 with 24 clusters, a 2% probe is 24 documents - one per cluster - and k=15
then averages each prediction over 15 of the 24 probes, across many clusters at once.

Sweep probe fraction and k to test it.
"""
import numpy as np
from scipy.stats import spearmanr
from e6_real import knn_graph, probe_knn_predict, morans_I

RNG = np.random.default_rng(11)
N, D = 1200, 64


def corpus(ncl):
    cent = RNG.normal(size=(ncl, D))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    lab = RNG.integers(0, ncl, N)
    C = cent[lab] + 0.25 * RNG.normal(size=(N, D))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    hot = set(RNG.choice(ncl, max(ncl // 4, 1), replace=False).tolist())
    m = np.where(np.isin(lab, list(hot)), RNG.gamma(4, 0.15, N), RNG.gamma(1.2, 0.02, N))
    return C, m


def rho_for(C, m, frac, k, trials=5):
    out = []
    for _ in range(trials):
        npr = max(int(frac * N), 5)
        pr = RNG.choice(N, npr, replace=False)
        rest = np.setdiff1d(np.arange(N), pr)
        out.append(spearmanr(probe_knn_predict(C, m, pr, k=k)[rest], m[rest]).statistic)
    return float(np.mean(out))


if __name__ == "__main__":
    for ncl in [8, 24]:
        C, m = corpus(ncl)
        nb = knn_graph(C, 15)
        print("\n=== %d clusters, N=%d, Moran I = %+.3f ===" % (ncl, N, morans_I(m, nb)))
        print("  %-8s" % "probe%", "".join("   k=%-6d" % k for k in [3, 5, 10, 15, 25]))
        for frac in [0.02, 0.05, 0.10, 0.20]:
            npr = int(frac * N)
            row = "".join("  %+.3f  " % rho_for(C, m, frac, k) for k in [3, 5, 10, 25, 25])
            print("  %-8s" % ("%.0f%% (%d)" % (100 * frac, npr)), row)
        print("  probes per cluster at 2%%: %.1f" % (0.02 * N / ncl))
