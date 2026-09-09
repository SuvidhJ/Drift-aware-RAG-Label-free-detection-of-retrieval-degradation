"""Detection power near the floor - open question #3 in FINDINGS.md sec.10.

The existing design saturates power at 1.000 in every cell, so power collapse cannot be
claimed from it (FINDINGS sec.4.4).  This sweeps effect size DOWN through the detection
floor on real embeddings so the power curve is informative, and reports the effect size
at which each detector reaches 50% and 80% power at a calibrated 5% false-alarm rate.
"""
import glob, json, os, numpy as np
from gate1_real import sliced_w1, mmd_rbf, centroid_shift, load_pool

RNG = np.random.default_rng(21)


def perturb(X, frac, scale, rng):
    """Revise a fraction of documents by moving them a controlled distance."""
    Y = X.copy()
    k = max(int(frac * len(X)), 1)
    idx = rng.choice(len(X), k, replace=False)
    d = rng.normal(size=(k, X.shape[1]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Y[idx] = X[idx] + scale * d
    Y[idx] /= np.maximum(np.linalg.norm(Y[idx], axis=1, keepdims=True), 1e-12)
    return Y


def stats(R, C, seed):
    mean_, max_ = sliced_w1(R, C, seed=seed)
    s = {"sw1_mean": mean_, "sw1_max": max_, "centroid": centroid_shift(R, C)}
    if len(R) <= 900 and len(C) <= 900:
        s["mmd"] = mmd_rbf(R, C)
    return s


if __name__ == "__main__":
    pool = load_pool()
    if pool is None:
        print("no embeddings")
        raise SystemExit(1)
    n = min(800, len(pool) // 2)
    print("=" * 78)
    print("POWER NEAR THE DETECTION FLOOR   pool=%d  window n=%d" % (len(pool), n))
    print("=" * 78)

    # calibrate a 5% false-alarm threshold from the null
    print("\ncalibrating null (5%% FAR)...", flush=True)
    null = {}
    for t in range(150):
        i = RNG.choice(len(pool), 2 * n, replace=2 * n > len(pool))
        s = stats(pool[i[:n]], pool[i[n:]], seed=t)
        for k, v in s.items():
            null.setdefault(k, []).append(v)
    thr = {k: float(np.quantile(v, 0.95)) for k, v in null.items()}
    print("  thresholds:", {k: round(v, 6) for k, v in thr.items()})

    FRACS = [0.10, 0.30]
    SCALES = [0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00]
    out = {"thresholds": thr, "power": {}}
    for frac in FRACS:
        print("\n--- %.0f%% of documents revised ---" % (100 * frac))
        hdr = "  %8s" % "scale" + "".join("%12s" % k for k in thr)
        print(hdr)
        for sc in SCALES:
            hits = {k: 0 for k in thr}
            T = 60
            for t in range(T):
                i = RNG.choice(len(pool), 2 * n, replace=2 * n > len(pool))
                R = pool[i[:n]]
                C = perturb(pool[i[n:]], frac, sc, RNG)
                s = stats(R, C, seed=5000 + t)
                for k, v in s.items():
                    if k in thr and v > thr[k]:
                        hits[k] += 1
            row = "  %8.2f" % sc + "".join("%12.3f" % (hits[k] / T) for k in thr)
            print(row, flush=True)
            out["power"]["%.2f_%.2f" % (frac, sc)] = {k: hits[k] / T for k in thr}

    print("\n--- effect size at 50%% and 80%% power ---")
    for frac in FRACS:
        for k in thr:
            p = [(sc, out["power"]["%.2f_%.2f" % (frac, sc)][k]) for sc in SCALES]
            e50 = next((sc for sc, v in p if v >= 0.5), None)
            e80 = next((sc for sc, v in p if v >= 0.8), None)
            print("  frac %.0f%%  %-10s  50%% power at scale %s   80%% at %s"
                  % (100 * frac, k, e50, e80))
            out.setdefault("thresholds_power", {})["%.2f_%s" % (frac, k)] = dict(e50=e50, e80=e80)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/power_floor.json", "w"), indent=1)
    print("\nwrote results/power_floor.json")
