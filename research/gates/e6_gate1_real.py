"""Gate 1 on REAL embeddings - open question #2 in FINDINGS.md sec.10.

Does the calibrated-threshold failure under traffic-volume shift reproduce on real
vectors?  The simulator gave FAR 0.057 -> 1.000 calibrating at n=4000 and deploying at
n=400, and -> 0.000 in the reverse direction, with score ratios ~0.51 against a
predicted 0.53 from (n_cur^-1/2 + n_ref^-1/2).  The mechanism is distribution-free, so
it should reproduce; "should" is not a result.
"""
import glob, json, os, numpy as np

RNG = np.random.default_rng(4)
NPROJ = 256


def sliced_w1(X, Y, nproj=NPROJ, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    T = rng.normal(size=(d, nproj))
    T /= np.linalg.norm(T, axis=0, keepdims=True)
    A, B = np.sort(X @ T, axis=0), np.sort(Y @ T, axis=0)
    q = np.linspace(0, 1, min(len(X), len(Y)))
    ai = np.stack([np.quantile(A[:, j], q) for j in range(nproj)])
    bi = np.stack([np.quantile(B[:, j], q) for j in range(nproj)])
    per = np.abs(ai - bi).mean(axis=1)
    return float(per.mean()), float(per.max())


def mmd_rbf(X, Y, gamma=None):
    Z = np.vstack([X, Y])
    if gamma is None:
        s = Z[RNG.choice(len(Z), min(400, len(Z)), replace=False)]
        d2 = ((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / max(med, 1e-9)

    def K(A, B):
        return np.exp(-gamma * (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :]
                                - 2 * A @ B.T))
    n, m = len(X), len(Y)
    kxx = (K(X, X).sum() - n) / max(n * (n - 1), 1)
    kyy = (K(Y, Y).sum() - m) / max(m * (m - 1), 1)
    kxy = K(X, Y).mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def centroid_shift(X, Y):
    return float(np.linalg.norm(X.mean(0) - Y.mean(0)))


def load_pool():
    """Largest available real embedding matrix."""
    best, bn = None, 0
    for f in glob.glob("C0*.npy") + glob.glob("C1*.npy"):
        try:
            a = np.load(f)
        except Exception:
            continue
        if len(a) > bn:
            best, bn = a, len(a)
    return best


def far_at(pool, n_ref, n_cur, thr, trials=200, seed=0):
    """False-alarm rate: both windows drawn from the SAME distribution."""
    rng = np.random.default_rng(seed)
    fa = {k: 0 for k in thr}
    for t in range(trials):
        i = rng.choice(len(pool), n_ref + n_cur, replace=n_ref + n_cur > len(pool))
        R, C = pool[i[:n_ref]], pool[i[n_ref:]]
        mean_, max_ = sliced_w1(R, C, seed=t)
        s = {"sw1_mean": mean_, "sw1_max": max_, "centroid": centroid_shift(R, C)}
        if n_cur <= 800:
            s["mmd"] = mmd_rbf(R[:800], C)
        for k, v in s.items():
            if k in thr and v > thr[k]:
                fa[k] += 1
    return {k: fa[k] / trials for k in thr}


def calibrate(pool, n_ref, n_cur, trials=200, q=0.95, seed=99):
    rng = np.random.default_rng(seed)
    acc = {"sw1_mean": [], "sw1_max": [], "centroid": [], "mmd": []}
    for t in range(trials):
        i = rng.choice(len(pool), n_ref + n_cur, replace=n_ref + n_cur > len(pool))
        R, C = pool[i[:n_ref]], pool[i[n_ref:]]
        mean_, max_ = sliced_w1(R, C, seed=1000 + t)
        acc["sw1_mean"].append(mean_)
        acc["sw1_max"].append(max_)
        acc["centroid"].append(centroid_shift(R, C))
        if n_cur <= 800:
            acc["mmd"].append(mmd_rbf(R[:800], C))
    return {k: float(np.quantile(v, q)) for k, v in acc.items() if v}, \
           {k: float(np.mean(v)) for k, v in acc.items() if v}


if __name__ == "__main__":
    pool = load_pool()
    if pool is None:
        print("no embeddings available")
        raise SystemExit(1)
    print("=" * 78)
    print("GATE 1 ON REAL EMBEDDINGS   pool = %d vectors, d = %d" % pool.shape)
    print("=" * 78)

    BIG, SMALL = 4000, 400
    if len(pool) < 1200:
        BIG, SMALL = 1000, 100
        print("(pool small; using n=%d / %d)" % (BIG, SMALL))

    out = {}
    print("\n--- calibrate at n_ref=n_cur=%d, deploy at n_cur=%d ---" % (BIG, SMALL))
    thr, mean_big = calibrate(pool, BIG, BIG)
    print("  thresholds (95th pct of null):", {k: round(v, 5) for k, v in thr.items()})
    far_same = far_at(pool, BIG, BIG, thr)
    far_small = far_at(pool, BIG, SMALL, {k: v for k, v in thr.items() if k != "mmd"})
    print("  FAR at calibration size  :", {k: round(v, 3) for k, v in far_same.items()})
    print("  FAR deployed at n=%-6d:" % SMALL, {k: round(v, 3) for k, v in far_small.items()})
    out["far_calibration"] = far_same
    out["far_deployed_small"] = far_small

    print("\n--- reverse: calibrate small, deploy big ---")
    thr_s, mean_small = calibrate(pool, SMALL, SMALL)
    far_big = far_at(pool, BIG, BIG, {k: v for k, v in thr_s.items() if k != "mmd"})
    print("  FAR deployed at n=%-6d:" % BIG, {k: round(v, 3) for k, v in far_big.items()})
    out["far_deployed_big"] = far_big

    print("\n--- score ratio vs the n^(-1/2) prediction ---")
    pred = ((SMALL ** -0.5 + BIG ** -0.5) / (BIG ** -0.5 + BIG ** -0.5))
    print("  predicted ratio  small/big = %.3f" % pred)
    _, mean_mix = calibrate(pool, BIG, SMALL)
    for k in ["sw1_mean", "sw1_max", "centroid"]:
        if k in mean_big and k in mean_mix:
            print("  %-10s measured %.4f / %.4f = %.3f"
                  % (k, mean_big[k], mean_mix[k], mean_big[k] / max(mean_mix[k], 1e-12)))
            out.setdefault("ratio", {})[k] = float(mean_big[k] / max(mean_mix[k], 1e-12))
    out["predicted_ratio"] = float(pred)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/gate1_real.json", "w"), indent=1)
    print("\nwrote results/gate1_real.json")
