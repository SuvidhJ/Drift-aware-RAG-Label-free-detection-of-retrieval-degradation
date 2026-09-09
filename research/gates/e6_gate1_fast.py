"""Faster Gate 1: same test, fewer trials/projections, enough to answer the question."""
import json, numpy as np
from gate1_real import sliced_w1, centroid_shift, load_pool
pool = load_pool()
print("pool", pool.shape)
BIG, SMALL, T, NP = 4000, 400, 60, 96

def stats_(R, C, seed):
    mean_, max_ = sliced_w1(R, C, nproj=NP, seed=seed)
    return {"sw1_mean": mean_, "sw1_max": max_, "centroid": centroid_shift(R, C)}

def sample(n_ref, n_cur, T, seed0):
    rng = np.random.default_rng(seed0); acc = {}
    for t in range(T):
        i = rng.choice(len(pool), n_ref + n_cur, replace=n_ref + n_cur > len(pool))
        for k, v in stats_(pool[i[:n_ref]], pool[i[n_ref:]], seed0 + t).items():
            acc.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in acc.items()}

big = sample(BIG, BIG, T, 1)
mix = sample(BIG, SMALL, T, 2)
sml = sample(SMALL, SMALL, T, 3)
thr_big = {k: float(np.quantile(v, 0.95)) for k, v in big.items()}
thr_sml = {k: float(np.quantile(v, 0.95)) for k, v in sml.items()}
print("\n--- calibrate at n=%d, deploy at n=%d (traffic volume drops 10x) ---" % (BIG, SMALL))
print("  %-10s %12s %12s %10s %10s" % ("stat", "thr(n=4000)", "mean(n=400)", "FAR@4000", "FAR@400"))
out = {"thr_big": thr_big, "thr_small": thr_sml, "far": {}}
for k in thr_big:
    far_same = float((big[k] > thr_big[k]).mean())
    far_mix = float((mix[k] > thr_big[k]).mean())
    print("  %-10s %12.5f %12.5f %10.3f %10.3f" % (k, thr_big[k], mix[k].mean(), far_same, far_mix))
    out["far"][k] = {"calib": far_same, "deployed_small": far_mix}
print("\n--- reverse: calibrate at n=%d, deploy at n=%d ---" % (SMALL, BIG))
print("  %-10s %12s %10s" % ("stat", "thr(n=400)", "FAR@4000"))
for k in thr_sml:
    f = float((big[k] > thr_sml[k]).mean())
    print("  %-10s %12.5f %10.3f" % (k, thr_sml[k], f))
    out["far"][k]["deployed_big"] = f
pred = (SMALL ** -0.5 + BIG ** -0.5) / (2 * BIG ** -0.5)
print("\n--- score inflation from the smaller window ---")
print("  predicted mix/big ratio from n^(-1/2): %.3f" % pred)
for k in big:
    r = float(mix[k].mean() / max(big[k].mean(), 1e-12))
    print("  %-10s measured %.3f" % (k, r))
    out.setdefault("ratio", {})[k] = r
out["predicted_ratio"] = float(pred)
json.dump(out, open("results/gate1_fast.json", "w"), indent=1)
print("\nwrote results/gate1_fast.json")
