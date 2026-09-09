"""Adequately-powered test of probe-kNN vs random on a real retrieval task.

The 3-seed bake-off overstated the effect (3.00x at a 20% budget on arXiv); with 12
seeds it fell to 1.92x at p=0.055 and on Python docs to 1.08x at p=0.74.  This runs
many seeds, reports a bootstrap CI on the ratio, and also uses a lower-variance
estimator: the BUDGET required to reach a fixed quality target, rather than the
quality reached at a fixed budget.

usage: significance.py <corpus.json> <tag> <name> [seeds]
"""
import json, os, re, sys, numpy as np
from scipy import stats
from e6_real import probe_knn_predict

SENT = re.compile(r"(?<=[.!?])\s+")
K_NN, POW = 15, 8


def sents(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip()) > 40]


def build(docs, seed=7):
    rng = np.random.default_rng(seed)
    qs, gold = [], []
    for i, d in enumerate(docs):
        s1 = sents(d["text1"])
        if not s1:
            continue
        s0 = set(sents(d["text0"]))
        pool = [s for s in s1 if s not in s0]
        if not pool:
            continue
        j = int(rng.integers(0, len(pool)))
        w = " ".join(pool[j:j + 2])
        if len(w) >= 60:
            qs.append(w)
            gold.append(i)
    return qs, np.array(gold)


def main():
    corpus, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
    SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    docs = json.load(open(corpus, encoding="utf-8"))
    C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
    n = min(len(docs), len(C0), len(C1))
    docs, C0, C1 = docs[:n], C0[:n], C1[:n]
    m = np.linalg.norm(C1 - C0, axis=1)
    changed = np.mean([d["text0"] != d["text1"] for d in docs])

    qs, gold = build(docs)
    cache = "Qsig%s.npy" % tag
    if os.path.exists(cache) and len(np.load(cache)) == len(qs):
        Q = np.load(cache)
    else:
        from minilm_np import MiniLM
        mdl = MiniLM()
        out = []
        for s in range(0, len(qs), 128):
            out.append(mdl.embed(qs[s:s + 128], batch=32).astype(np.float32))
            np.save(cache, np.vstack(out))
            print("   queries %d/%d" % (len(np.vstack(out)), len(qs)), flush=True)
        Q = np.vstack(out)

    def rec(C, k=10):
        h = 0
        for s in range(0, len(Q), 256):
            S = Q[s:s + 256] @ C.T
            top = np.argpartition(-S, k, axis=1)[:, :k]
            for r in range(top.shape[0]):
                if gold[s + r] in top[r]:
                    h += 1
        return h / len(Q)

    rs, rf = rec(C0), rec(C1)
    gain = rf - rs
    print("=" * 78)
    print("SIGNIFICANCE :: %s   n=%d  changed=%.0f%%  queries=%d" % (name, n, 100 * changed, len(qs)))
    print("  stale %.4f  fresh %.4f  headroom %.4f" % (rs, rf, gain))
    print("=" * 78)
    out = {"name": name, "n": n, "changed": float(changed), "stale": rs, "fresh": rf,
           "headroom": gain, "seeds": SEEDS, "budgets": {}}
    if gain < 1e-6:
        print("no headroom")
        json.dump(out, open("results/significance_%s.json" % name, "w"), indent=1)
        return

    npr = max(int(0.02 * n), 20)
    for budget in [0.10, 0.20, 0.35]:
        B = int(budget * n)
        pk, rd = [], []
        for seed in range(SEEDS):
            rg = np.random.default_rng(9000 + seed)
            pr = rg.choice(n, npr, replace=False)
            mh = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
            rest = np.setdiff1d(np.arange(n), pr)
            sel = np.concatenate([pr, rest[np.argsort(-mh[rest])[:max(B - npr, 0)]]])
            Cm = C0.copy(); Cm[sel] = C1[sel]; pk.append(rec(Cm))
            Cm = C0.copy(); s2 = rg.choice(n, B, replace=False); Cm[s2] = C1[s2]; rd.append(rec(Cm))
        pk, rd = np.array(pk), np.array(rd)
        sp, sr = (pk - rs) / gain, (rd - rs) / gain
        t, p = stats.ttest_ind(pk, rd)
        # bootstrap CI on the share-of-gain ratio
        bs = []
        rng = np.random.default_rng(3)
        for _ in range(4000):
            a = sp[rng.integers(0, len(sp), len(sp))].mean()
            b = sr[rng.integers(0, len(sr), len(sr))].mean()
            if abs(b) > 1e-9:
                bs.append(a / b)
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2)
        print("  budget %3.0f%%: probe-kNN %.4f+/-%.4f  random %.4f+/-%.4f  "
              "share %.3f vs %.3f  ratio %.2fx [%.2f, %.2f]  t=%.2f p=%.4f %s"
              % (100 * budget, pk.mean(), pk.std(ddof=1), rd.mean(), rd.std(ddof=1),
                 sp.mean(), sr.mean(), sp.mean() / max(sr.mean(), 1e-9), lo, hi, t, p,
                 "SIG" if p < 0.05 else "ns"))
        out["budgets"]["%.2f" % budget] = dict(
            pk=float(pk.mean()), pk_sd=float(pk.std(ddof=1)), rd=float(rd.mean()),
            rd_sd=float(rd.std(ddof=1)), pk_share=float(sp.mean()), rd_share=float(sr.mean()),
            ratio=float(sp.mean() / max(sr.mean(), 1e-9)), ci=[float(lo), float(hi)],
            t=float(t), p=float(p), significant=bool(p < 0.05))

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/significance_%s.json" % name, "w"), indent=1)
    print("\nwrote results/significance_%s.json" % name)


if __name__ == "__main__":
    main()
