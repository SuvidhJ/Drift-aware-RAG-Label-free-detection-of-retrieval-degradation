import collections, json
import numpy as np
from vcommon import CACHE, P9, auroc, jl, load_emb, mine_commit, pages, sample, versions, wilson
pgs = pages(); _, _, qs = sample(); vs, fs = versions(pgs)
cases = jl(CACHE / "p13_cases.jsonl")
R11 = jl(CACHE / "p11/records_scale.jsonl")
print("p13 cases", len(cases), collections.Counter((c["retriever"], c["gen"]) for c in cases))
# each case must be a p11 append/none record with identical output
key11 = {(r["retriever"], r["gen"], r["q"]): r["output"] for r in R11 if r["policy"] == "append" and r["variant"] == "none"}
print("cases matching p11 output:", sum(key11.get((c["retriever"], c["gen"], c["q"])) == c["output"] for c in cases))
# label check vs P9.commit and excluded "both" count
lab_ok = sum(c["commit"] == P9.commit(qs[c["q"]]["new"], qs[c["q"]]["old"], c["output"]) for c in cases)
print("stored commit == P9.commit:", lab_ok, "/", len(cases))
for retr in ("minilm", "bge"):
    nb = sum(P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"]) == "both" for r in R11
             if r["retriever"] == retr and r["policy"] == "append" and r["variant"] == "none" and r["gen"] in ("phi3:mini", "qwen2.5:7b-instruct"))
    print(f"[{retr}] both excluded = {nb}")
rng = np.random.default_rng(7)
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    cs = [c for c in cases if c["gen"] == gen]
    pos = [c for c in cs if c["commit"] == "old"]; neg = [c for c in cs if c["commit"] == "new"]
    print(f"\n{gen}: stale {len(pos)} correct {len(neg)}")
    for m in ("M2", "M4"):
        tp = sum(c[f"{m}_flag"] for c in pos); fp = sum(c[f"{m}_flag"] for c in neg)
        print(f"  {m}: TPR {tp}/{len(pos)} = {tp/len(pos):.3f} Wilson {np.round(wilson(tp, len(pos)), 3)}  FPR {fp}/{len(neg)} = {fp/len(neg):.3f} Wilson {np.round(wilson(fp, len(neg)), 3)}")
    print(f"  M3b AUROC {auroc([c['M3b_score'] for c in pos], [c['M3b_score'] for c in neg]):.3f}")
    ag = np.mean([c["M4_flag"] == (c["commit"] == "old") for c in cs])
    print(f"  M4 agreement {ag:.4f} (n={len(cs)}); disagreements:")
    for c in cs:
        if c["M4_flag"] != (c["commit"] == "old"):
            q = qs[c["q"]]
            print(f"     {c['retriever']} flag={c['M4_flag']} commit={c['commit']} out={c['output']!r} new={q['new']} old={q['old']}")
    # M3b per retriever (the pooled number mixes two embedding spaces)
    for retr in ("minilm", "bge"):
        p = [c["M3b_score"] for c in pos if c["retriever"] == retr]; n = [c["M3b_score"] for c in neg if c["retriever"] == retr]
        print(f"  M3b AUROC [{retr}] {auroc(p, n):.3f}")

# ---- M3a: independent permutation tests (own sliced-W1 and own RBF-MMD)
uniq = sorted({x for v0, v1 in vs for x in v0 + v1}); idx = {x: i for i, x in enumerate(uniq)}
changed = [(x, y) for v0, v1 in vs for x, y in zip(sorted(set(v0) - set(v1)), sorted(set(v1) - set(v0)))]
X_all = sorted({x for v0, v1 in vs for x in set(v0) - set(v1)}); Y_all = sorted({y for v0, v1 in vs for y in set(v1) - set(v0)})
print(f"\nM3a changed pairs n={len(changed)}; all removed {len(X_all)}, all added {len(Y_all)}")
def sw1(A, B, P):
    a = np.sort(A @ P, axis=0); b = np.sort(B @ P, axis=0)
    # equal sizes here
    return np.mean(np.abs(a - b))
def mmd2(K, n):
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    m = len(K) - n
    return (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean()
for retr in ("minilm", "bge"):
    E = load_emb(retr, "p11_scale", "passages", uniq); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    for seed in (101, 202, 303):
        r = np.random.default_rng(seed)
        X = E[[idx[x] for x, _ in changed]]; Y = E[[idx[y] for _, y in changed]]
        X = X[r.choice(len(X), 1000, replace=False)]; Y = Y[r.choice(len(Y), 1000, replace=False)]
        Z = np.vstack([X, Y]); n = len(X)
        P = r.normal(size=(Z.shape[1], 200)); P /= np.linalg.norm(P, axis=0)
        D2 = np.maximum(2 - 2 * Z @ Z.T, 0)
        sig2 = np.median(D2[np.triu_indices(len(Z), 1)])
        K = np.exp(-D2 / sig2)
        o1, o2 = sw1(X, Y, P), mmd2(K, n)
        n1 = n2 = 0
        for _ in range(499):
            pi = r.permutation(len(Z)); A, B = Z[pi[:n]], Z[pi[n:]]
            n1 += sw1(A, B, P) >= o1
            Kp = K[np.ix_(pi, pi)]; n2 += mmd2(Kp, n) >= o2
        print(f"  [{retr}] seed {seed}: sliced-W1 p = {(1+n1)/500:.3f}; MMD p = {(1+n2)/500:.3f}")
