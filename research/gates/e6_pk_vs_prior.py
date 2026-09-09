"""probe-kNN vs the closest deployed prior art (cluster-threshold refresh), 30 seeds.

This is the decisive comparison: beating random is necessary but not sufficient.
If probe-kNN cannot beat cluster-threshold adaptive refresh, its claim to be a distinct
method is in serious trouble regardless of how it compares to random.
"""
import json, os, re, sys, numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from e6_real import probe_knn_predict
from significance import build

K_NN, POW = 15, 8
corpus, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 30

docs = json.load(open(corpus, encoding="utf-8"))
C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
n = min(len(docs), len(C0), len(C1)); docs, C0, C1 = docs[:n], C0[:n], C1[:n]
m = np.linalg.norm(C1 - C0, axis=1)
edited = np.array([d["text0"] != d["text1"] for d in docs])
qs, gold = build(docs)
cache = "Qsig%s.npy" % tag
Q = np.load(cache) if os.path.exists(cache) and len(np.load(cache)) == len(qs) else None
if Q is None:
    from minilm_np import MiniLM
    mdl = MiniLM(); out = []
    for s in range(0, len(qs), 128):
        out.append(mdl.embed(qs[s:s+128], batch=32).astype(np.float32)); np.save(cache, np.vstack(out))
    Q = np.vstack(out)

def rec(C, k=10):
    h = 0
    for s in range(0, len(Q), 256):
        S = Q[s:s+256] @ C.T; top = np.argpartition(-S, k, axis=1)[:, :k]
        for r in range(top.shape[0]):
            if gold[s+r] in top[r]: h += 1
    return h/len(Q)

def mix(sel):
    C = C0.copy(); C[sel] = C1[sel]; return C

rs, rf = rec(C0), rec(C1); gain = rf - rs
print("%s n=%d changed=%.0f%% stale %.4f fresh %.4f headroom %.4f" % (name, n, 100*edited.mean(), rs, rf, gain))
npr = max(int(0.02*n), 20)
# cluster once per seed (KMeans is the expensive part)
res = {}
for budget in [0.10, 0.20, 0.35]:
    B = int(budget*n); pk, ct, cl = [], [], []
    for seed in range(SEEDS):
        rg = np.random.default_rng(9000+seed)
        pr = rg.choice(n, npr, replace=False)
        mh = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
        rest = np.setdiff1d(np.arange(n), pr)
        pk.append(rec(mix(np.concatenate([pr, rest[np.argsort(-mh[rest])[:max(B-npr,0)]]]))))
        km = KMeans(n_clusters=min(40, max(n//20,2)), n_init=3, random_state=seed).fit(C0)
        lab = km.labels_
        score = {c: (m[pr[lab[pr]==c]].mean() if (lab[pr]==c).any() else 0.0) for c in set(lab.tolist())}
        sel = []
        for c in sorted(score, key=lambda c: -score[c]):
            sel.extend(np.where(lab==c)[0].tolist())
            if len(sel) >= B: break
        ct.append(rec(mix(np.array(sel[:B], dtype=int))))
        ed = np.where(edited)[0]; rg.shuffle(ed); cl.append(rec(mix(ed[:B])))
    pk, ct, cl = map(np.array, (pk, ct, cl))
    t, p = stats.ttest_ind(pk, ct)
    print("  budget %3.0f%%: probe-kNN %.4f+-%.4f  cluster-thr %.4f+-%.4f  diff %+.4f  t=%+.2f p=%.4f  %s   [changelog %.4f]"
          % (100*budget, pk.mean(), pk.std(ddof=1), ct.mean(), ct.std(ddof=1),
             pk.mean()-ct.mean(), t, p,
             ("probe-kNN WINS" if p<0.05 and pk.mean()>ct.mean() else
              "cluster-thr WINS" if p<0.05 else "TIE (ns)"), cl.mean()))
    res["%.2f"%budget] = dict(pk=float(pk.mean()), pk_sd=float(pk.std(ddof=1)),
                              ct=float(ct.mean()), ct_sd=float(ct.std(ddof=1)),
                              changelog=float(cl.mean()), t=float(t), p=float(p),
                              winner=("pk" if p<0.05 and pk.mean()>ct.mean() else "ct" if p<0.05 else "tie"))
json.dump(dict(name=name, n=n, stale=rs, fresh=rf, headroom=gain, budgets=res),
          open("results/pk_vs_prior_%s.json"%name, "w"), indent=1)
print("wrote results/pk_vs_prior_%s.json"%name)
