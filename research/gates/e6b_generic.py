"""Selective re-index bake-off on a real corpus.

usage: e6b_generic.py <corpus.json> <embedding_tag> <name>

Queries are sentence windows that exist only in the revised text (targeted), plus a
uniformly-sampled condition (untargeted) for the population-level effect.
Selectors: probe-kNN, random, oracle, change-log, cluster-threshold refresh.
"""
import csv, json, os, re, sys, numpy as np
from minilm_np import MiniLM
from e6_real import probe_knn_predict

RNG = np.random.default_rng(7)
SENT = re.compile(r"(?<=[.!?])\s+")
K_NN, POW = 15, 8

corpus, tag, name = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(corpus, encoding="utf-8"))
C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
n = min(len(docs), len(C0), len(C1))
docs, C0, C1 = docs[:n], C0[:n], C1[:n]
m = np.linalg.norm(C1 - C0, axis=1)
edited = np.array([d["text0"] != d["text1"] for d in docs])


def sentences(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip()) > 40]


def build(mode):
    qs, gold = [], []
    for i, d in enumerate(docs):
        s1 = sentences(d["text1"])
        if not s1:
            continue
        if mode == "targeted":
            s0 = set(sentences(d["text0"]))
            pool = [s for s in s1 if s not in s0]
        else:
            pool = s1
        if not pool:
            continue
        j = int(RNG.integers(0, len(pool)))
        w = " ".join(pool[j:j + 2])
        if len(w) >= 60:
            qs.append(w)
            gold.append(i)
    return qs, np.array(gold)


def recall_mrr(Q, C, gold, k=10):
    hit, rr = 0, 0.0
    for s in range(0, len(Q), 256):
        S = Q[s:s + 256] @ C.T
        top = np.argsort(-S, axis=1)[:, :k]
        for r in range(top.shape[0]):
            w = np.where(top[r] == gold[s + r])[0]
            if len(w):
                hit += 1
                rr += 1.0 / (w[0] + 1)
    return hit / max(len(Q), 1), rr / max(len(Q), 1)


def mixed(sel):
    C = C0.copy()
    C[sel] = C1[sel]
    return C


def cluster_threshold_select(B, nclus=40, seed=0):
    """Closest deployed prior art: cluster the stale index, re-embed the clusters whose
    probe-measured drift is highest, until the budget is spent."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(nclus, max(n // 20, 2)), n_init=4,
                random_state=seed).fit(C0)
    lab = km.labels_
    rg = np.random.default_rng(seed)
    probe = rg.choice(n, max(int(0.02 * n), 20), replace=False)
    score = {}
    for c in set(lab.tolist()):
        pc = probe[lab[probe] == c]
        score[c] = m[pc].mean() if len(pc) else 0.0
    order = sorted(score, key=lambda c: -score[c])
    sel = []
    for c in order:
        idx = np.where(lab == c)[0]
        sel.extend(idx.tolist())
        if len(sel) >= B:
            break
    return np.array(sel[:B], dtype=int)


model = None
rows = []
for mode in ["targeted", "untargeted"]:
    qs, gold = build(mode)
    if len(qs) < 30:
        print("%s: only %d queries, skipping" % (mode, len(qs)))
        continue
    cache = "Q%s_%s.npy" % (tag, mode)
    if os.path.exists(cache) and len(np.load(cache)) == len(qs):
        Q = np.load(cache)
    else:
        if model is None:
            model = MiniLM()
        out = []
        for s in range(0, len(qs), 128):
            out.append(model.embed(qs[s:s + 128], batch=32).astype(np.float32))
            np.save(cache, np.vstack(out))
        Q = np.vstack(out)

    rs, rf = recall_mrr(Q, C0, gold), recall_mrr(Q, C1, gold)
    gain = rf[0] - rs[0]
    print("\n" + "=" * 78)
    print("%s :: %s   %d queries" % (name.upper(), mode, len(qs)))
    print("  stale recall@10=%.4f mrr=%.4f | fresh recall@10=%.4f mrr=%.4f | headroom %.4f"
          % (rs[0], rs[1], rf[0], rf[1], gain))
    if gain < 1e-4:
        print("  no headroom, selective re-index is vacuous here")
        continue
    npr = max(int(0.02 * n), 20)
    print("  %7s %10s %9s %9s %10s %11s %9s"
          % ("budget", "probe-kNN", "random", "oracle", "changelog", "cluster-thr", "vs rand"))
    for budget in [0.05, 0.10, 0.20, 0.35, 0.50]:
        B = max(int(budget * n), 1)
        pk, rd, oc, cl, ct = [], [], [], [], []
        for seed in range(3):
            rg = np.random.default_rng(100 + seed)
            pr = rg.choice(n, npr, replace=False)
            mhat = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
            rest = np.setdiff1d(np.arange(n), pr)
            sel = np.concatenate([pr, rest[np.argsort(-mhat[rest])[:max(B - npr, 0)]]])
            pk.append(recall_mrr(Q, mixed(sel), gold)[0])
            rd.append(recall_mrr(Q, mixed(rg.choice(n, B, replace=False)), gold)[0])
            oc.append(recall_mrr(Q, mixed(np.argsort(-m)[:B]), gold)[0])
            ed = np.where(edited)[0]
            rg.shuffle(ed)
            cl.append(recall_mrr(Q, mixed(ed[:B]), gold)[0])
            try:
                ct.append(recall_mrr(Q, mixed(cluster_threshold_select(B, seed=seed)), gold)[0])
            except Exception:
                ct.append(float("nan"))
        pkm, rdm, ocm, clm, ctm = (np.mean(x) for x in (pk, rd, oc, cl, ct))
        sp, sr = (pkm - rs[0]) / gain, (rdm - rs[0]) / gain
        adv = sp / sr if abs(sr) > 1e-6 else float("nan")
        print("  %6.0f%% %10.4f %9.4f %9.4f %10.4f %11.4f %8.2fx"
              % (100 * budget, pkm, rdm, ocm, clm, ctm, adv))
        rows.append(dict(name=name, mode=mode, budget=budget, probe_knn=float(pkm),
                         random=float(rdm), oracle=float(ocm), changelog=float(clm),
                         cluster_thr=float(ctm), stale=rs[0], fresh=rf[0],
                         pk_share=float(sp), rnd_share=float(sr)))

if rows:
    with open("results/retrieval_%s.csv" % name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/retrieval_%s.csv" % name)
