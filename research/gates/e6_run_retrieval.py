"""Gate 2 on the REAL arXiv corpus: does selective re-indexing beat random?

Index   : 1786 arXiv abstracts.  stale = v1 (2023 H1), fresh = latest (Sep 2026).
Queries : sentence windows from the CURRENT abstract that are NEW since v1 - i.e. a
          user asking about content that only exists in the revised version.
Gold    : the paper the window came from.
Baselines: random, oracle, and change-log oracle (re-embed exactly what was edited).
"""
import json, os, re, sys, numpy as np
from minilm_np import MiniLM
from e6_real import probe_knn_predict

RNG = np.random.default_rng(7)
SENT = re.compile(r"(?<=[.!?])\s+")
K_NN, POW = 15, 8


def sentences(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip()) > 40]


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
    return hit / len(Q), rr / len(Q)


def mixed(C0, C1, sel):
    C = C0.copy()
    C[sel] = C1[sel]
    return C


docs = json.load(open("ax_corpus.json", encoding="utf-8"))
C0, C1 = np.load("C0_ax.npy"), np.load("C1_ax.npy")
n = len(docs)
m = np.linalg.norm(C1 - C0, axis=1)
edited = np.array([d["text0"] != d["text1"] for d in docs])

qs, gold = [], []
for i, d in enumerate(docs):
    s0 = set(sentences(d["text0"]))
    new = [s for s in sentences(d["text1"]) if s not in s0]
    if not new:
        continue
    j = int(RNG.integers(0, len(new)))
    win = " ".join(new[j:j + 2])
    if len(win) >= 60:
        qs.append(win)
        gold.append(i)
gold = np.array(gold)
print("targeted queries: %d over %d edited docs (of %d total)" % (len(qs), edited.sum(), n))

if os.path.exists("Q_ax.npy") and len(np.load("Q_ax.npy")) == len(qs):
    Q = np.load("Q_ax.npy")
else:
    model = MiniLM()
    out = []
    for s in range(0, len(qs), 128):
        out.append(model.embed(qs[s:s + 128], batch=32).astype(np.float32))
        np.save("Q_ax.npy", np.vstack(out))
        print("   queries %d/%d" % (len(np.vstack(out)), len(qs)), flush=True)
    Q = np.vstack(out)

r_stale = recall_mrr(Q, C0, gold)
r_fresh = recall_mrr(Q, C1, gold)
gain = r_fresh[0] - r_stale[0]
print("\n  stale index  recall@10=%.4f  mrr=%.4f" % r_stale)
print("  fresh index  recall@10=%.4f  mrr=%.4f" % r_fresh)
print("  headroom (full re-index gain) = %.4f" % gain)

npr = max(int(0.02 * n), 20)
print("\n  %7s %11s %9s %9s %11s %10s %10s"
      % ("budget", "probe-kNN", "random", "oracle", "changelog", "pk share", "advantage"))
rows = []
for budget in [0.05, 0.10, 0.20, 0.35, 0.50]:
    B = int(budget * n)
    pk, rd, oc, cl = [], [], [], []
    for seed in range(3):
        rg = np.random.default_rng(100 + seed)
        pr = rg.choice(n, npr, replace=False)
        mhat = probe_knn_predict(C0, m, pr, k=K_NN, power=POW)
        rest = np.setdiff1d(np.arange(n), pr)
        sel = np.concatenate([pr, rest[np.argsort(-mhat[rest])[:max(B - npr, 0)]]])
        pk.append(recall_mrr(Q, mixed(C0, C1, sel), gold)[0])
        rd.append(recall_mrr(Q, mixed(C0, C1, rg.choice(n, B, replace=False)), gold)[0])
        oc.append(recall_mrr(Q, mixed(C0, C1, np.argsort(-m)[:B]), gold)[0])
        ed = np.where(edited)[0]
        rg.shuffle(ed)
        cl.append(recall_mrr(Q, mixed(C0, C1, ed[:B]), gold)[0])
    pkm, rdm, ocm, clm = map(np.mean, (pk, rd, oc, cl))
    sp = (pkm - r_stale[0]) / gain if gain > 1e-9 else float("nan")
    sr = (rdm - r_stale[0]) / gain if gain > 1e-9 else float("nan")
    adv = sp / sr if sr > 1e-6 else float("nan")
    print("  %6.0f%% %11.4f %9.4f %9.4f %11.4f %11.3f %9.2fx"
          % (100 * budget, pkm, rdm, ocm, clm, sp, adv))
    rows.append(dict(budget=budget, probe_knn=float(pkm), random=float(rdm),
                     oracle=float(ocm), changelog=float(clm), pk_share=float(sp),
                     rnd_share=float(sr), stale=r_stale[0], fresh=r_fresh[0]))

import csv
with open("e6_retrieval_real.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote e6_retrieval_real.csv")
