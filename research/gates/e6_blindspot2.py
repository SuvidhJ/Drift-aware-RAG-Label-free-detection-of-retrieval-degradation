"""Well-powered real-data test of the blind spot (FINDINGS sec.6).

The simulator claim: corpus-side distributional distance is a poor predictor of
retrieval damage.  The binary in-place/migrating split fails on real data because only
~2% of real revisions move a document out of its own neighbourhood (n=35).

Direct, per-document test instead, using every revised document:
  (a) displacement magnitude m_i  - exactly what drives corpus-side W1/MMD/centroid
  (b) actual retrieval damage     - stale index misses the doc, fresh index finds it
How well does (a) predict (b)?  Weak prediction = the blind spot, well powered.
"""
import json, os, re, sys, numpy as np
from scipy import stats
from significance import build

SENT = re.compile(r"(?<=[.!?])\s+")
for corpus, tag, name in [("ax_corpus_big.json", "_axbig", "ax"),
                          ("pd_corpus.json", "_pd", "pd")]:
    if not os.path.exists(corpus): continue
    docs = json.load(open(corpus, encoding="utf-8"))
    C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
    n = min(len(docs), len(C0), len(C1)); docs, C0, C1 = docs[:n], C0[:n], C1[:n]
    m = np.linalg.norm(C1 - C0, axis=1)
    qs, gold = build(docs)
    cache = "Qsig%s.npy" % tag
    if not (os.path.exists(cache) and len(np.load(cache)) == len(qs)):
        print("%s: no query cache, skip" % name); continue
    Q = np.load(cache)

    def hits(C, k=10):
        out = np.zeros(len(Q), dtype=bool)
        for s in range(0, len(Q), 256):
            S = Q[s:s+256] @ C.T; top = np.argpartition(-S, k, axis=1)[:, :k]
            for r in range(top.shape[0]):
                out[s+r] = gold[s+r] in top[r]
        return out
    h0, h1 = hits(C0), hits(C1)
    dmg = (h1 & ~h0)                      # fresh finds it, stale does not = real damage
    mq = m[gold]
    print("\n" + "="*78)
    print("BLIND SPOT (per-document, well powered) :: %s   n=%d queries=%d" % (name, n, len(Q)))
    print("  stale hit %.4f  fresh hit %.4f  damaged %d (%.2f%%)"
          % (h0.mean(), h1.mean(), dmg.sum(), 100*dmg.mean()))
    if dmg.sum() < 10:
        print("  too few damaged docs"); continue
    auc = stats.mannwhitneyu(mq[dmg], mq[~dmg]).statistic / (dmg.sum() * (~dmg).sum())
    r_pb = stats.pointbiserialr(dmg.astype(int), mq)
    print("  displacement m of damaged docs   : mean %.4f  median %.4f" % (mq[dmg].mean(), np.median(mq[dmg])))
    print("  displacement m of undamaged docs : mean %.4f  median %.4f" % (mq[~dmg].mean(), np.median(mq[~dmg])))
    print("  AUC of m as a predictor of damage: %.3f   (0.5 = no information)" % auc)
    print("  point-biserial r = %+.3f  p = %.2g" % (r_pb.statistic, r_pb.pvalue))
    # decile table: does high displacement mean high damage?
    q = np.quantile(mq, np.linspace(0, 1, 6))
    print("  %-22s %8s %10s" % ("displacement quintile", "n", "damage rate"))
    for i in range(5):
        sel = (mq >= q[i]) & ((mq <= q[i+1]) if i == 4 else (mq < q[i+1]))
        if sel.sum(): print("  [%.4f, %.4f)%8d %9.3f" % (q[i], q[i+1], sel.sum(), dmg[sel].mean()))
    # how much corpus-side signal do the damaging docs actually generate?
    dmg_idx = np.unique(gold[dmg]); und_idx = np.unique(gold[~dmg & h1])
    k = min(len(dmg_idx), len(und_idx))
    if k > 20:
        rng = np.random.default_rng(0)
        a, b = rng.choice(dmg_idx, k, False), rng.choice(und_idx, k, False)
        cs_d = np.linalg.norm(C1[a].mean(0) - C0[a].mean(0))
        cs_u = np.linalg.norm(C1[b].mean(0) - C0[b].mean(0))
        print("  matched-size centroid shift: damaging %.5f vs non-damaging %.5f  ratio %.2fx"
              % (cs_d, cs_u, cs_d / max(cs_u, 1e-12)))
