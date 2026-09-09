"""The blind spot on REAL revisions, plus published detection baselines.

FINDINGS sec.6 showed on a simulator that corpus-side distributional distance is a 33x
unreliable predictor of retrieval damage.  That is Part 1's headline and it has never
been shown on real data.  Here it is tested on real revisions.

Method: split the genuinely-revised documents by whether the revision kept the document
in its own neighbourhood (IN-PLACE) or moved it out (MIGRATING), using rank of the old
position among neighbours of the new position.  Then, for matched numbers of revised
documents, compare (a) retrieval damage and (b) corpus-side distributional distance.
If damage is similar while distance differs greatly, the blind spot is real.

Also runs MMD and Page-Hinkley alongside sliced-Wasserstein so Part 1's negative result
is about distributional drift detection generally, not one estimator.
"""
import glob, json, os, re, sys, numpy as np
from gate1_real import sliced_w1, mmd_rbf, centroid_shift

RNG = np.random.default_rng(11)
SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(t):
    return [s.strip() for s in SENT.split(t) if len(s.strip()) > 40]


def recall_at(Q, C, gold, k=10):
    hit = 0
    for s in range(0, len(Q), 256):
        S = Q[s:s + 256] @ C.T
        top = np.argpartition(-S, k, axis=1)[:, :k]
        for r in range(top.shape[0]):
            if gold[s + r] in top[r]:
                hit += 1
    return hit / max(len(Q), 1)


def page_hinkley(stream, delta=0.005, lam=0.05):
    """Classic streaming change detector. Returns index of first alarm, or -1."""
    mean, mt, Mt = 0.0, 0.0, 0.0
    for i, x in enumerate(stream):
        mean += (x - mean) / (i + 1)
        mt += x - mean - delta
        Mt = min(Mt, mt)
        if mt - Mt > lam:
            return i
    return -1


def analyse(corpus, tag, name):
    docs = json.load(open(corpus, encoding="utf-8"))
    C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
    n = min(len(docs), len(C0), len(C1))
    docs, C0, C1 = docs[:n], C0[:n], C1[:n]
    edited = np.array([d["text0"] != d["text1"] for d in docs])
    if edited.sum() < 60:
        print("  %s: only %d revised docs, skipping" % (name, edited.sum()))
        return None

    # how far did each revision move the document relative to its own neighbourhood?
    # rank of the OLD vector among the corpus, as seen from the NEW vector
    ranks = np.zeros(n)
    for s in range(0, n, 512):
        S = C1[s:s + 512] @ C0.T
        for r in range(S.shape[0]):
            i = s + r
            ranks[i] = int((S[r] > S[r, i]).sum())
    ed = np.where(edited)[0]
    med = np.median(ranks[ed])
    inplace = ed[ranks[ed] <= med]
    migrate = ed[ranks[ed] > med]
    k = min(len(inplace), len(migrate))
    inplace, migrate = inplace[:k], migrate[:k]
    print("\n  %s: %d revised; %d in-place vs %d migrating (rank split at %.0f)"
          % (name, len(ed), len(inplace), len(migrate), med))

    rows = {}
    for label, idx in (("in-place", inplace), ("migrating", migrate)):
        # corpus-side distributional distance when ONLY this subset is refreshed
        Cs = C0.copy()
        Cs[idx] = C1[idx]
        mean_, max_ = sliced_w1(C0, Cs, seed=3)
        mmd = mmd_rbf(C0[:800], Cs[:800])
        cen = centroid_shift(C0, Cs)

        # retrieval damage on queries targeting exactly those documents
        qs, gold = [], []
        for i in idx:
            s0 = set(sentences(docs[i]["text0"]))
            new = [s for s in sentences(docs[i]["text1"]) if s not in s0]
            if new:
                qs.append(" ".join(new[:2]))
                gold.append(i)
        if len(qs) < 20:
            print("    %s: too few queries (%d)" % (label, len(qs)))
            continue
        cache = "Qbs%s_%s.npy" % (tag, label)
        if os.path.exists(cache) and len(np.load(cache)) == len(qs):
            Q = np.load(cache)
        else:
            from minilm_np import MiniLM
            mdl = MiniLM()
            out = []
            for s in range(0, len(qs), 128):
                out.append(mdl.embed(qs[s:s + 128], batch=32).astype(np.float32))
                np.save(cache, np.vstack(out))
            Q = np.vstack(out)
        gold = np.array(gold)
        r_stale = recall_at(Q, C0, gold)
        r_fresh = recall_at(Q, C1, gold)
        dmg = r_fresh - r_stale
        rows[label] = dict(n=len(idx), nq=len(qs), recall_stale=r_stale, recall_fresh=r_fresh,
                           damage=dmg, sw1_mean=mean_, sw1_max=max_, mmd=mmd, centroid=cen)
        print("    %-10s damage %.4f (stale %.4f -> fresh %.4f) | sw1_max %.5f  mmd %.6f  centroid %.5f"
              % (label, dmg, r_stale, r_fresh, max_, mmd, cen))

    if len(rows) == 2:
        a, b = rows["in-place"], rows["migrating"]
        dr = a["damage"] / max(b["damage"], 1e-9)
        sr = b["sw1_max"] / max(a["sw1_max"], 1e-12)
        print("    ==> damage ratio in-place/migrating = %.3f   "
              "sliced-W1 signal ratio migrating/in-place = %.1fx" % (dr, sr))
        rows["damage_ratio"] = float(dr)
        rows["signal_ratio"] = float(sr)
    return rows


if __name__ == "__main__":
    print("=" * 78)
    print("BLIND SPOT ON REAL REVISIONS + published detection baselines")
    print("=" * 78)
    out = {}
    for corpus, tag, name in [("pd_corpus.json", "_pd", "pd"),
                              ("wp_corpus.json", "_wp", "wp"),
                              ("ax_corpus_big.json", "_axbig", "ax"),
                              ("ax_corpus.json", "_ax", "ax_small")]:
        if not (os.path.exists(corpus) and os.path.exists("C0%s.npy" % tag)):
            continue
        try:
            r = analyse(corpus, tag, name)
            if r:
                out[name] = r
        except Exception as e:
            print("  %s failed: %r" % (name, e))

    # Page-Hinkley on a real drifting score stream, for the detector comparison
    print("\n--- Page-Hinkley on a real score stream ---")
    for corpus, tag, name in [("pd_corpus.json", "_pd", "pd"),
                              ("ax_corpus_big.json", "_axbig", "ax")]:
        if not os.path.exists("C0%s.npy" % tag):
            continue
        C0, C1 = np.load("C0%s.npy" % tag), np.load("C1%s.npy" % tag)
        n = min(len(C0), len(C1))
        ref = C0[:n // 2]
        stream = ([float(np.max(C0[i] @ ref.T)) for i in range(n // 2, n)]
                  + [float(np.max(C1[i] @ ref.T)) for i in range(n // 2, n)])
        idx = page_hinkley(stream)
        print("  %-4s stream len %d, true change at %d, PH alarm at %s"
              % (name, len(stream), n - n // 2, idx))
        out.setdefault("page_hinkley", {})[name] = dict(alarm=idx, change_at=n - n // 2,
                                                        length=len(stream))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/baselines.json", "w"), indent=1)
    print("\nwrote results/baselines.json")
