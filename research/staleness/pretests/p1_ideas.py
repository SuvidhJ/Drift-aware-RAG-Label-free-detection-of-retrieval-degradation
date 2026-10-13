"""Pre-tests for journal-extension ideas. Exploratory; nothing here is a result yet."""
import re
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from research.staleness import answer_level, data, facts, measure, queries  # noqa: E402

K = 10
rng = np.random.default_rng(0)


def l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ranks_and_scores(Q, D, gold):
    S = Q @ D.T
    g = S[np.arange(len(gold)), gold]
    return 1 + (S > g[:, None]).sum(1), S, g


for model in ("minilm", "bge"):
    for corpus in ("arxiv_big", "python_docs"):
        docs = data.load_corpus(corpus)
        D0 = l2(data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs]))
        D1 = l2(data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs]))
        qs, gold, answers = facts.fact_questions(docs)
        gold = np.array(gold)
        Q = l2(data.cached(model, corpus, "q_answer_changed", lambda: facts.fact_questions(docs)[0]))
        cq, cg, ca = answer_level.control_questions(docs)
        cg = np.array(cg)
        CQ = l2(data.cached(model, corpus, "q_answer_control", lambda: answer_level.control_questions(docs)[0]))
        print(f"\n===== {model} / {corpus}  (changed n={len(gold)}, control n={len(cg)})")

        # ---- T2: per-question score change vs margin to the k-th competitor
        r0, S0, g0 = ranks_and_scores(Q, D0, gold)
        r1, S1, g1 = ranks_and_scores(Q, D1, gold)
        print(f"T2 rank identical: {np.mean(r0 == r1):.2f}   hit@10 flips: {np.mean((r0<=K)!=(r1<=K)):.3f}")
        dg = np.abs(g1 - g0)
        Sx = S0.copy(); Sx[np.arange(len(gold)), gold] = -9
        kth = np.sort(Sx, 1)[:, -K]
        margin = np.abs(g0 - kth)
        disp = np.linalg.norm(D1[gold] - D0[gold], axis=1)
        print(f"T2 gold |score change| median {np.median(dg):.4f}; margin to 10th median {np.median(margin):.4f};"
              f" doc displacement ||d1-d0|| median {np.median(disp):.4f}")
        print(f"T2 share with ||Δ|| < margin (bound certifies invariance, ignoring competitor moves): "
              f"{np.mean(disp < margin):.2f}")

        # ---- T3: stale vectors, fresh text
        has0 = np.array([bool({facts.norm(a) for a in A} & facts.values(docs[g]["text0"])) for g, A in zip(gold, answers)])
        has1 = np.array([bool({facts.norm(a) for a in A} & facts.values(docs[g]["text1"])) for g, A in zip(gold, answers)])
        print(f"T3 answer@10  stale {np.mean((r0<=K)&has0):.3f}  fresh {np.mean((r1<=K)&has1):.3f}"
              f"  HYBRID(stale vec, fresh text) {np.mean((r0<=K)&has1):.3f}")
        # selective re-embedding by text novelty (text-only signal) -> net hit@1 vs fresh
        res = measure.per_pool(model, corpus)
        w = res["weights"]
        rev = facts.revised(docs)
        nov = np.zeros(len(docs))
        from drift.freshness import shingles
        for i in rev:
            s0, s1 = shingles(docs[i]["text0"]), shingles(docs[i]["text1"])
            nov[i] = len(s1 - s0) / max(1, len(s1))
        pools_gold = {p: fn(docs)[1] for p, fn in queries.POOLS.items()}
        order = [i for i in np.argsort(-nov) if nov[i] > 0]
        print("T3 re-embed top-x% revised docs by novelty -> net hit@1 vs stale (fresh=all):")
        line = []
        for frac in (0, 0.1, 0.25, 0.5, 1.0):
            chosen = np.zeros(len(docs), bool); chosen[order[: int(frac * len(order))]] = True
            Dm = np.where(chosen[:, None], D1, D0)
            net = 0
            for p, r in res["pools"].items():
                Qp = l2(data.cached(model, corpus, measure.WHATS[p], lambda p=p: queries.POOLS[p](docs)[0]))
                rm = measure.gold_ranks(Qp, Dm, np.array(pools_gold[p]))
                net += w[p] * ((rm <= 1).mean() - (r["stale"] <= 1).mean())
            line.append(f"{frac:.0%}:{net:+.4f}")
        print("   ", "  ".join(line))

        # ---- T4: mixed index (both versions present)
        for lab, QQ, gg in (("changed", Q, gold), ("control", CQ, cg)):
            a = (QQ * D0[gg]).sum(1); b = (QQ * D1[gg]).sum(1)
            Dmix = np.vstack([D0, D1[rev]])
            S = QQ @ Dmix.T
            rk0 = 1 + (S > a[:, None]).sum(1); rk1 = 1 + (S > b[:, None]).sum(1)
            isrev = np.isin(gg, rev)
            print(f"T4 {lab:<8} P(stale version outranks fresh) {np.mean(a[isrev] > b[isrev]):.2f}"
                  f"  both in top-3 {np.mean(((rk0<=3)&(rk1<=3))[isrev]):.2f}"
                  f"  |score gap| median {np.median(np.abs(a-b)[isrev]):.4f}")


# ---- T1: controlled perturbations -- one number vs one word vs one capitalised name
print("\n===== T1 controlled single-token perturbations (doc level)")
NUMTOK = re.compile(r"\b\d+(?:\.\d+)?\b")
for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    texts = [d["text1"] for d in docs]
    vocab = sorted({w for t in texts[:2000] for w in re.findall(r"\b[a-z]{5,12}\b", t)})
    caps = sorted({w for t in texts[:2000] for w in re.findall(r"(?<=[a-z,] )[A-Z][a-z]{3,12}\b", t)})
    cand = [t for t in texts if NUMTOK.search(t) and re.search(r"(?<=[a-z,] )[A-Z][a-z]{3,12}\b", t)]
    pick = [cand[i] for i in rng.choice(len(cand), min(150, len(cand)), replace=False)]

    def num_sub(t):
        ms = list(NUMTOK.finditer(t)); m = ms[rng.integers(len(ms))]
        old = m.group()
        new = old
        while new == old:
            new = "".join(str(rng.integers(10)) if c.isdigit() else c for c in old)
            if new[0] == "0" and len(new) > 1 and new[1] != ".":
                new = str(rng.integers(1, 10)) + new[1:]
        return t[: m.start()] + new + t[m.end():]

    def word_sub(t):
        ms = list(re.finditer(r"\b[a-z]{5,12}\b", t)); m = ms[rng.integers(len(ms))]
        return t[: m.start()] + vocab[rng.integers(len(vocab))] + t[m.end():]

    def name_sub(t):
        ms = list(re.finditer(r"(?<=[a-z,] )[A-Z][a-z]{3,12}\b", t)); m = ms[rng.integers(len(ms))]
        return t[: m.start()] + caps[rng.integers(len(caps))] + t[m.end():]

    for model in ("minilm", "bge"):
        be = data.backend(model)
        E = l2(np.asarray(be.embed(pick, batch_size=32)))
        out = []
        for lab, fn in (("number", num_sub), ("word", word_sub), ("name", name_sub)):
            pert = [fn(t) for t in pick]
            P = l2(np.asarray(be.embed(pert, batch_size=32)))
            d = 1 - (E * P).sum(1)
            # a substitution past the encoder window cannot move the vector: report inside-window separately
            v = np.array([data.visible(model, t, pt) for t, pt in zip(pick, pert)])
            out.append(f"{lab}: median 1-cos {np.median(d):.5f} (IQR {np.percentile(d,25):.5f}-{np.percentile(d,75):.5f}),"
                       f" past window {np.mean(~v):.2f}, inside median {np.median(d[v]):.5f}")
        print(f"{model}/{corpus} n={len(pick)}  " + " | ".join(out), flush=True)
