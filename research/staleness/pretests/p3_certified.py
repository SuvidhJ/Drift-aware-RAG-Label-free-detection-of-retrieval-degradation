import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from research.staleness import data, facts, queries
rng = np.random.default_rng(2)
l2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
tok = data.backend("minilm").tok
ntok = lambda t: sum(tok.encode(t, max_len=100000)[1]) - 2
for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    edits = facts.aligned_edits(docs)
    pos = [ntok(docs[i]["text1"][:docs[i]["text1"].find(s)]) for i, s, o in edits]
    print(f"{corpus}: fact-edit sentences starting beyond token 510 (bge native window): {np.mean(np.array(pos) >= 510):.2f}")
    for model in ("minilm", "bge"):
        be = data.backend(model)
        D0 = l2(data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs])); D1 = l2(data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs]))
        # numbers-only swap displacement, per fact-edit doc (within window)
        dn = {}
        for i, s, o in edits:
            if o in docs[i]["text0"] and i not in dn:
                no, ns = facts.NUM.findall(o), facts.NUM.findall(s)
                if len(no) == len(ns):
                    it = iter(ns)
                    dn[i] = docs[i]["text0"].replace(o, facts.NUM.sub(lambda m: next(it), o), 1)
        ids = np.array(list(dn), dtype=int)
        E = l2(np.asarray(be.embed(list(dn.values()), batch_size=32)))
        dnum = dict(zip(ids, np.linalg.norm(E - D0[ids], axis=1)))
        # certified: for fact questions on those docs, margin of gold to the 10th/11th boundary > ||Δ_num||
        qs, gold, _ = facts.fact_questions(docs); gold = np.array(gold)
        Q = l2(data.cached(model, corpus, "q_answer_changed", lambda: facts.fact_questions(docs)[0]))
        keep = np.isin(gold, ids)
        S = Q[keep] @ D0.T; g = gold[keep]
        gs = S[np.arange(len(g)), g]; Sx = S.copy(); Sx[np.arange(len(g)), g] = -9
        srt = np.sort(Sx, 1)
        m10 = np.abs(gs - srt[:, -10]); m1 = gs - srt[:, -1]  # m1>0 means gold is rank 1
        bound = np.array([dnum[x] for x in g])
        print(f"  [{model}] n_docs {len(ids)}  ||Δ_num|| median {np.median(bound):.4f} max {bound.max():.4f}; "
              f"questions n={keep.sum()}: certified hit@10-invariant {np.mean(bound < m10):.2f}, "
              f"certified top-1 invariant {np.mean(bound < np.abs(m1)):.2f}")
        # an edit past the encoder window cannot move the vector: report it separately
        vis = {i: data.visible(model, docs[i]["text0"], t) for i, t in dn.items()}
        w = np.array([vis[x] for x in g])
        print(f"  [{model}]   past window {np.sum(~w)} of {len(w)} questions; inside n={w.sum()}: "
              f"||Δ_num|| median {np.median(bound[w]):.4f}, margin to 10th median {np.median(m10[w]):.4f}, "
              f"certified hit@10 {np.mean((bound < m10)[w]):.2f}, top-1 {np.mean((bound < np.abs(m1))[w]):.2f}")
        # dilution: mean signed cos(q, Δ) on retained-text queries, bootstrap CI (Δ = 0 masked, as in p4)
        Qr = l2(data.cached(model, corpus, "q_retained", lambda: queries.retained_text(docs)[0])); _, gr = queries.retained_text(docs); gr = np.array(gr)
        Dr = D1[gr] - D0[gr]; ok = np.linalg.norm(Dr, axis=1) > 1e-6
        c = (Qr[ok] * Dr[ok]).sum(1) / np.linalg.norm(Dr[ok], axis=1)
        Qn = l2(data.cached(model, corpus, "q_new", lambda: queries.new_text(docs)[0])); _, gn = queries.new_text(docs); gn = np.array(gn)
        Dn = D1[gn] - D0[gn]; ok = np.linalg.norm(Dn, axis=1) > 1e-6
        cn = (Qn[ok] * Dn[ok]).sum(1) / np.linalg.norm(Dn[ok], axis=1)
        bs = lambda x: np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)], [2.5, 97.5])
        print(f"  [{model}] signed cos(q,Δ): retained {c.mean():+.4f} {np.round(bs(c),4)}   new-text {cn.mean():+.4f} {np.round(bs(cn),4)}")
