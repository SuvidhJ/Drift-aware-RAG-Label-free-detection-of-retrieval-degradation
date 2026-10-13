"""Pre-test 2: truncation confound, random-projection model, fact-edit share of displacement."""
import sys

import numpy as np
from scipy.stats import norm as N

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from research.staleness import data, facts, queries  # noqa: E402

rng = np.random.default_rng(1)
MAXLEN = 256


def l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def ntok(tok, text):
    return sum(tok.encode(text, max_len=100000)[1]) - 2  # mask counts real tokens; minus CLS/SEP


def eff_dim(X):
    X = X - X.mean(0)
    ev = np.linalg.eigvalsh(X.T @ X / len(X))
    return ev.sum() ** 2 / (ev ** 2).sum()


for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    tok = data.backend("minilm").tok
    edits = facts.aligned_edits(docs)
    # where does each fact-edit sentence start, in tokens, in the revised document?
    beyond = []
    for i, s, o in edits:
        pos = docs[i]["text1"].find(s)
        beyond.append(ntok(tok, docs[i]["text1"][:pos]) >= MAXLEN - 2)
    ndoc = len({i for i, _, _ in edits})
    doc_beyond = {}
    for (i, _, _), b in zip(edits, beyond):
        doc_beyond[i] = doc_beyond.get(i, True) and b
    print(f"\n### {corpus}: {len(edits)} fact-edit sentences in {ndoc} docs; "
          f"sentence starts beyond token {MAXLEN}: {np.mean(beyond):.2f}; "
          f"docs whose EVERY fact edit is beyond: {np.mean(list(doc_beyond.values())):.2f}")

    for model in ("minilm", "bge"):
        be = data.backend(model)
        D0 = l2(data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs]))
        D1 = l2(data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs]))
        print(f"  [{model}] effective dim of doc embeddings: {eff_dim(D1):.1f} (nominal {D1.shape[1]})")

        # --- fact-edit-only displacement: text0 with ONLY the edited sentence swapped in
        sel = [(i, s, o) for (i, s, o), b in zip(edits, beyond) if not b and o in docs[i]["text0"]]
        sel = [sel[j] for j in rng.choice(len(sel), min(120, len(sel)), replace=False)]
        only = l2(np.asarray(be.embed([docs[i]["text0"].replace(o, s, 1) for i, s, o in sel], batch_size=32)))
        idx = np.array([i for i, _, _ in sel], dtype=int)
        d_fact = np.linalg.norm(only - D0[idx], axis=1)
        d_full = np.linalg.norm(D1[idx] - D0[idx], axis=1)
        print(f"  [{model}] within-window fact edits n={len(sel)}: ||Δ_fact|| median {np.median(d_fact):.4f}, "
              f"||Δ_revision|| median {np.median(d_full):.4f}, ratio median {np.median(d_fact/np.maximum(d_full,1e-9)):.2f}")
        # the window test above uses text1 positions but the swap edits text0: check the token ids
        vs = np.array([data.visible(model, docs[i]["text0"], docs[i]["text0"].replace(o, s, 1)) for i, s, o in sel])
        print(f"  [{model}]   sentence swap past window {np.mean(~vs):.2f}; inside n={vs.sum()}: "
              f"ratio median {np.median((d_fact/np.maximum(d_full,1e-9))[vs]):.2f}")
        # strict-number-only part: replace only the numbers in the old sentence
        strict = []
        for i, s, o in sel:
            nums_o, nums_s = facts.NUM.findall(o), facts.NUM.findall(s)
            if len(nums_o) == len(nums_s):
                it = iter(nums_s)
                strict.append((i, docs[i]["text0"].replace(o, facts.NUM.sub(lambda m: next(it), o), 1)))
        if strict:
            E = l2(np.asarray(be.embed([t for _, t in strict], batch_size=32)))
            ii = np.array([i for i, _ in strict], dtype=int)
            d_num = np.linalg.norm(E - D0[ii], axis=1)
            print(f"  [{model}] numbers-only swap n={len(strict)}: ||Δ_num|| median {np.median(d_num):.4f}")
            vn = np.array([data.visible(model, docs[i]["text0"], t) for i, t in strict])
            ratio = np.linalg.norm(D1[ii] - D0[ii], axis=1) / np.maximum(d_num, 1e-12)
            print(f"  [{model}]   numbers-only past window {np.mean(~vn):.2f}; inside n={vn.sum()}: "
                  f"||Δ_num|| median {np.median(d_num[vn]):.4f}, revision/number per-doc median {np.median(ratio[vn]):.1f}x")

        # --- random-projection model on the changed-fact questions
        qs, gold, _ = facts.fact_questions(docs)
        gold = np.array(gold)
        Q = l2(data.cached(model, corpus, "q_answer_changed", lambda: facts.fact_questions(docs)[0]))
        Dl = D1[gold] - D0[gold]
        nD = np.linalg.norm(Dl, axis=1)
        nz = nD > 1e-6                                          # Δ = 0: the edit is past the window
        cosqd = (Q[nz] * Dl[nz]).sum(1) / nD[nz]
        de = eff_dim(D1)
        print(f"  [{model}] Δ=0 for {np.mean(~nz):.2f}; |cos(q, Δ)| median {np.median(np.abs(cosqd)):.3f} (Δ≠0); "
              f"random-direction median 0.674/sqrt(d_eff) = {0.674/np.sqrt(de):.3f}, "
              f"nominal-d = {0.674/np.sqrt(D1.shape[1]):.3f}; mean signed cos {cosqd.mean():+.3f}")
        # all-revised-doc check with retained-text queries (no construction bias)
        Qr = l2(data.cached(model, corpus, "q_retained", lambda: queries.retained_text(docs)[0]))
        _, gr = queries.retained_text(docs)
        gr = np.array(gr)
        Dr = D1[gr] - D0[gr]
        cr = (Qr * Dr).sum(1) / np.maximum(np.linalg.norm(Dr, axis=1), 1e-12)
        print(f"  [{model}] retained-text queries n={len(gr)}: |cos(q,Δ)| median {np.median(np.abs(cr)):.3f}, "
              f"mean signed {cr.mean():+.3f}")
        # predicted vs observed hit@10 flips for retained-text queries (competitors held fixed: stale index + one doc refreshed)
        S0 = Qr @ D0.T
        g0 = S0[np.arange(len(gr)), gr]
        Sx = S0.copy(); Sx[np.arange(len(gr)), gr] = -9
        kth = np.sort(Sx, 1)[:, -10]
        margin = g0 - kth                                     # >0: gold in top 10
        g1 = (Qr * D1[gr]).sum(1)
        obs = ((g0 > kth) != (g1 > kth)).mean()
        sigma = np.linalg.norm(Dr, axis=1) / np.sqrt(de)       # model: q·Δ ~ N(0, ||Δ||²/d_eff)
        # one-sided: must cross toward the boundary; Δ = 0 cannot flip, so it predicts 0
        pred = np.where(sigma > 0, N.sf(np.abs(margin) / np.maximum(sigma, 1e-300)), 0.0).mean()
        print(f"  [{model}] hit@10 flips when ONLY the gold doc is refreshed: observed {obs:.4f}, "
              f"random-projection prediction {pred:.4f}")
