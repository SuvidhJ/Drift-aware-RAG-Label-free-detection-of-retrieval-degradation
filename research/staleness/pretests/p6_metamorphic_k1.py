"""Pre-test 6: do real updates expose more faults than synthetic counterfactual mutation?

Context: "When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations"
(arXiv 2607.26843) injects LLM-generated counterfactuals; its worst operator violates the
variance relation in 27.6% of cases, at top-k = 1.  Here the update is real: a document's
revision, indexed the way frameworks do by default (both versions kept, `append`) or
replaced in place (`replace`).  With k = 1 the generator sees one chunk, so if that chunk
holds the replaced value and not the new one, the variance relation ("the answer follows
the update") is violated whatever the generator does: a deterministic lower bound, no LLM.

Pre-registered 2026-10-10: kill if, under `append` at k = 1, the stale-only rate is below
0.276 in every one of the 4 model x corpus settings.

    .venv/bin/python research/staleness/pretests/p6_metamorphic_k1.py
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as P  # noqa: E402  (same chunks, same cache)
from research.staleness import data, facts  # noqa: E402


def questions(docs):
    """Changed-fact questions with new AND old (replaced) answer values, deduplicated as in facts."""
    out, seen = [], set()
    for i, s, o in facts.aligned_edits(docs):
        new = {facts.norm(x) for x in facts.NUM.findall(s) if facts.norm(x) not in facts.values(o) and len(x) >= facts.MIN_ANSWER}
        old = {facts.norm(x) for x in facts.NUM.findall(o) if facts.norm(x) not in facts.values(s) and len(x) >= facts.MIN_ANSWER}
        q = facts.strip_nums(s)
        if new and (q, i) not in seen:
            seen.add((q, i))
            out.append((q, i, new, old))
    return out


for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    cl = P.classify(docs)
    uniq = sorted({c for c0, c1, _ in cl for c in c0 + c1})
    idx = {c: i for i, c in enumerate(uniq)}
    qs = questions(docs)
    has_old = np.array([bool(o) for _, _, _, o in qs])
    for model in ("minilm", "bge"):
        E = data.cached(model, corpus, f"chunks{P.CHUNK}", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(model, corpus, "q_answer_changed", lambda: [q for q, _, _, _ in qs])
        for pol in ("replace", "append"):
            items = []
            for d, (c0, c1, lab) in enumerate(cl):
                if pol == "append":
                    items += [(c, d, 0 if c not in set(c1) else 1) for c in c0]
                    items += [(c, d, 1) for c, (k, _, _) in zip(c1, lab) if k != "identical"]
                else:
                    items += [(c, d, 1) for c in c1]
            V = E[[idx[c] for c, _, _ in items]]
            S = Q @ V.T
            row = []
            for k in (1, 3):
                top = np.argsort(-S, axis=1)[:, :k]
                corr = np.zeros(len(qs), bool); stale = np.zeros(len(qs), bool); stale_v0 = np.zeros(len(qs), bool)
                for n, ((_, g, new, old), tt) in enumerate(zip(qs, top)):
                    gt = [t for t in tt if items[t][1] == g]
                    vals = set().union(*(facts.values(items[t][0]) for t in gt)) if gt else set()
                    old_v0 = set().union(*(facts.values(items[t][0]) for t in gt if items[t][2] == 0)) if gt else set()
                    corr[n] = bool(vals & new); stale[n] = bool(vals & old); stale_v0[n] = bool(old_v0 & old)
                only_stale = (stale & ~corr)[has_old].mean()
                # strict: the replaced value comes from a chunk that exists only in the OLD version
                strict = (stale_v0 & ~corr)[has_old]
                row.append(f"k={k}: correct {corr.mean():.3f}  stale-only {only_stale:.3f} (old-version chunk {strict.mean():.3f}, "
                           f"{int(strict.sum())}/{int(has_old.sum())})  both {(stale & corr)[has_old].mean():.3f}")
            print(f"{model}/{corpus} {pol:<8} n={len(qs)} (with a replaced value {has_old.sum()})   " + "   ".join(row), flush=True)
