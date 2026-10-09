"""
GATE: is staleness a retrieval SUCCESS on outdated content, worst for factual edits?

A FACT EDIT is defined with no reference to embeddings: a sentence in the revised
version that becomes identical to a sentence in the old version once every number is
masked. So the edit changed a number and nothing else in that sentence. The query is
the revised sentence with its numbers removed -- a question that does not contain its
own answer. Its gold document is known; the correct answer is the NEW number.

Three gates, each with a stated way to fail:
  G1 prevalence     -- fact edits must be a non-trivial share of revisions
  G2 retrieval      -- stale and fresh indexes must retrieve the document about equally
                       (if stale retrieves much less, it IS a retrieval failure)
  G3 invisibility   -- fact-edited documents must rank LOW under text diff and under
                       embedding displacement (if high, existing signals already work)
"""
import re, sys, numpy as np
from scipy import stats as st
from . import data as D, queries as Qm, measure as M
from drift.freshness import canonical, shingles, jaccard

NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
def mask(s): return re.sub(r"\s+", " ", NUM.sub("#", s)).strip()
def strip_nums(s): return re.sub(r"\s+", " ", NUM.sub("", s)).strip()

def fact_edits(docs):
    """(doc index, new sentence, old sentence) for every number-only sentence edit."""
    out = []
    for i, d in enumerate(docs):
        if canonical(d["text0"]) == canonical(d["text1"]): continue
        old = Qm.sentences(d["text0"]); new = Qm.sentences(d["text1"])
        old_set = set(old); by_mask = {}
        for s in old: by_mask.setdefault(mask(s), s)
        for s in new:
            if s in old_set or not NUM.search(s): continue
            o = by_mask.get(mask(s))
            if o is not None and o != s:
                out.append((i, s, o))
    return out

for corpus in ("arxiv_big", "python_docs", "arxiv"):
    docs = D.load_corpus(corpus); n = len(docs)
    revised = [i for i, d in enumerate(docs) if canonical(d["text0"]) != canonical(d["text1"])]
    fe = fact_edits(docs)
    fe_docs = sorted({i for i, _, _ in fe})
    print("=" * 84); print(f"{corpus}   n={n}  revised={len(revised)}"); print("=" * 84)
    print(f" G1 PREVALENCE: {len(fe)} number-only sentence edits in {len(fe_docs)} documents"
          f" = {len(fe_docs)/max(len(revised),1):.1%} of revised documents")
    for i, s, o in fe[:3]:
        a = NUM.findall(o); b = NUM.findall(s)
        print(f"     e.g. {[x for x in a if x not in b][:3]} -> {[x for x in b if x not in a][:3]} :: {strip_nums(s)[:70]}...")
    if len(fe) < 20:
        print("   too few fact edits to test G2/G3 on this corpus"); continue

    # G2: retrieval of the document for number-masked questions
    qtexts = [strip_nums(s) for _, s, _ in fe]; qgold = np.array([i for i, _, _ in fe])
    Qv = D.cached("minilm", corpus, "q_facts", lambda: qtexts)
    D0 = D.cached("minilm", corpus, "doc0", lambda: [d["text0"] for d in docs])
    D1 = D.cached("minilm", corpus, "doc1", lambda: [d["text1"] for d in docs])
    r0 = M.gold_ranks(Qv, D0, qgold); r1 = M.gold_ranks(Qv, D1, qgold)
    # contrast: the corpus's general new-text queries
    tn, gn = Qm.new_text(docs); Qn = D.cached("minilm", corpus, "q_new", lambda: tn)
    n0 = M.gold_ranks(Qn, D0, gn); n1 = M.gold_ranks(Qn, D1, gn)
    print(f"\n G2 RETRIEVAL   (document found in top-k, stale index vs fresh index)")
    print(f"     {'query type':<28}{'@1 stale':>9}{'@1 fresh':>9}{'@10 stale':>10}{'@10 fresh':>10}")
    for lab, a, b in [("fact questions (masked)", r0, r1), ("new-text queries (all)", n0, n1)]:
        print(f"     {lab:<28}{(a<=1).mean():>9.3f}{(b<=1).mean():>9.3f}"
              f"{(a<=10).mean():>10.3f}{(b<=10).mean():>10.3f}")
    disc = ((r0 <= 1) != (r1 <= 1)).sum()
    p = st.binomtest(int(((r1 <= 1) & ~(r0 <= 1)).sum()), disc, .5).pvalue if disc else 1.0
    print(f"     fact @1: stale finds the doc in {(r0<=1).mean():.1%} of cases, and every one of those "
          f"retrievals carries the OLD number   (stale vs fresh p={p:.3f})")

    # G3: where do fact-edited documents rank under the cheap and expensive signals?
    rv = np.array(revised)
    diff = np.array([1 - jaccard(shingles(canonical(docs[i]["text0"]), 5),
                                 shingles(canonical(docs[i]["text1"]), 5)) for i in rv])
    disp = np.linalg.norm(D1[rv] - D0[rv], axis=1)
    isf = np.isin(rv, fe_docs)
    # documents whose ONLY changes are number-only edits
    def only_numeric(i):
        o = set(Qm.sentences(docs[i]["text0"])); new = [s for s in Qm.sentences(docs[i]["text1"]) if s not in o]
        bm = {mask(s) for s in o}
        return len(new) > 0 and all(mask(s) in bm for s in new)
    pure = np.array([only_numeric(i) for i in rv])
    print(f"\n G3 INVISIBILITY   (percentile among revised docs; 100 = changed most)")
    for lab, sel in [("docs with any fact edit", isf), ("docs with ONLY fact edits", pure),
                     ("all other revised docs", ~isf)]:
        if sel.sum() == 0: print(f"     {lab:<28} none"); continue
        pd_ = st.rankdata(diff)[sel] / len(rv) * 100; pp = st.rankdata(disp)[sel] / len(rv) * 100
        top20 = (st.rankdata(-diff)[sel] <= 0.2 * len(rv)).mean()
        print(f"     {lab:<28} n={sel.sum():<5} text-diff pct {np.median(pd_):5.1f}   "
              f"displacement pct {np.median(pp):5.1f}   inside a 20% refresh budget {top20:.0%}")
    print()
