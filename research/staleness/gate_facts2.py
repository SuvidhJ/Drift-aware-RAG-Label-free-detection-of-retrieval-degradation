"""
Prevalence under three definitions of 'a fact changed', reported side by side.

  A  strict   -- sentence identical to an old sentence once numbers are masked
                 (pre-registered; this is the one G1 failed on)
  B  aligned  -- sentence's closest old counterpart (shingle Jaccard >= 0.5, so the
                 same sentence lightly reworded) carries a DIFFERENT set of numbers
  C  any      -- the document's multiset of numbers changed at all (loosest; also
                 counts newly added sentences that happen to contain numbers)

B is the definition I would defend: it is still objective and embedding-free, and it
admits real factual updates made inside a reworded sentence. It is chosen AFTER seeing
A fail, so both are reported, and G2/G3 are re-run on B to check the mechanism holds.
"""
import re, collections, numpy as np
from scipy import stats as st
from . import data as D, queries as Qm, measure as M
from drift.freshness import canonical, shingles, jaccard
NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
def nums(s): return collections.Counter(NUM.findall(s))
def mask(s): return re.sub(r"\s+"," ",NUM.sub("#",s)).strip()
def strip_nums(s): return re.sub(r"\s+"," ",NUM.sub("",s)).strip()

def edits_B(docs, revised):
    out=[]
    for i in revised:
        old=Qm.sentences(docs[i]["text0"]); oset=set(old)
        osh=[(o,shingles(canonical(o),5)) for o in old]
        for s in Qm.sentences(docs[i]["text1"]):
            if s in oset or not NUM.search(s): continue
            ss=shingles(canonical(s),5)
            best=max(osh,key=lambda t: jaccard(ss,t[1]),default=None)
            if best and jaccard(ss,best[1])>=0.5 and nums(s)!=nums(best[0]):
                out.append((i,s,best[0]))
    return out

for corpus in ("arxiv_big","python_docs"):
    docs=D.load_corpus(corpus)
    rev=[i for i,d in enumerate(docs) if canonical(d["text0"])!=canonical(d["text1"])]
    A=set()
    for i in rev:
        o=Qm.sentences(docs[i]["text0"]); bm={}
        for s in o: bm.setdefault(mask(s),s)
        for s in Qm.sentences(docs[i]["text1"]):
            if s not in set(o) and NUM.search(s) and bm.get(mask(s)) not in (None,s): A.add(i)
    eb=edits_B(docs,rev); B=sorted({i for i,_,_ in eb})
    C={i for i in rev if nums(docs[i]["text0"])!=nums(docs[i]["text1"])}
    print("="*82); print(f"{corpus}   revised={len(rev)}"); print("="*82)
    print(f"  PREVALENCE (share of revised documents)")
    for lab,s in [("A strict (pre-registered)",A),("B aligned sentence",B),("C any number changed",C)]:
        print(f"    {lab:<28} {len(s):>5} docs  {len(s)/len(rev):6.1%}")

    # G2 + G3 on definition B
    qt=[strip_nums(s) for _,s,_ in eb]; qg=np.array([i for i,_,_ in eb])
    Qv=D.cached("minilm",corpus,"q_factsB",lambda: qt)
    D0=D.cached("minilm",corpus,"doc0",lambda:[d["text0"] for d in docs])
    D1=D.cached("minilm",corpus,"doc1",lambda:[d["text1"] for d in docs])
    r0=M.gold_ranks(Qv,D0,qg); r1=M.gold_ranks(Qv,D1,qg)
    disc=((r0<=1)!=(r1<=1)).sum()
    p=st.binomtest(int(((r1<=1)&~(r0<=1)).sum()),disc,.5).pvalue if disc else 1.0
    print(f"\n  G2 on B  ({len(qt)} fact questions)   @1 stale {(r0<=1).mean():.3f} fresh {(r1<=1).mean():.3f}"
          f"   @10 stale {(r0<=10).mean():.3f} fresh {(r1<=10).mean():.3f}   p={p:.3f}")
    rv=np.array(rev)
    diff=np.array([1-jaccard(shingles(canonical(docs[i]["text0"]),5),shingles(canonical(docs[i]["text1"]),5)) for i in rv])
    disp=np.linalg.norm(D1[rv]-D0[rv],axis=1); isB=np.isin(rv,B)
    budget=(st.rankdata(-diff)<=0.2*len(rv))
    print(f"  G3 on B  fact-edited docs: text-diff pct {np.median(st.rankdata(diff)[isB]/len(rv)*100):.1f}  "
          f"displacement pct {np.median(st.rankdata(disp)[isB]/len(rv)*100):.1f}  "
          f"inside 20% budget {budget[isB].mean():.0%}  (others {budget[~isB].mean():.0%})")
    print()
