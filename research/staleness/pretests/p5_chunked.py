"""Pre-test 5: which edits need re-embedding? Chunked keep-vectors vs re-embedding.

Pre-registered 2026-10-10, before any chunk was embedded.  Chunks: sentence-aligned,
content-defined boundaries (stable under insertions, so fair to diff re-embedding),
at most CHUNK wordpieces (MiniLM tokenizer, shared by both models so the chunks are
identical).  Each chunk of a revised document's new version is

  identical  its text is one of the old version's chunks
  edited     its closest old chunk (char 5-gram Jaccard) is >= THETA but not identical
  new        no old chunk reaches THETA

Index policies (all over the whole corpus; unrevised documents are the same in each):

  stale      old chunks, old vectors                              (never update)
  full       new chunks, every chunk of a revised doc re-embedded  (re-embed revised docs)
  diff       new chunks, only edited + new chunks embedded         (SmartVector / LiveVectorLake)
             -> same vectors as `full`, so same accuracy; only the cost differs
  keep       new chunks; edited chunks keep their old chunk's vector, only new chunks are embedded
  append     old chunks plus every non-identical new chunk         (both versions indexed)

Kill criteria (all four model x corpus settings, CHUNK = 128):
  K1  keep answer@10 on changed-fact questions >= full - 0.02
  K2  keep doc-level hit@10 on the new-text and retained-text pools >= full - 0.01
  K3  keep embeds <= 70% of the chunks diff embeds
  Go if some THETA in {0.3, 0.5, 0.7} passes K1-K3 everywhere; the whole curve is reported.

    .venv/bin/python research/staleness/pretests/p5_chunked.py stats   # text only (K3)
    .venv/bin/python research/staleness/pretests/p5_chunked.py run [chunk]   # embeds chunks
"""
import hashlib
import re
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from drift.freshness import canonical, jaccard, shingles  # noqa: E402
from embeddings.backends import WordPieceTokenizer  # noqa: E402
from research.staleness import data, facts, measure, queries  # noqa: E402

CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 128
THETAS = (0.3, 0.5, 0.7)
CDC_MOD = 3
K = 10
SPLIT = re.compile(r"(?<=[.!?])\s+")
TOK = WordPieceTokenizer(data.model_dir("minilm") / "vocab.txt")
_ntok: dict[str, int] = {}


def ntok(s: str) -> int:
    if s not in _ntok:
        _ntok[s] = sum(len(TOK.wordpiece(w)) for w in TOK.basic_tokenize(s))
    return _ntok[s]


def chunks(text: str, limit: int = CHUNK) -> list[str]:
    """Greedy sentence packing into <= limit - 2 wordpieces; long sentences split by words."""
    units = []
    for s in SPLIT.split(text.strip()):
        if not s:
            continue
        if ntok(s) <= limit - 2:
            units.append(s)
            continue
        cur = []
        for w in s.split():
            if cur and ntok(" ".join(cur + [w])) > limit - 2:
                units.append(" ".join(cur)); cur = []
            cur.append(w)
        if cur:
            units.append(" ".join(cur))
    # Content-defined boundaries (as in dedup systems): a chunk also ends after any
    # sentence whose hash is 0 mod CDC_MOD.  An inserted or deleted sentence then only
    # disturbs nearby chunks instead of shifting every later boundary -- the best case
    # for diff-based re-embedding, so the comparison is fair to that baseline.
    out, cur = [], []
    for u in units:
        if cur and ntok(" ".join(cur + [u])) > limit - 2:
            out.append(" ".join(cur)); cur = []
        cur.append(u)
        if int(hashlib.md5(canonical(u).encode()).hexdigest(), 16) % CDC_MOD == 0:
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def classify(docs):
    """Per doc: old chunks, new chunks, and for each new chunk (kind, best old index, jaccard)."""
    out = []
    for d in docs:
        c0, c1 = chunks(d["text0"]), chunks(d["text1"])
        if canonical(d["text0"]) == canonical(d["text1"]):
            out.append((c0, c0, [("identical", j, 1.0) for j in range(len(c0))]))
            continue
        s0 = [shingles(canonical(c), 5) for c in c0]
        pos0 = {c: j for j, c in enumerate(c0)}
        lab = []
        for c in c1:
            if c in pos0:
                lab.append(("identical", pos0[c], 1.0)); continue
            sc = shingles(canonical(c), 5)
            js = [jaccard(sc, s) for s in s0]
            b = int(np.argmax(js)) if js else -1
            lab.append(("changed", b, js[b] if js else 0.0))
        out.append((c0, c1, lab))
    return out


def stats():
    for corpus in ("arxiv_big", "python_docs"):
        docs = data.load_corpus(corpus)
        cl = classify(docs)
        rev = [x for x, d in zip(cl, docs) if canonical(d["text0"]) != canonical(d["text1"])]
        n1 = sum(len(c1) for _, c1, _ in rev)
        ident = sum(k == "identical" for _, _, lab in rev for k, _, _ in lab)
        js = np.array([j for _, _, lab in rev for k, _, j in lab if k == "changed"])
        print(f"\n{corpus}: {len(docs)} docs, {len(rev)} revised; chunks/doc (new version) "
              f"{np.mean([len(c1) for _, c1, _ in cl]):.1f}; revised-doc chunks {n1}: identical {ident} "
              f"({ident / n1:.2f}), changed {len(js)} ({len(js) / n1:.2f})")
        for th in THETAS:
            new = int((js < th).sum()); edited = len(js) - new
            print(f"  THETA {th}: edited {edited}, new {new}; embeds: full {n1}, diff {len(js)}, keep {new} "
                  f"-> keep/diff {new / max(len(js), 1):.2f}  (K3 needs <= 0.70)")


# ------------------------------------------------------------------ retrieval
def doc_ranks(Q, V, owner, gold, n_docs):
    """Doc-level rank of the gold doc, scoring each doc by its best chunk (MaxP)."""
    S = Q @ V.T
    best = np.full((len(Q), n_docs), -np.inf)
    order = np.argsort(owner, kind="stable")
    so = owner[order]
    starts = np.r_[0, np.flatnonzero(np.diff(so)) + 1]
    best[:, so[starts]] = np.maximum.reduceat(S[:, order], starts, axis=1)
    g = best[np.arange(len(Q)), gold]
    return (best > g[:, None]).sum(1) + 1


def run():
    for corpus in ("arxiv_big", "python_docs"):
        docs = data.load_corpus(corpus)
        cl = classify(docs)
        uniq = sorted({c for c0, c1, _ in cl for c in c0 + c1})
        idx = {c: i for i, c in enumerate(uniq)}
        # questions
        fq, fg, fa = facts.fact_questions(docs)
        fg = np.array(fg)
        pools = {p: fn(docs) for p, fn in queries.POOLS.items() if p in ("new", "retained")}
        for model in ("minilm", "bge"):
            E = data.cached(model, corpus, f"chunks{CHUNK}", lambda: uniq)
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
            QF = data.cached(model, corpus, "q_answer_changed", lambda: fq)
            QP = {p: data.cached(model, corpus, measure.WHATS[p], lambda t=t: t) for p, (t, _) in pools.items()}
            for th in THETAS:
                idxs = {"stale": [], "full": [], "keep": [], "append": []}   # (text, vector row, doc)
                cost = {"full": 0, "diff": 0, "keep": 0}
                for d, (c0, c1, lab) in enumerate(cl):
                    revised = canonical(docs[d]["text0"]) != canonical(docs[d]["text1"])
                    for c in c0:
                        idxs["stale"].append((c, idx[c], d)); idxs["append"].append((c, idx[c], d))
                    for c, (kind, b, j) in zip(c1, lab):
                        idxs["full"].append((c, idx[c], d))
                        if kind == "identical":
                            idxs["keep"].append((c, idx[c], d))
                        else:
                            idxs["append"].append((c, idx[c], d))
                            reuse = j >= th and b >= 0
                            idxs["keep"].append((c, idx[c0[b]] if reuse else idx[c], d))
                            cost["diff"] += 1; cost["keep"] += 0 if reuse else 1
                        if revised:
                            cost["full"] += 1
                row = {"cost": cost}           # embeds counted over revised docs only
                for pol, items in idxs.items():
                    texts = [t for t, _, _ in items]
                    V = E[[r for _, r, _ in items]]
                    owner = np.array([o for _, _, o in items])
                    S = QF @ V.T
                    top = np.argsort(-S, axis=1)[:, :K]
                    ans = [any(owner[t] == g and ({facts.norm(a) for a in A} & facts.values(texts[t]))
                               for t in tt) for tt, g, A in zip(top, fg, fa)]
                    row[pol] = {"answer@10": float(np.mean(ans))}
                    for p, (_, gold) in pools.items():
                        r = doc_ranks(QP[p], V, owner, gold, len(docs))
                        row[pol][f"{p}_hit@10"] = float(np.mean(r <= K))
                f, k = row["full"], row["keep"]
                print(f"{model}/{corpus} THETA {th}: answer@10 stale {row['stale']['answer@10']:.3f} "
                      f"full {f['answer@10']:.3f} keep {k['answer@10']:.3f} append {row['append']['answer@10']:.3f} | "
                      f"new hit@10 full {f['new_hit@10']:.3f} keep {k['new_hit@10']:.3f} | "
                      f"retained full {f['retained_hit@10']:.3f} keep {k['retained_hit@10']:.3f} | "
                      f"embeds full {cost['full']} diff {cost['diff']} keep {cost['keep']}", flush=True)
                k1 = k["answer@10"] >= f["answer@10"] - 0.02
                k2 = min(k["new_hit@10"] - f["new_hit@10"], k["retained_hit@10"] - f["retained_hit@10"]) >= -0.01
                k3 = cost["keep"] <= 0.70 * cost["diff"]
                print(f"    K1 {'pass' if k1 else 'FAIL'}  K2 {'pass' if k2 else 'FAIL'}  K3 {'pass' if k3 else 'FAIL'}")


if __name__ == "__main__":
    {"stats": stats, "run": run}[sys.argv[1]]()
