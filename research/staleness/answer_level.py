"""Retrieval-level vs answer-level staleness, on the same questions.

    python -m research.staleness.answer_level [--models minilm bge]

For each question two quantities, for a fully stale index and a fully fresh one:

  retrieval-level   the gold document is in the top-k.  This is what every staleness
                    number in FINDINGS.md, and the usual retrieval metrics, measure.
  answer-level      the gold document is in the top-k AND the version retrieved
                    contains a correct answer anywhere in its text.  A faithful
                    generator cannot state a number absent from its context, so this
                    is a ceiling on answer correctness -- pure string containment,
                    no embeddings, no LLM.

Two question sets:

  changed facts     a number in the sentence was revised (facts.fact_questions)
  CONTROL           a number in a sentence that is identical in both versions of a
                    revised document.  The stale document holds the right answer, so
                    stale and fresh should agree at answer level.  If they do not, the
                    metric is biased toward fresh indexes and the main result is void.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import data, facts, measure, queries
from .data import REPO


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def control_questions(docs: list[dict]):
    qs, gold, answers = [], [], []
    for i in facts.revised(docs):
        old = set(queries.sentences(docs[i]["text0"]))
        for s in queries.sentences(docs[i]["text1"]):
            if s not in old:
                continue
            a = {x for x in facts.NUM.findall(s) if len(x) >= facts.MIN_ANSWER}
            if a:
                qs.append(facts.strip_nums(s))
                gold.append(i)
                answers.append(a)
                break  # one per document, so no document dominates
    return qs, gold, answers


def evaluate(model: str, corpus: str, which: str, k: int = 10) -> dict:
    docs = data.load_corpus(corpus)
    qs, gold, answers = (
        facts.fact_questions(docs) if which == "changed" else control_questions(docs)
    )
    gold = np.array(gold, dtype=np.int64)
    Q = data.cached(model, corpus, f"q_answer_{which}", lambda: qs)
    D0 = data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs])
    D1 = data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs])
    r0, r1 = measure.gold_ranks(Q, D0, gold), measure.gold_ranks(Q, D1, gold)
    # containment by VALUE over whole-number tokens: substring matching would find
    # '13.4' inside '113.45', and spelling matching would miss '20000' vs '20,000'
    has0 = np.array([bool({facts.norm(a) for a in A} & facts.values(docs[g]["text0"]))
                     for g, A in zip(gold, answers)])
    has1 = np.array([bool({facts.norm(a) for a in A} & facts.values(docs[g]["text1"]))
                     for g, A in zip(gold, answers)])
    n = len(gold)
    out = {"n": n, "k": k}
    for lab, x in [
        ("retrieval_stale", r0 <= k), ("retrieval_fresh", r1 <= k),
        ("answer_stale", (r0 <= k) & has0), ("answer_fresh", (r1 <= k) & has1),
    ]:
        out[lab] = (float(x.mean()), *wilson(int(x.sum()), n))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["minilm", "bge"])
    ap.add_argument("--corpora", nargs="+", default=["arxiv_big", "python_docs"])
    a = ap.parse_args()
    res = {}
    print(f"{'model/corpus':<24}{'questions':<16}{'n':>5}   "
          f"{'retrieval stale/fresh':>24}   {'answer stale/fresh':>22}")
    for m in a.models:
        for c in a.corpora:
            for which in ("changed", "control"):
                r = evaluate(m, c, which)
                res[f"{m}/{c}/{which}"] = r
                rs, rf, as_, af = (r[x][0] for x in
                                   ("retrieval_stale", "retrieval_fresh", "answer_stale", "answer_fresh"))
                print(f"{m+'/'+c:<24}{which:<16}{r['n']:>5}   {rs:.3f} / {rf:.3f} ({rf-rs:+.3f})"
                      f"   {as_:.3f} / {af:.3f} ({af-as_:+.3f})", flush=True)
    path = REPO / "research" / "staleness" / "results_answer_level.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {path}")
