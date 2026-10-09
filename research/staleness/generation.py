"""Does a RAG generator actually state the outdated fact?

answer_level.py measures a ceiling: whether the correct number is even present in the
retrieved context. This closes the loop with a real generator:

    python -m research.staleness.generation build   # project venv: retrieve, write prompts
    <torch venv> research/staleness/generate_hf.py   # run the LLM (kept out of this venv)
    python -m research.staleness.generation score   # project venv: score the outputs

For each question, the top-3 documents are retrieved from the fully stale index and,
separately, from the fully fresh one -- the generator sees only what was retrieved, as
in a deployed system. It is asked to fill the blanked number(s) from that context.

Each output is scored by string match:
    correct  states a number that is new in the revision (the right answer)
    stale    states the number the revision replaced -- confidently outdated
    unknown  declines, or states no number
    other    states some other number

CONTROL: facts that did not change. Both indexes hold the right answer, so correct
rates must agree. If the generator is weak there, it is too weak for the main result.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import answer_level, data, facts
from .data import REPO

OUT = REPO / "embeddings" / "cache" / "staleness" / "generation"
TOPK = 3
MAX_DOC_CHARS = 2000
CONTROL_SAMPLE = 120


_norm = facts.norm


def blanked(s: str, targets: set[str]) -> str:
    """Blank only the numbers being asked about; leave every other number visible.

    Blanking all numbers made the task unnatural (one control sentence had five
    blanks) and the adequacy pilot failed on it. The question a user would ask names
    one fact, so only that fact is blanked. Chosen on CONTROL outputs only, before any
    changed-fact question was generated.
    """
    want = {facts.norm(t) for t in targets}
    return facts.NUM.sub(lambda m: "___" if facts.norm(m.group()) in want else m.group(), s)


def build(model: str = "minilm") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    items = []
    for corpus in ("arxiv_big", "python_docs"):
        docs = data.load_corpus(corpus)
        D0 = data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs])
        D1 = data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs])

        # changed facts, with the sentence each came from and the numbers it replaced
        sets = {}
        rows = []
        for i, s, o in facts.aligned_edits(docs):
            # compare numbers by VALUE, not spelling: '17.67%' -> '17.67' or
            # '20000' -> '20,000' is a formatting change, not a changed fact
            sv = {_norm(x) for x in facts.NUM.findall(s)}
            ov = {_norm(x) for x in facts.NUM.findall(o)}
            new = {x for x in facts.NUM.findall(s) if _norm(x) not in ov and len(x) >= facts.MIN_ANSWER}
            old = {x for x in facts.NUM.findall(o) if _norm(x) not in sv and len(x) >= facts.MIN_ANSWER}
            if new:
                rows.append((facts.strip_nums(s), i, new, old, blanked(s, new)))
        sets["changed"] = rows
        # control: numbers in sentences identical in both versions
        cq, cg, ca = answer_level.control_questions(docs)
        crow = []
        for q, g, a in zip(cq, cg, ca):
            src = next(s for s in data_sentences(docs[g]["text1"]) if facts.strip_nums(s) == q)
            # one target per control question: its first distinctive number
            target = next(x for x in facts.NUM.findall(src) if len(x) >= facts.MIN_ANSWER)
            crow.append((q, g, {target}, set(), blanked(src, {target})))
        pick = rng.choice(len(crow), min(CONTROL_SAMPLE, len(crow)), replace=False)
        sets["control"] = [crow[j] for j in sorted(pick)]

        for which, rows in sets.items():
            Q = data.cached(model, corpus, f"q_gen_{which}", lambda r=rows: [x[0] for x in r])
            for index, D, col in (("stale", D0, "text0"), ("fresh", D1, "text1")):
                top = np.argsort(-(Q @ D.T), axis=1)[:, :TOPK]
                for (q, g, new, old, stmt), t in zip(rows, top):
                    ctx = "\n\n".join(f"[{k+1}] {docs[j][col][:MAX_DOC_CHARS]}"
                                      for k, j in enumerate(t))
                    items.append({
                        "corpus": corpus, "set": which, "index": index, "gold": int(g),
                        "gold_retrieved": bool(g in t), "statement": stmt, "context": ctx,
                        "answer": sorted(new), "old": sorted(old),
                    })
    path = OUT / "prompts.jsonl"
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} prompts -> {path}")


def data_sentences(text: str) -> list[str]:
    from .queries import sentences
    return sentences(text)


def classify(output: str, answer: list[str], old: list[str]) -> str:
    said = {_norm(x) for x in facts.NUM.findall(output)} | facts.spoken_values(output)
    if said & {_norm(a) for a in answer}:
        return "correct"
    if said & {_norm(o) for o in old}:
        return "stale"
    if not said:
        return "unknown"
    return "other"


def score() -> None:
    rows = [json.loads(l) for l in open(OUT / "outputs.jsonl")]
    print(f"{len(rows)} generations\n")
    print(f"{'corpus':<12}{'questions':<10}{'index':<7}{'n':>5}{'correct':>9}{'STALE':>8}"
          f"{'unknown':>9}{'other':>7}{'  gold retrieved':>17}")
    summary = {}
    for corpus in ("arxiv_big", "python_docs"):
        for which in ("changed", "control"):
            for index in ("stale", "fresh"):
                r = [x for x in rows if (x["corpus"], x["set"], x["index"]) == (corpus, which, index)]
                if not r:
                    continue
                c = [classify(x["output"], x["answer"], x["old"]) for x in r]
                n = len(c)
                frac = {k: c.count(k) / n for k in ("correct", "stale", "unknown", "other")}
                ci = {k: answer_level.wilson(c.count(k), n) for k in frac}
                gr = float(np.mean([x["gold_retrieved"] for x in r]))
                # among answers where the model committed to a number: abstentions
                # cannot hide anything, and this is where a confident wrong answer shows
                committed = [k for k in c if k != "unknown"]
                m = len(committed)
                among = {k: (committed.count(k) / m if m else 0.0) for k in ("correct", "stale", "other")}
                summary[f"{corpus}/{which}/{index}"] = {
                    "n": n, **frac, "ci": ci, "gold_retrieved": gr,
                    "committed": m, "among_committed": among,
                    "among_ci": {k: answer_level.wilson(committed.count(k), m) for k in among},
                }
                print(f"{corpus:<12}{which:<10}{index:<7}{n:>5}{frac['correct']:>9.3f}"
                      f"{frac['stale']:>8.3f}{frac['unknown']:>9.3f}{frac['other']:>7.3f}{gr:>17.3f}"
                      f"   | committed {m:>3}: correct {among['correct']:.2f}  STALE {among['stale']:.2f}")
            print()
    (REPO / "research" / "staleness" / "results_generation.json").write_text(
        json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
