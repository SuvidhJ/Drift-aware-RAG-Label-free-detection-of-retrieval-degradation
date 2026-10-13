"""Pre-test 7: with BOTH versions of a revised document indexed, does the generator state the
outdated value?  The decisive test for improving "When Knowledge Changes: Metamorphic
Testing of RAG Systems with Mutations" (ASE 2026, arXiv 2607.26843) with real revisions.

Pre-registered 2026-10-10, before any generation.

Setup:
  index       128-token content-defined chunks (p5); `append` keeps both versions,
              `replace` keeps only the new version (the control)
  retrievers  minilm, bge;  k = 3 (primary) and k = 1
  questions   changed-fact questions that have a replaced value (arXiv 50, Python docs 58)
  generator   qwen2.5:7b-instruct via a local Ollama server, temperature 0, with the
              project's own SYSTEM/TEMPLATE (accepted on control questions in generation.py)
  context     top-k chunks in rank order, numbered, with no version metadata (as in the ASE
              paper's counterfactual operator)
  statement   the revised sentence with the new number(s) blanked (generation.blanked)
  scoring     generation.classify: correct / stale / unknown / other, by value

  adequacy    in `replace` at k = 3, where the new value is in context, correct >= 0.60;
              otherwise the generator is too weak and the test is void
  KILL        in `append` at k = 3, among cases where both values are in the context,
              pooled stale-answer rate < 0.10 -> real updates fail only at k = 1, and the
              improvement is too thin

    .venv/bin/python research/staleness/pretests/p7_generator_conflict.py [model]
"""
import hashlib
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct"
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as P  # noqa: E402
from research.staleness import answer_level, data, facts, generation  # noqa: E402

SYSTEM = "You answer strictly from the provided context and never from memory."
TEMPLATE = (
    "Context:\n{context}\n\n"
    "Fill in each blank (___) in the statement below with the exact number taken from "
    "the context. Reply with only the number(s), separated by commas. If the context "
    "does not contain it, reply: unknown\n\nStatement: {statement}"
)
OUT = data.CACHE / "p7_generation"
OUT.mkdir(parents=True, exist_ok=True)
STORE = OUT / f"{MODEL.replace(':', '_').replace('/', '_')}.jsonl"
_done = {}
if STORE.exists():
    for line in open(STORE):
        r = json.loads(line)
        _done[r["key"]] = r["output"]


def generate(context: str, statement: str) -> str:
    key = hashlib.sha256(f"{MODEL}|{context}|{statement}".encode()).hexdigest()
    if key in _done:
        return _done[key]
    body = {"model": MODEL, "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 24},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": TEMPLATE.format(context=context, statement=statement)}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["message"]["content"].strip()
    _done[key] = out
    with open(STORE, "a") as f:
        f.write(json.dumps({"key": key, "output": out}) + "\n")
    return out


def questions(docs):
    out, seen = [], set()
    for i, s, o in facts.aligned_edits(docs):
        new = {x for x in facts.NUM.findall(s) if facts.norm(x) not in facts.values(o) and len(x) >= facts.MIN_ANSWER}
        old = {x for x in facts.NUM.findall(o) if facts.norm(x) not in facts.values(s) and len(x) >= facts.MIN_ANSWER}
        q = facts.strip_nums(s)
        if new and (q, i) not in seen:
            seen.add((q, i))
            out.append((q, i, new, old, generation.blanked(s, new)))
    return out


pooled = {"both": [], "adequacy": []}
RECORDS = []
summary = {}
for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    cl = P.classify(docs)
    uniq = sorted({c for c0, c1, _ in cl for c in c0 + c1})
    idx = {c: i for i, c in enumerate(uniq)}
    qs = [x for x in questions(docs)]
    keep = [n for n, x in enumerate(qs) if x[3]]                 # has a replaced value
    for model in ("minilm", "bge"):
        E = data.cached(model, corpus, f"chunks{P.CHUNK}", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(model, corpus, "q_answer_changed", lambda: [x[0] for x in qs])
        for pol in ("replace", "append"):
            items = []
            for d, (c0, c1, lab) in enumerate(cl):
                if pol == "append":
                    items += [(c, d) for c in c0] + [(c, d) for c, (k, _, _) in zip(c1, lab) if k != "identical"]
                else:
                    items += [(c, d) for c in c1]
            S = Q @ E[[idx[c] for c, _ in items]].T
            for k in (3, 1):
                top = np.argsort(-S, axis=1)[:, :k]
                lab_, both_ = [], []
                for n in keep:
                    _, g, new, old, stmt = qs[n]
                    ctx_texts = [items[t][0] for t in top[n]]
                    vals = set().union(*(facts.values(t) for t in ctx_texts))
                    has_new = bool(vals & {facts.norm(a) for a in new})
                    has_old = bool(vals & {facts.norm(a) for a in old})
                    out = generate("\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(ctx_texts)), stmt)
                    c = generation.classify(out, sorted(new), sorted(old))
                    lab_.append(c)
                    both_.append(has_new and has_old)
                    RECORDS.append({"model": model, "corpus": corpus, "policy": pol, "k": k, "statement": stmt,
                                    "new": sorted(new), "old": sorted(old), "has_new": has_new, "has_old": has_old,
                                    "output": out, "class": c, "q": int(n)})
                    if pol == "append" and k == 3 and has_new and has_old:
                        pooled["both"].append(c)
                    if pol == "replace" and k == 3 and has_new:
                        pooled["adequacy"].append(c)
                lab_ = np.array(lab_); both_ = np.array(both_)
                frac = {x: float(np.mean(lab_ == x)) for x in ("correct", "stale", "unknown", "other")}
                bstale = float(np.mean(lab_[both_] == "stale")) if both_.any() else float("nan")
                summary[f"{model}/{corpus}/{pol}/k{k}"] = {**frac, "n": len(lab_), "both_in_ctx": int(both_.sum()),
                                                           "stale_when_both": bstale}
                print(f"{model}/{corpus} {pol:<7} k={k}  n={len(lab_)}  correct {frac['correct']:.2f}  STALE {frac['stale']:.2f}"
                      f"  unknown {frac['unknown']:.2f}  other {frac['other']:.2f}   both-in-context {int(both_.sum())}:"
                      f" stale {bstale:.2f}", flush=True)

b = np.array(pooled["both"]); a = np.array(pooled["adequacy"])
ka, kb = int((a == "correct").sum()), int((b == "stale").sum())
print(f"\nADEQUACY  replace k=3, new value in context: correct {ka}/{len(a)} = {ka/len(a):.2f} "
      f"{np.round(answer_level.wilson(ka, len(a)), 3)}  -> {'ok' if ka/len(a) >= 0.60 else 'VOID (generator too weak)'}")
print(f"PRIMARY   append k=3, both values in context: stale {kb}/{len(b)} = {kb/len(b):.2f} "
      f"{np.round(answer_level.wilson(kb, len(b)), 3)}  -> {'NO-GO (< 0.10)' if kb/len(b) < 0.10 else 'pass'}"
      f"   [correct {np.mean(b == 'correct'):.2f}, unknown {np.mean(b == 'unknown'):.2f}, other {np.mean(b == 'other'):.2f}]")
(OUT / f"summary_{MODEL.replace(':', '_')}.json").write_text(json.dumps(summary, indent=1))
with open(OUT / f"records_{MODEL.replace(':', '_')}.jsonl", "w") as f:
    for r in RECORDS:
        f.write(json.dumps(r) + "\n")
