"""Pre-test 8: do version labels stop the generator from stating the outdated value?

Answers the main reviewer risk of p7: Stale-Document Poisoning (arXiv 2609.31342) reports
models following up-to-date evidence when they can tell it is current.  p7's context
carried no version information.  Here every passage is labelled with its document and
version (arXiv 'v1'/'v3'; Python docs '3.8'/'3.13'; a passage present in both versions
carries both labels).

Pre-registered 2026-10-11, before any generation.  A framing test, not a kill test:
  cases       p7's both-in-context cases: `append` index, k = 3, both the new and the
              replaced value in the top-3 context (161 per generator)
  variants    A  labels only (same prompt as p7)
              B  labels + "If passages disagree, use the one from the most recent version."
  generators  phi3:mini, qwen2.5:7b-instruct (local Ollama, temperature 0)
  reading     B stale < 0.05 for both generators -> "the default (unlabelled) fails ~17%;
              labelling versions fixes it"; B stale >= 0.10 -> "labels do not fix it";
              otherwise a partial fix.  The p7 (unlabelled) rate is printed for reference.

    .venv/bin/python research/staleness/pretests/p8_version_labels.py
"""
import hashlib
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as P  # noqa: E402
from research.staleness import answer_level, data, facts, generation  # noqa: E402

SYSTEM = "You answer strictly from the provided context and never from memory."
TEMPLATE = (                     # p7's template, unchanged
    "Context:\n{context}\n\n"
    "Fill in each blank (___) in the statement below with the exact number taken from "
    "the context. Reply with only the number(s), separated by commas. If the context "
    "does not contain it, reply: unknown\n\nStatement: {statement}"
)
INSTRUCTION = "If passages disagree, use the one from the most recent version.\n\n"
OUT = data.CACHE / "p8_generation"
OUT.mkdir(parents=True, exist_ok=True)
_cache: dict[str, dict[str, str]] = {}


def generate(model: str, prompt: str) -> str:
    store = OUT / f"{model.replace(':', '_')}.jsonl"
    if model not in _cache:
        _cache[model] = {}
        if store.exists():
            for line in open(store):
                r = json.loads(line)
                _cache[model][r["key"]] = r["output"]
    key = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()
    if key in _cache[model]:
        return _cache[model][key]
    body = {"model": model, "stream": False, "options": {"temperature": 0, "seed": 0, "num_predict": 24},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())["message"]["content"].strip()
    _cache[model][key] = out
    with open(store, "a") as f:
        f.write(json.dumps({"key": key, "output": out}) + "\n")
    return out


def questions(docs):       # identical to p7.questions
    out, seen = [], set()
    for i, s, o in facts.aligned_edits(docs):
        new = {x for x in facts.NUM.findall(s) if facts.norm(x) not in facts.values(o) and len(x) >= facts.MIN_ANSWER}
        old = {x for x in facts.NUM.findall(o) if facts.norm(x) not in facts.values(s) and len(x) >= facts.MIN_ANSWER}
        q = facts.strip_nums(s)
        if new and (q, i) not in seen:
            seen.add((q, i))
            out.append((q, i, new, old, generation.blanked(s, new)))
    return out


def label(doc: dict, which: str) -> str:
    v0, v1 = doc["rev0"], doc["rev1"]
    if "v" in str(v1) and "." in str(v1):          # arXiv ids such as 2401.00609v3
        v0, v1 = str(v0).rsplit("v", 1)[-1], str(v1).rsplit("v", 1)[-1]
        v0, v1 = f"v{v0}", f"v{v1}"
    ver = {"old": v0, "new": v1, "both": f"{v0} and {v1}"}[which]
    return f"(document {doc['pageid']}, version {ver})"


cases = []                                        # (corpus, retriever, q, labelled ctx, stmt, new, old)
for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    cl = P.classify(docs)
    uniq = sorted({c for c0, c1, _ in cl for c in c0 + c1})
    idx = {c: i for i, c in enumerate(uniq)}
    qs = questions(docs)
    for retr in ("minilm", "bge"):
        E = data.cached(retr, corpus, f"chunks{P.CHUNK}", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(retr, corpus, "q_answer_changed", lambda: [x[0] for x in qs])
        items = []                                # (text, doc, version tag) -- p7's `append` index
        for d, (c0, c1, lab) in enumerate(cl):
            s1 = set(c1)
            items += [(c, d, "both" if c in s1 else "old") for c in c0]
            items += [(c, d, "new") for c, (k, _, _) in zip(c1, lab) if k != "identical"]
        top = np.argsort(-(Q @ E[[idx[c] for c, _, _ in items]].T), axis=1)[:, :3]
        for n, (_, g, new, old, stmt) in enumerate(qs):
            if not old:
                continue
            texts = [items[t][0] for t in top[n]]
            vals = set().union(*(facts.values(t) for t in texts))
            if not (vals & {facts.norm(a) for a in new} and vals & {facts.norm(a) for a in old}):
                continue
            ctx = "\n\n".join(f"[{j+1}] {label(docs[items[t][1]], items[t][2])} {items[t][0]}"
                              for j, t in enumerate(top[n]))
            cases.append((corpus, retr, n, ctx, stmt, sorted(new), sorted(old)))
print(f"{len(cases)} both-in-context cases (p7 had 161)")

RECORDS = []
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    for variant in ("A", "B"):
        cls = []
        for corpus, retr, n, ctx, stmt, new, old in cases:
            prompt = TEMPLATE.format(context=ctx, statement=stmt)
            if variant == "B":
                prompt = prompt.replace("Fill in each blank", INSTRUCTION + "Fill in each blank", 1)
            out = generate(gen, prompt)
            c = generation.classify(out, new, old)
            cls.append(c)
            RECORDS.append({"gen": gen, "variant": variant, "corpus": corpus, "retriever": retr, "q": n,
                            "output": out, "class": c, "new": new, "old": old})
        cls = np.array(cls)
        k = int((cls == "stale").sum())
        print(f"{gen:<22} variant {variant}: stale {k}/{len(cls)} = {k/len(cls):.2f} "
              f"{np.round(answer_level.wilson(k, len(cls)), 3)}  correct {np.mean(cls == 'correct'):.2f}  "
              f"unknown {np.mean(cls == 'unknown'):.2f}  other {np.mean(cls == 'other'):.2f}", flush=True)
    ref = [json.loads(l) for l in open(data.CACHE / "p7_generation" / f"records_{gen.replace(':', '_')}.jsonl")]
    ref = [r for r in ref if r["policy"] == "append" and r["k"] == 3 and r["has_new"] and r["has_old"]]
    print(f"{gen:<22} p7 (no labels):  stale {sum(r['class'] == 'stale' for r in ref)}/{len(ref)}", flush=True)
with open(OUT / "records.jsonl", "w") as f:
    for r in RECORDS:
        f.write(json.dumps(r) + "\n")
