"""Pre-test 12 (RQ2): synthetic counterfactual mutation (ASE 2026) vs real updates, same data.

Kim, Pasini & Tonella, "When Knowledge Changes: Metamorphic Testing of RAG Systems with
Mutations" (ASE 2026, arXiv 2607.26843).  Their counterfactual operator: "prompts a separate
LLM to generate a plausible counterfactual statement given the original fact with the user
query ... CN adds one counterfactual passage per document"; the variance relation expects
"y' aligns with the injected fact".

Pre-registered 2026-10-12, before any counterfactual was generated.
Data      p11 SCALE: the same frozen pages, the same 400 sampled questions, and the same
          per-field passages, retrievers (minilm, bge), k = 3 and prompt.
CN index  the CURRENT version of every page, plus one LLM-written counterfactual passage for
          the queried field: "<page> -- <field>: <value'>.", where value' is a plausible
          number different from both the current and the old value (qwen2.5:7b-instruct,
          temperature 0; a fallback perturbs the number deterministically if the model
          repeats a value).
Case      the CN passage and the current value are both in the top-3 context: the synthetic
          analogue of p11's both-in-context `append` case.
Measures  per generator: P(answer = injected value) and P(answer = current value).
          - The ASE relation counts "not injected" as a violation.
          - Under real updates (p11 append), the analogous model choice is P(answer = old
            value); there the expected answer is the NEW value.
Reading   RQ2 is answered by comparing P(choose injected | CN) with P(choose old | append),
          per generator, with Wilson and page-clustered CIs.  A difference of >= 0.10 in
          either direction means synthetic CN mis-estimates the fault rate of real updates.
          < 0.10 means CN is a fair proxy for this fault.  Either outcome is reported.
Amended 2026-10-13, before any p12 result: both sides are scored with P9.commit (digit-
          boundary matching), comparing "injected only" with "old only"; "both" is reported
          for each.  The first version counted "both" as a violation for CN but as correct
          for real updates, and misread comma-joined answers.

    .venv/bin/python research/staleness/pretests/p12_counterfactual.py [--14b]
"""
import json
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
GENS = ["phi3:mini", "qwen2.5:7b-instruct"] + (["qwen2.5:14b-instruct"] if "--14b" in sys.argv else [])
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as C  # noqa: E402
import p9_wiki_infobox as P9  # noqa: E402
from research.staleness import answer_level, data, facts, generation  # noqa: E402

W = P9.W
OUT = data.CACHE / "p12"
CF_MODEL = "qwen2.5:7b-instruct"
CF_PROMPT = ("Here is a fact:\n{fact}\n\nRewrite this exact sentence, changing only the number to a "
             "different but plausible value. Output only the rewritten sentence.")


def counterfactual(fact: str, avoid: set[str]) -> tuple[str, set[str]]:
    """An LLM-written counterfactual of `fact`, with its new numeric values."""
    out = P9.generate(CF_MODEL, CF_PROMPT.format(fact=fact)).strip().splitlines()[0]
    vals = {facts.norm(x) for x in facts.NUM.findall(out) if len(x) >= facts.MIN_ANSWER}
    if not vals or vals & avoid:
        # deterministic fallback: scale the first value by 1.13 (keeps magnitude plausible)
        m = facts.NUM.search(fact.split(":", 1)[-1])
        if m:
            try:
                v = float(facts.norm(m.group()))
                new = f"{v * 1.13:,.{2 if v < 1000 else 0}f}"
                out = fact.replace(m.group(), new, 1)
                vals = {facts.norm(new)}
            except ValueError:
                pass
    return out, vals


def main():
    frozen = data.CACHE / "p11_wiki_frozen"
    P9.RAW = frozen
    pages = []
    for path in sorted(frozen.glob("*.json")):
        if path.name == "pages.json":
            continue
        p = json.loads(path.read_text())
        if p.get("skip") or not p.get("text0") or not p.get("text1"):
            continue
        pages.append(p)
    _, allq = P9.build()
    qs = random.Random(0).sample(allq, min(P9.SAMPLE, len(allq)))
    cur = []
    for p in pages:
        f1 = P9.fields(p["text1"])
        cur.append(([f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in f1.items()]
                    + C.chunks(W.prose(p["text1"])), f1))
    # one counterfactual per question
    cfs = []
    for q in qs:
        f1 = cur[q["doc"]][1]
        fact = f"{pages[q['doc']]['title']} -- {q['key'].replace('_', ' ')}: {f1[q['key']]}."
        avoid = {facts.norm(a) for a in q["new"] + q["old"]}
        cfs.append(counterfactual(fact, avoid))
    ok = [bool(v) and not (v & {facts.norm(a) for a in q["new"] + q["old"]}) for q, (_, v) in zip(qs, cfs)]
    print(f"{len(qs)} questions; counterfactuals with a fresh value: {sum(ok)}", flush=True)

    # reuse p11's passage embeddings (built exactly as p11 builds them: old + new versions),
    # and embed only the counterfactual passages
    p11_uniq = set()
    for p in pages:
        for t in ("text0", "text1"):
            f = P9.fields(p[t])
            p11_uniq |= {f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in f.items()}
            p11_uniq |= set(C.chunks(W.prose(p[t])))
    p11_uniq = sorted(p11_uniq)
    cf_list = [c for c, _ in cfs]
    passages = p11_uniq + cf_list
    idx = {}
    for i, x in enumerate(passages):
        idx.setdefault(x, i)
    recs = []
    for retr in ("minilm", "bge"):
        E = np.vstack([data.cached(retr, "p11_scale", "passages", lambda: p11_uniq),
                       data.cached(retr, "p12", "counterfactuals", lambda: cf_list)])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(retr, "p11_scale", "questions", lambda: [q["question"] for q in qs])
        base_items = [(x, d) for d, (ps, _) in enumerate(cur) for x in ps]
        base_rows = np.array([idx[x] for x, _ in base_items])
        for n, q in enumerate(qs):
            if not ok[n]:
                continue
            cf, cfv = cfs[n]
            items = base_items + [(cf, q["doc"])]               # inject one passage
            rows = np.append(base_rows, idx[cf])
            s = E[rows] @ Q[n]
            top = np.argsort(-s)[:3]
            texts = [items[t][0] for t in top]
            vals = set().union(*(facts.values(t) for t in texts))
            if not (vals & {facts.norm(a) for a in q["new"]} and vals & cfv):
                continue
            ctx = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))
            cf_raw = [x for x in facts.NUM.findall(cf.split(":", 1)[-1]) if facts.norm(x) in cfv]
            for gen in GENS:
                out = P9.generate(gen, P9.TEMPLATE.format(context=ctx, statement=q["statement"]))
                # symmetric with the real-update side: injected plays the role of 'old'
                choice = {"old": "injected", "new": "current", "both": "both", "neither": "neither"}[
                    P9.commit(q["new"], cf_raw, out)]
                recs.append({"retriever": retr, "gen": gen, "q": n, "page": q["doc"], "choice": choice, "output": out})
        print(f"[{retr}] {len(recs)} records", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")

    p11 = [json.loads(l) for l in open(data.CACHE / "p11" / "records_scale.jsonl")]
    rng = np.random.default_rng(0)

    def rate(rows, key, val):
        x = np.array([r[key] == val for r in rows]); pages_ = np.array([r["page"] for r in rows])
        up = np.unique(pages_)
        bs = [np.concatenate([x[pages_ == p] for p in rng.choice(up, len(up))]).mean() for _ in range(2000)]
        return x.sum(), len(x), x.mean(), np.percentile(bs, [2.5, 97.5])

    for r in p11:
        assert r["qtext"] == qs[r["q"]]["statement"], "p11 records come from a different question sample"
        r["commit"] = P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"])
    print()
    for gen in GENS:
        cn = [r for r in recs if r["gen"] == gen]
        k, n, m, (lo, hi) = rate(cn, "choice", "injected")
        share = lambda rows, key, v: sum(r[key] == v for r in rows) / max(len(rows), 1)  # noqa: E731
        real = [r for r in p11 if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
        if real:
            k2, n2, m2, (lo2, hi2) = rate(real, "commit", "old")
            diff = m - m2
            verdict = "CN mis-estimates real updates" if abs(diff) >= 0.10 else "CN is a fair proxy (|diff| < 0.10)"
            tail = (f" | real append chose OLD only {k2}/{n2} = {m2:.2f} [{lo2:.2f},{hi2:.2f}]"
                    f" (both {share(real, 'commit', 'both'):.2f}) | diff {diff:+.2f} -> {verdict}")
        else:
            tail = " | no p11 records for this generator"
        print(f"{gen:<22} CN: chose INJECTED only {k}/{n} = {m:.2f} [{lo:.2f},{hi:.2f}] "
              f"(current {share(cn, 'choice', 'current'):.2f}, both {share(cn, 'choice', 'both'):.2f})" + tail)


if __name__ == "__main__":
    main()
