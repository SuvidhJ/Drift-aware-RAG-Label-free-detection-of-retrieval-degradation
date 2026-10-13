"""Recompute p12 (CN vs real) and count the x1.13 fallback, from the generation cache."""
import collections
import json
import re

import numpy as np

from vcommon import (CACHE, P9, YEAR, C, gen_cache, gkey, jl, numbers, page_boot, pages, raw_variants,
                     sample, wilson)

CF_MODEL = "qwen2.5:7b-instruct"
CF_PROMPT = ("Here is a fact:\n{fact}\n\nRewrite this exact sentence, changing only the number to a "
             "different but plausible value. Output only the rewritten sentence.")
NUMRE = P9.facts.NUM
norm = P9.facts.norm
pgs = pages()
docs, allq, qs = sample()
cache = gen_cache(CF_MODEL)

cfs, info = [], collections.Counter()
examples = collections.defaultdict(list)
for n, q in enumerate(qs):
    f1 = P9.fields(pgs[q["doc"]]["text1"])
    fact = f"{pgs[q['doc']]['title']} -- {q['key'].replace('_', ' ')}: {f1[q['key']]}."
    k = gkey(CF_MODEL, CF_PROMPT.format(fact=fact))
    assert k in cache, "counterfactual not in cache"
    llm = cache[k].strip().splitlines()[0]
    avoid = {norm(a) for a in q["new"] + q["old"]}
    vals = {norm(x) for x in NUMRE.findall(llm) if len(x) >= 3}
    fallback = (not vals) or bool(vals & avoid)
    out = llm
    if fallback:
        m = NUMRE.search(fact.split(":", 1)[-1])
        v = float(norm(m.group()))
        newv = f"{v * 1.13:,.{2 if v < 1000 else 0}f}"
        out = fact.replace(m.group(), newv, 1)
        vals = {norm(newv)}
        examples["fallback"].append((fact, llm, out))
    info["fallback" if fallback else "llm"] += 1
    # does the CF passage still carry the CURRENT value? (LLM changed some other number)
    cur_vals = {norm(x) for x in q["new"]}
    if {norm(x) for x in NUMRE.findall(out.split(":", 1)[-1])} & cur_vals:
        info["cf_still_contains_current_value"] += 1
        examples["still_current"].append((fact, out))
    if any(YEAR.fullmatch(v) for v in vals):
        info["cf_values_include_a_year"] += 1
        examples["year"].append((fact, out, sorted(vals)))
    cfs.append((out, vals))
print(dict(info))
for k, ex in examples.items():
    print(f"\n-- {k} ({len(ex)}):")
    for e in ex[:8]:
        print("  ", e)

# ---- recompute CN choices independently
R12 = jl(CACHE / "p12/records.jsonl")
print(f"\np12 records {len(R12)}; by gen {collections.Counter(r['gen'] for r in R12)}")
print("p12 page == sample doc:", all(r["page"] == qs[r["q"]]["doc"] for r in R12))
mism = []
for r in R12:
    q = qs[r["q"]]
    cf, cfv = cfs[r["q"]]
    inj = [x for x in NUMRE.findall(cf.split(":", 1)[-1]) if norm(x) in cfv]
    # digit-boundary matching, written independently: a value counts if it appears as a whole
    # number (not part of a longer digit run), with or without its thousands commas
    def has(vs, out):
        return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", out)
                   for v in vs for f in {v, v.replace(",", "")})
    c = {(1, 0): "current", (0, 1): "injected", (1, 1): "both", (0, 0): "neither"}[(has(q["new"], r["output"]), has(inj, r["output"]))]
    r["mine"] = c
    if c != r["choice"]:
        mism.append((r["output"], q["new"], inj, r["choice"], c))
print("choice mismatches vs stored:", len(mism), mism[:5])

R11 = jl(CACHE / "p11/records_scale.jsonl")
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    cn = [r for r in R12 if r["gen"] == gen]
    cc = collections.Counter(r["mine"] for r in cn)
    k = cc["injected"]
    x = [r["mine"] == "injected" for r in cn]
    print(f"\n{gen}: CN n={len(cn)} pages={len({r['page'] for r in cn})} injected-only {k}/{len(cn)} = {k/len(cn):.3f} "
          f"Wilson {tuple(round(v,3) for v in wilson(k, len(cn)))} page-boot {tuple(round(v,3) for v in page_boot(x, [r['page'] for r in cn]))}")
    print("   CN shares:", {c: round(cc[c] / len(cn), 3) for c in ("injected", "current", "both", "neither")})
    real = [r for r in R11 if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    ko = sum(P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"]) == "old" for r in real)
    kb = sum(P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"]) == "both" for r in real)
    print(f"   real old-only {ko}/{len(real)} = {ko/len(real):.3f}; both {kb/len(real):.3f}; diff CN-real = {k/len(cn) - ko/len(real):+.3f}")

# ---- post hoc: paired subset (same retriever, question) present on both sides
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    cn = {(r["retriever"], r["q"]): r["mine"] for r in R12 if r["gen"] == gen}
    real = {(r["retriever"], r["q"]): P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"])
            for r in R11 if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"}
    both = sorted(cn.keys() & real.keys())
    a = np.mean([cn[k] == "injected" for k in both]); b = np.mean([real[k] == "old" for k in both])
    print(f"POST HOC paired {gen}: {len(both)} shared cases; CN injected-only {a:.3f} vs real old-only {b:.3f} (diff {a-b:+.3f})")
json.dump([c for c, _ in cfs], open("cfs.json", "w"))
