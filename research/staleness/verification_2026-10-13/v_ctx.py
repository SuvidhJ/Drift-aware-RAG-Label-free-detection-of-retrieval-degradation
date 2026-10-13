"""Rebuild retrieval contexts with my own top-3 code; check both-in-context, leakage, years, M1,
and the p12 CN year issue."""
import collections
import json
import re

import numpy as np

from vcommon import (CACHE, P9, YEAR, jl, load_emb, numbers, pages, raw_variants, sample, versions)

norm = P9.facts.norm
NUMRE = P9.facts.NUM
pgs = pages()
docs, allq, qs = sample()
vs, fs = versions(pgs)
uniq = sorted({x for v0, v1 in vs for x in v0 + v1})
idx = {x: i for i, x in enumerate(uniq)}
R11 = jl(CACHE / "p11/records_scale.jsonl")
rec_keys = collections.defaultdict(set)
for r in R11:
    if r["gen"] == "phi3:mini" and r["variant"] == "none":
        rec_keys[(r["retriever"], r["policy"])].add(r["q"])


def literal_in(v, text):
    """value string v appears in text as a whole number (raw or without commas), OR its value
    appears among the numbers of the text."""
    for f in {v, v.replace(",", "")}:
        if re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", text):
            return True
    return bool(numbers(text) & raw_variants(v))


# ---- leakage: question text and years
leak = [q for q in qs if any(literal_in(v, q["statement"]) for v in q["new"] + q["old"])]
yrs = [q for q in qs if any(YEAR.fullmatch(norm(v)) for v in q["new"] + q["old"])]
same = [q for q in qs if {norm(v) for v in q["new"]} & {norm(v) for v in q["old"]}]
print(f"LEAK: {len(leak)} sampled questions whose text contains an answer value: {[q['statement'] for q in leak][:5]}")
print(f"YEARS: {len(yrs)} sampled questions with a 19xx/20xx value: {[(q['statement'], q['new'], q['old']) for q in yrs][:5]}")
print(f"new == old overlap: {len(same)}")
print("value-length < 3 chars:", sum(len(v) < 3 for q in qs for v in q["new"] + q["old"]))
# year-like but formatted (e.g. '2,025'), or near-year numbers worth a look
print("values in 1900-2100 numerically:", [(q["statement"], q["new"], q["old"]) for q in qs
      if any(1900 <= float(norm(v)) <= 2100 for v in q["new"] + q["old"] if re.fullmatch(r"[\d.]+", norm(v)))][:10])

out = {}
for retr in ("minilm", "bge"):
    E = load_emb(retr, "p11_scale", "passages", uniq)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    Q = load_emb(retr, "p11_scale", "questions", [q["question"] for q in qs])
    for pol in ("replace", "append", "stale"):
        items = []
        for d, (v0, v1) in enumerate(vs):
            s0 = set(v0)
            if pol == "append":
                items += [(x, d) for x in v0] + [(x, d) for x in v1 if x not in s0]
            elif pol == "replace":
                items += [(x, d) for x in v1]
            else:
                items += [(x, d) for x in v0]
        M = E[[idx[x] for x, _ in items]]
        S = Q @ M.T
        top = np.argsort(-S, axis=1)[:, :3]
        hit = np.mean([any(items[t][1] == q["doc"] for t in top[n]) for n, q in enumerate(qs)])
        print(f"M1 [{retr}] {pol:<7} gold-page hit@3 = {hit:.3f}")
        # near-ties at rank 3/4 could make the top-3 implementation dependent
        srt = -np.sort(-S, axis=1)
        ties = int(np.sum(np.abs(srt[:, 2] - srt[:, 3]) < 1e-9))
        if pol == "stale":
            continue
        sel_lib, sel_mine, notlit = set(), set(), []
        for n, q in enumerate(qs):
            texts = [items[t][0] for t in top[n]]
            vals = set().union(*(P9.facts.values(t) for t in texts))
            hn = bool(vals & {norm(a) for a in q["new"]}); ho = bool(vals & {norm(a) for a in q["old"]})
            ctx = "\n".join(texts)
            mn = any(literal_in(v, ctx) for v in q["new"]); mo = any(literal_in(v, ctx) for v in q["old"])
            if (pol == "replace" and hn) or (pol == "append" and hn and ho):
                sel_lib.add(n)
                if not ((pol == "replace" and mn) or (pol == "append" and mn and mo)):
                    notlit.append((q["statement"], q["new"], q["old"], texts))
            if (pol == "replace" and mn) or (pol == "append" and mn and mo):
                sel_mine.add(n)
            out[(retr, pol, n)] = texts
        recs = rec_keys[(retr, pol)]
        print(f"   {pol}: rank3/4 exact ties {ties}; selected(by facts.values) {len(sel_lib)}; records {len(recs)}; "
              f"same set {sel_lib == recs}; selected by my literal check {len(sel_mine)}; "
              f"selected but value NOT literally in context: {len(notlit)}")
        for e in notlit[:4]:
            print("      ", e[0], e[1], e[2], [t[:90] for t in e[3]])
        extra = sel_mine - sel_lib
        for n in list(extra)[:4]:
            print("      mine-only:", qs[n]["statement"], qs[n]["new"], qs[n]["old"], [t[:80] for t in out[(retr, pol, n)]])

# where does the old/new value come from in append cases: the field passage or prose?
src = collections.Counter()
for (retr, pol, n), texts in out.items():
    if pol != "append" or n not in rec_keys[(retr, pol)]:
        continue
    q = qs[n]
    key = q["key"].replace("_", " ")
    fld = [t for t in texts if f" -- {key}:" in t]
    src["field passage in context" if fld else "no field passage (values from prose/other fields)"] += 1
print("append cases by value source:", dict(src))
json.dump({f"{a}|{b}|{c}": t for (a, b, c), t in out.items()}, open("ctx.json", "w"))
