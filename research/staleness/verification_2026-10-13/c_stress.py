"""Adversarial stress tests of the p11/p12 design (critical review, 2026-10-13).
All analyses here are POST HOC.  Reads raw records + rebuilt contexts (ctx.json from v_ctx.py)."""
import collections
import json
import re

import numpy as np

from vcommon import CACHE, P9, jl, mcnemar_exact, page_boot, sample, wilson

_, _, qs = sample()
ctx = json.load(open("ctx.json"))
R = jl(CACHE / "p11/records_scale.jsonl")
GENS = ["phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]
YR = re.compile(r"\b(?:FY\s*)?((?:19|20)\d\d)\b")


def has(vs, text):
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", text) for v in vs for f in {v, v.replace(",", "")})


def lab(r):
    q = qs[r["q"]]
    return P9.commit(q["new"], q["old"], r["output"])


def ctx_info(retr, n):
    """positions of the passage carrying the old / new value, and the temporal cue"""
    q = qs[n]
    texts = ctx[f"{retr}|append|{n}"]
    key = " -- " + q["key"].replace("_", " ") + ":"
    po = [i for i, t in enumerate(texts) if key in t and has(q["old"], t.split(":", 1)[1])]
    pn = [i for i, t in enumerate(texts) if key in t and has(q["new"], t.split(":", 1)[1])]
    yo = [int(y) for i in po for y in YR.findall(texts[i].split(":", 1)[1])]
    yn = [int(y) for i in pn for y in YR.findall(texts[i].split(":", 1)[1])]
    # a separate "<field> year: 2025" passage in context is an ambiguous cue (which value does it date?)
    sep_year = any((" -- " + q["key"].replace("_", " ") + " year:") in t for t in texts)
    if yo and yn and max(yn) > max(yo):
        cue = "in-passage years, new later"
    elif yo and yn and max(yn) <= max(yo):
        cue = "in-passage years, NOT new later"
    elif yo or yn:
        cue = "year on one passage only"
    else:
        cue = "no in-passage date"
    return {"po": po[0] if po else None, "pn": pn[0] if pn else None, "cue": cue, "sep_year": sep_year}


A = [r for r in R if r["policy"] == "append" and r["variant"] == "none"]
for r in A:
    r.update(ctx_info(r["retriever"], r["q"]))
    r["lab"] = lab(r)

print("== A. Temporal cue available in the NO-LABEL context (per case, 605 per generator)")
cues = collections.Counter(r["cue"] for r in A if r["gen"] == GENS[0])
print(dict(cues))
print("cases with a separate '<field> year: YYYY' passage in context:", sum(r["sep_year"] for r in A if r["gen"] == GENS[0]))
for gen in GENS:
    print(f"\n  {gen}")
    for c in ["no in-passage date", "in-passage years, new later", "year on one passage only", "in-passage years, NOT new later"]:
        rows = [r for r in A if r["gen"] == gen and r["cue"] == c]
        if not rows:
            continue
        cc = collections.Counter(r["lab"] for r in rows)
        n = len(rows)
        print(f"    {c:<34} n={n:<4} new {cc['new']/n:.2f}  old {cc['old']/n:.2f}  both {cc['both']/n:.2f}  neither {cc['neither']/n:.2f}"
              f"   | among single commits P(old) = {cc['old']/max(cc['old']+cc['new'],1):.2f}")

print("\n== B. Position effect: which passage comes first in the context?")
for gen in GENS:
    rows = [r for r in A if r["gen"] == gen and r["po"] is not None and r["pn"] is not None]
    oldfirst = [r for r in rows if r["po"] < r["pn"]]
    newfirst = [r for r in rows if r["pn"] < r["po"]]
    f = lambda rs, v: sum(r["lab"] == v for r in rs) / max(len(rs), 1)  # noqa: E731
    # among single-value commits: P(choose the FIRST-listed value)
    single = [r for r in rows if r["lab"] in ("old", "new")]
    first = np.mean([(r["lab"] == "old") == (r["po"] < r["pn"]) for r in single])
    print(f"  {gen:<22} old-first n={len(oldfirst):<3} old-only {f(oldfirst,'old'):.2f} new-only {f(oldfirst,'new'):.2f} | "
          f"new-first n={len(newfirst):<3} old-only {f(newfirst,'old'):.2f} new-only {f(newfirst,'new'):.2f} | "
          f"single commits choosing the FIRST-listed value: {first:.2f} (n={len(single)})")
    # same within the no-cue stratum
    nc = [r for r in single if r["cue"] == "no in-passage date"]
    if nc:
        print(f"  {'':<22} no-cue stratum: P(choose first-listed) = {np.mean([(r['lab'] == 'old') == (r['po'] < r['pn']) for r in nc]):.2f} (n={len(nc)})")

print("\n== C. Pseudo-replication: identical prompts counted twice (minilm and bge)")
for pol in ("append", "replace"):
    keys = {(r["retriever"], r["q"]) for r in R if r["policy"] == pol and r["gen"] == GENS[0] and r["variant"] == "none"}
    qn = collections.defaultdict(list)
    for retr, n in keys:
        qn[n].append(tuple(ctx[f"{retr}|{pol}|{n}"]))
    dup = sum(1 for n, cs in qn.items() if len(cs) == 2 and cs[0] == cs[1])
    print(f"  {pol}: {len(keys)} cases from {len(qn)} distinct questions; {dup} questions have byte-identical contexts in both retrievers "
          f"-> {len(keys) - dup} distinct prompts")
for gen in GENS:
    seen, rows = set(), []
    for r in A:
        if r["gen"] != gen:
            continue
        k = (r["q"], tuple(ctx[f"{r['retriever']}|append|{r['q']}"]))
        if k in seen:
            continue
        seen.add(k); rows.append(r)
    k = sum(r["lab"] == "old" for r in rows)
    pages = [r["page"] for r in rows]
    print(f"  {gen:<22} distinct-prompt old-only {k}/{len(rows)} = {k/len(rows):.3f} Wilson {np.round(wilson(k, len(rows)), 3)} "
          f"page-boot {np.round(page_boot([r['lab']=='old' for r in rows], pages), 3)}; distinct questions {len({r['q'] for r in rows})}, pages {len(set(pages))}")

print("\n== D. Closed-book-style filter analogue: keep append cases whose replace answer (same retriever, q) was new-only")
rep = {(r["gen"], r["retriever"], r["q"]): lab(r) for r in R if r["policy"] == "replace"}
for gen in GENS:
    rows = [r for r in A if r["gen"] == gen and rep.get((gen, r["retriever"], r["q"])) == "new"]
    cc = collections.Counter(r["lab"] for r in rows)
    print(f"  {gen:<22} n={len(rows)}  old-only {cc['old']/len(rows):.3f}  both {cc['both']/len(rows):.3f}  new-only {cc['new']/len(rows):.3f}")

print("\n== E. Concentration of the sample")
first = [r for r in A if r["gen"] == GENS[0]]
fields = collections.Counter(qs[r["q"]]["key"] for r in first)
print("  top fields:", fields.most_common(12))
print("  share of *rank* fields:", round(sum(v for k, v in fields.items() if "rank" in k) / len(first), 3))
pc = collections.Counter(r["page"] for r in first)
print("  pages:", len(pc), "| cases on top-10 pages:", sum(v for _, v in pc.most_common(10)), "of", len(first))
cat = collections.Counter()
pg = json.load(open(CACHE / "p11_wiki_frozen" / "pages.json")) if (CACHE / "p11_wiki_frozen" / "pages.json").exists() else []

print("\n== F. Value heuristic: among single commits, P(choose the LARGER number)")
for gen in GENS:
    single = [r for r in A if r["gen"] == gen and r["lab"] in ("old", "new")]
    def val(v):
        try:
            return float(P9.facts.norm(v))
        except ValueError:
            return None
    xs = []
    for r in single:
        q = qs[r["q"]]
        o, n_ = val(q["old"][0]), val(q["new"][0])
        if o is None or n_ is None or o == n_:
            continue
        chose_old = r["lab"] == "old"
        xs.append((o > n_) == chose_old)
    print(f"  {gen:<22} P(choose larger) = {np.mean(xs):.2f} (n={len(xs)}); new is larger in {np.mean([float(P9.facts.norm(qs[r['q']]['new'][0])) > float(P9.facts.norm(qs[r['q']]['old'][0])) for r in single]):.2f} of these")

json.dump([{k: r[k] for k in ("retriever", "gen", "q", "page", "lab", "po", "pn", "cue")} for r in A], open("stress_cases.json", "w"))
