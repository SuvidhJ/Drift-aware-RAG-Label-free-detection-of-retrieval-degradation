"""Recompute every p11 number in §0.3 from records_scale.jsonl / records_fresh.jsonl."""
import collections
import hashlib
import json

from vcommon import (CACHE, P9, jl, mcnemar_exact, mine_commit, page_boot, sample, wilson)

docs, allq, qs = sample()
h = hashlib.sha256(json.dumps([(q["doc"], q["key"], q["new"], q["old"]) for q in qs]).encode()).hexdigest()[:16]
print(f"pages with both snapshots {len(docs)}; questions {len(allq)}; sampled {len(qs)}; sample hash {h}")

R = jl(CACHE / "p11/records_scale.jsonl")
bad_q = [r for r in R if r["qtext"] != qs[r["q"]]["statement"]]
bad_p = [r for r in R if r["page"] != qs[r["q"]]["doc"]]
print(f"records {len(R)}; qtext mismatches {len(bad_q)}; page mismatches {len(bad_p)}")

# independent label for every record
for r in R:
    q = qs[r["q"]]
    r["mine"] = mine_commit(q["new"], q["old"], r["output"])
    r["p9c"] = P9.commit(q["new"], q["old"], r["output"])  # only to compare scorers

GENS = ["phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]
print("\n== counts by (policy, variant, gen)")
print(collections.Counter((r["policy"], r["variant"], r["gen"]) for r in R))
for gen in GENS:
    rep = [r for r in R if r["gen"] == gen and r["policy"] == "replace"]
    ap = [r for r in R if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    dt = [r for r in R if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "dated"]
    k = sum(r["class"] == "correct" for r in rep)
    print(f"\n{gen}\n  adequacy (stored class=correct) {k}/{len(rep)} = {k/len(rep):.3f}  Wilson {wilson(k, len(rep))}")
    km = sum(r["mine"] in ("new", "both") for r in rep)
    kn = sum(r["mine"] == "new" for r in rep)
    print(f"  adequacy (mine: new in answer) {km}/{len(rep)} = {km/len(rep):.3f}; new-only {kn/len(rep):.3f}")
    print(f"  replace commit (mine) {dict(collections.Counter(r['mine'] for r in rep))}  old-only = "
          f"{sum(r['mine']=='old' for r in rep)/len(rep):.3f}")
    s = [r["class"] == "stale" for r in ap]
    pg = [r["page"] for r in ap]
    print(f"  append none: n={len(ap)} pages={len(set(pg))} stale(stored) {sum(s)}/{len(s)} = {sum(s)/len(s):.3f} "
          f"Wilson {tuple(round(x,3) for x in wilson(sum(s), len(s)))} page-boot {tuple(round(x,3) for x in page_boot(s, pg))}")
    for lab in ("mine", "p9c"):
        c = collections.Counter(r[lab] for r in ap)
        print(f"  append none commit[{lab}]: " + " / ".join(f"{x} {c[x]/len(ap):.3f} ({c[x]})" for x in ("new", "old", "both", "neither")))
    dis = [(r["output"], r["mine"], r["p9c"]) for r in R if r["gen"] == gen and r["mine"] != r["p9c"]]
    print(f"  scorer disagreements mine vs P9.commit (all policies): {len(dis)}  e.g. {dis[:6]}")
    if dt:
        base = {(r["retriever"], r["q"]): r["class"] for r in ap}
        sd = sum(r["class"] == "stale" for r in dt)
        b = sum(base[(r["retriever"], r["q"])] == "stale" and r["class"] != "stale" for r in dt)
        c = sum(base[(r["retriever"], r["q"])] != "stale" and r["class"] == "stale" for r in dt)
        print(f"  dated: stale {sd}/{len(dt)} = {sd/len(dt):.3f}; McNemar fixed {b} broke {c} exact p = {mcnemar_exact(b, c):.2g}")
        # same test under the independent commit labels (post hoc, for robustness)
        bm = {(r["retriever"], r["q"]): r["mine"] for r in ap}
        sdm = sum(r["mine"] == "old" for r in dt)
        b2 = sum(bm[(r["retriever"], r["q"])] == "old" and r["mine"] != "old" for r in dt)
        c2 = sum(bm[(r["retriever"], r["q"])] != "old" and r["mine"] == "old" for r in dt)
        print(f"  dated (mine old-only): {sdm}/{len(dt)} = {sdm/len(dt):.3f}; fixed {b2} broke {c2} p = {mcnemar_exact(b2, c2):.2g}")
    else:
        print("  dated: none")

# key consistency: same (retriever,q) set in append/none for every gen?
sets = {g: {(r["retriever"], r["q"]) for r in R if r["gen"] == g and r["policy"] == "append" and r["variant"] == "none"} for g in GENS}
print("\nappend case sets identical across generators:", sets[GENS[0]] == sets[GENS[1]] == sets[GENS[2]])
print("replace case sets identical:", len({frozenset((r['retriever'], r['q']) for r in R if r['gen'] == g and r['policy'] == 'replace') for g in GENS}) == 1)
print("both retrievers counts append:", collections.Counter(r["retriever"] for r in R if r["gen"] == GENS[0] and r["policy"] == "append" and r["variant"] == "none"))

# run 1
R1 = jl(CACHE / "p11/records_scale_run1_2026-10-12.jsonl")
print("\n== run 1 file:", len(R1), "records")
for gen in GENS[:2]:
    ap = [r for r in R1 if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    rep = [r for r in R1 if r["gen"] == gen and r["policy"] == "replace"]
    if not ap:
        continue
    s = sum(r["class"] == "stale" for r in ap)
    pgs = {r.get("page") for r in ap}
    k = sum(r["class"] == "correct" for r in rep)
    print(f"  {gen}: append n={len(ap)} pages={len(pgs)} stale {s/len(ap):.3f}; adequacy {k}/{len(rep)} = {k/len(rep):.3f}")

# FRESH
F = jl(CACHE / "p11/records_fresh.jsonl")
print("\n== FRESH", len(F), "records")
for gen in GENS:
    for pol, var in (("replace", "none"), ("append", "none"), ("append", "dated")):
        rows = [r for r in F if r["gen"] == gen and r["policy"] == pol and r["variant"] == var]
        if rows:
            c = collections.Counter(r["class"] for r in rows)
            print(f"  {gen:<22} {pol:<8}{var:<6} n={len(rows)} pages={len({r['page'] for r in rows})} {dict(c)}")
print("\n  FRESH append/none outputs (for hand reading):")
for r in F:
    if r["policy"] == "append" and r["variant"] == "none":
        print(f"   {r['gen'][:12]:<12} {r['retriever']:<6} q{r['q']:<3} {r['class']:<8} | {r['qtext'][:60]} | {r['output'][:70]!r}")
