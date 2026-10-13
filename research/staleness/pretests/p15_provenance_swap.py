"""Pre-test 15 (C4): MR-Provenance-Swap, a decidable knowledge-conflict relation, vs the published CN
oracle, on real Wikipedia revisions and on CN counterfactuals.

PRE-REGISTERED 2026-10-13, before any p15 generation.  sha256 logged in `p15_prereg.log` before the run;
the first line of the run log repeats it.

Units    all eligible questions of the p14 builder (p11 400-question sample; counterfactual with a fresh
         non-year value, no x1.13 fallback; a same-page distractor), reused verbatim from p14.
Sources  REAL: orig = old-revision value passage, mut = new-revision value passage.
         CN:   orig = current value passage,        mut = p12 counterfactual passage (qwen2.5:7b).
Dates    d_early = page rev0 date, d_late = page rev1 date (real revision dates), as "(as of YYYY-MM-DD)".
         NONE    : no dates.
         CORRECT : mut gets d_late, orig gets d_early.
         SWAPPED : mut gets d_early, orig gets d_late.
Order    M-LAST [orig, mut, distractor] and M-FIRST [mut, orig, distractor].
ADEQ     [new value passage, distractor] (REAL), once per question.
Prompt   p9 TEMPLATE unchanged; temperature 0; num_predict 24.
Models   phi3:mini, qwen2.5:7b-instruct, qwen2.5:14b-instruct; one at a time, unloaded after use.
Label    four-way committed value, P9.commit(mut, orig, answer): 'new' = mut only, 'old' = orig only,
         'both', 'neither'.

Relations (per question x source x order)
  MR-PS (valid, decidable without ground truth): the committed value must be the LATER-dated one.
         PASS        CORRECT -> mut  and SWAPPED -> orig
         INSENSITIVE same single value under CORRECT and SWAPPED (date-insensitive)
         INVERTED    CORRECT -> orig and SWAPPED -> mut
         ACK         any 'both' or 'neither' in the pair (acknowledges / abstains; reported separately)
         PS violation = INSENSITIVE or INVERTED.
  CN-oracle (published ASE relation, numeric branch, applied to the NONE condition):
         VIOLATED iff the answer commits to the ORIGINAL value alone.  ('both' counts as satisfied, as in
         their judge rubric; 'neither' as satisfied, as any change does in their numeric branch.)
Adequacy ADEQ committed 'new' >= 0.60 per model, else that model is VOID.
Hypotheses (all reported whichever way they go)
  H1 decidable faults exist: for >= 1 valid model, PS violation rate (pairs, pooled over orders and
     sources) >= 0.10 with Wilson lower bound > 0.05.
  H2 the two oracles flag different cases: per model, phi coefficient between CN-oracle VIOLATED (NONE,
     M-LAST = ASE post-chunk placement) and PS violation (same question, source, order M-LAST) < 0.30.
     Supported if phi < 0.30 for every valid model.
  H3 model ranking: ranks of valid models by CN-oracle violation rate vs PS violation rate (REAL source);
     reported as "same order" or "different order" (descriptive; 3 models).
  H4 mutant validity under a valid oracle: |PS(CN) - PS(REAL)| < 0.05 per valid model (difference with a
     question-clustered bootstrap 95% CI).
Cost      13 prompts per question per model.

    .venv/bin/python research/staleness/pretests/p15_provenance_swap.py --dry-run
    .venv/bin/python research/staleness/pretests/p15_provenance_swap.py [--models phi3:mini,...]
"""
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARGV = sys.argv[1:]
DRY = "--dry-run" in ARGV
GENS = ["phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]
if "--models" in ARGV:
    GENS = ARGV[ARGV.index("--models") + 1].split(",")
sys.argv = sys.argv[:1] + ["--dry-run"]          # keep p14's own argv handling inert
import p14_order_oracle as P14  # noqa: E402

P9, data = P14.P9, P14.data
OUT = data.CACHE / "p15"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def build():
    frozen = data.CACHE / "p11_wiki_frozen"
    pages = [json.loads(p.read_text()) for p in sorted(frozen.glob("*.json")) if p.name != "pages.json"]
    pages = [p for p in pages if not p.get("skip") and p.get("text0") and p.get("text1")]
    P14.N_Q = 10 ** 6                                   # take ALL eligible questions
    cases, pool = P14.build()
    for c in cases:
        c["d_early"], c["d_late"] = pages[c["page"]]["ts0"][:10], pages[c["page"]]["ts1"][:10]
    return sorted(cases, key=lambda c: c["q"]), pool


def prompts(c):
    lab = lambda t, d: f"(as of {d}) {t}" if d else t  # noqa: E731
    out = {}
    for src, orig, mut in (("REAL", c["old_p"], c["new_p"]), ("CN", c["new_p"], c["cf_p"])):
        for dates, (do, dm) in (("NONE", (None, None)), ("CORRECT", (c["d_early"], c["d_late"])),
                                ("SWAPPED", (c["d_late"], c["d_early"]))):
            po, pm = lab(orig, do), lab(mut, dm)
            for order, ps in (("M-LAST", [po, pm, c["dis_p"]]), ("M-FIRST", [pm, po, c["dis_p"]])):
                ctx = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(ps))
                out[(src, dates, order)] = P9.TEMPLATE.format(context=ctx, statement=c["statement"])
    ctx = f"[1] {c['new_p']}\n\n[2] {c['dis_p']}"
    out[("ADEQ", "-", "-")] = P9.TEMPLATE.format(context=ctx, statement=c["statement"])
    return out


def ps_label(a, b):                                    # a = CORRECT, b = SWAPPED committed labels
    if a in ("both", "neither") or b in ("both", "neither"):
        return "ACK"
    if a == "new" and b == "old":
        return "PASS"
    if a == b:
        return "INSENSITIVE"
    return "INVERTED"


def phi(x, y):
    a = sum(1 for i, j in zip(x, y) if i and j); b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if not i and j); d = sum(1 for i, j in zip(x, y) if not i and not j)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / den if den else float("nan"), (a, b, c, d)


def main():
    cases, pool = build()
    print(f"eligible questions {pool}; used {len(cases)}; prompts per model {13 * len(cases)}", flush=True)
    if DRY:
        for k, v in list(prompts(cases[0]).items())[:4]:
            print("\n###", k, "\n" + v)
        return
    print("script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for gen in GENS:
        for c in cases:
            for (src, dates, order), pr in prompts(c).items():
                out = P9.generate(gen, pr)
                if src == "ADEQ":
                    lab = P9.commit(c["new"], c["old"], out)
                else:
                    orig, mut = (c["old"], c["new"]) if src == "REAL" else (c["new"], c["inj"])
                    lab = P9.commit(mut, orig, out)
                recs.append({"gen": gen, "q": c["q"], "page": c["page"], "src": src, "dates": dates,
                             "order": order, "output": out, "label": lab})
        subprocess.run(["ollama", "stop", gen], check=False)
        (OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        print(f"{gen} done ({len(recs)} records so far)", flush=True)
    report(recs)


def report(recs):
    rng = random.Random(0)
    gens = list(dict.fromkeys(r["gen"] for r in recs))
    valid, rank_cn, rank_ps = [], {}, {}
    for gen in gens:
        R = {(r["q"], r["src"], r["dates"], r["order"]): r["label"] for r in recs if r["gen"] == gen}
        qs = sorted({q for q, *_ in R})
        ad = sum(R[(q, "ADEQ", "-", "-")] == "new" for q in qs) / len(qs)
        ok = ad >= 0.60
        print(f"\n===== {gen}: ADEQUACY {ad:.3f} -> {'ok' if ok else 'VOID'}")
        if not ok:
            continue
        valid.append(gen)
        allpairs = Counter(); ps_by = {}
        for src in ("REAL", "CN"):
            cnt = Counter()
            for q in qs:
                for o in ("M-LAST", "M-FIRST"):
                    l = ps_label(R[(q, src, "CORRECT", o)], R[(q, src, "SWAPPED", o)])
                    cnt[l] += 1; allpairs[l] += 1
                    ps_by[(q, src, o)] = l in ("INSENSITIVE", "INVERTED")
            n = sum(cnt.values()); v = cnt["INSENSITIVE"] + cnt["INVERTED"]
            print(f"  {src:<4} MR-PS pairs {n}: {dict(cnt)}  violation {v/n:.3f} {wilson(v, n)}")
            cnv = [R[(q, src, "NONE", "M-LAST")] == "old" for q in qs]
            print(f"  {src:<4} CN-oracle (NONE, M-LAST) violation {sum(cnv)/len(qs):.3f} {wilson(sum(cnv), len(qs))}")
            f, tab = phi(cnv, [ps_by[(q, src, "M-LAST")] for q in qs])
            print(f"  {src:<4} phi(CN-oracle violation, PS violation) at M-LAST = {f:.3f}  [a,b,c,d]={tab}")
            if src == "REAL":
                rank_cn[gen], rank_ps[gen], phi_real = sum(cnv) / len(qs), v / n, f
            if src == "CN":
                phi_cn = f
        n = sum(allpairs.values()); v = allpairs["INSENSITIVE"] + allpairs["INVERTED"]
        print(f"  H1 pooled PS violation {v}/{n} = {v/n:.3f} {wilson(v, n)}")
        # H4: question-clustered bootstrap of PS(CN) - PS(REAL)
        diffs = []
        for _ in range(2000):
            pick = [rng.choice(qs) for _ in qs]
            a = sum(ps_by[(q, "CN", o)] for q in pick for o in ("M-LAST", "M-FIRST")) / (2 * len(pick))
            b = sum(ps_by[(q, "REAL", o)] for q in pick for o in ("M-LAST", "M-FIRST")) / (2 * len(pick))
            diffs.append(a - b)
        diffs.sort()
        point = (sum(ps_by[(q, "CN", o)] for q in qs for o in ("M-LAST", "M-FIRST"))
                 - sum(ps_by[(q, "REAL", o)] for q in qs for o in ("M-LAST", "M-FIRST"))) / (2 * len(qs))
        print(f"  H4 PS(CN) - PS(REAL) = {point:+.3f} [{diffs[49]:+.3f}, {diffs[1949]:+.3f}] -> "
              f"{'within 0.05' if abs(point) < 0.05 else 'differs by >= 0.05'}")
        print(f"  H2 phi REAL {phi_real:.3f}, CN {phi_cn:.3f}")
    if valid:
        o_cn = sorted(valid, key=lambda g: rank_cn[g]); o_ps = sorted(valid, key=lambda g: rank_ps[g])
        print(f"\nH3 ranking by CN-oracle violation (REAL, low->high): {o_cn}\n   ranking by PS violation (REAL, low->high): {o_ps}"
              f" -> {'same order' if o_cn == o_ps else 'DIFFERENT order'}")


if __name__ == "__main__":
    main()
