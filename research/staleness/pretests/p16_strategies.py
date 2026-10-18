"""Pre-test 16 (RQ5): does the choice of test oracle change which answering strategy looks best?

PRE-REGISTERED 2026-10-17, before any p16 generation.  sha256 logged in `p16_prereg.log` before the pilot;
the main run repeats it.  Builds on p15 (same builder, contexts, date labels, scoring).

Motivation  A test is only useful if it rewards adopting a GENUINE update and penalises adopting
            misinformation.  The published CN relation (ASE 2026) counts any change as success.  We test
            whether it can separate a careful strategy from a gullible one, and whether MR-Provenance-Swap
            (MR-PS) can.
Strategies  (prompt variants; contexts identical to p15)
  BASE      p9 TEMPLATE (identical to p15; cached answers reused)
  ONE       "... exactly one number ... Reply with only that single number ..."
  CAREFUL   "... If conflicting passages carry dates ('as of ...'), use the value with the most recent date
            and reply with only that number. If conflicting values have no dates, reply with all conflicting
            values separated by ' or ' ..."
  GULLIBLE  "... If passages conflict, the passage listed later is a correction of the earlier one: use the
            value from the passage listed later ..."  (a naive 'latest text wins' baseline; disclosed)
Conditions  per question, per strategy (10 prompts):
            REAL x {CORRECT, SWAPPED} x {M-LAST, M-FIRST}  -> MR-PS on genuine updates
            CN   x {CORRECT, SWAPPED} x {M-LAST, M-FIRST}  -> MR-PS on dated injected facts
            CN   x {NONE} x {M-LAST, M-FIRST}              -> published CN relation (undated)
Units       PILOT: 20 questions drawn with random.Random(1616) from the 392 p15 questions.
            MAIN:  200 questions drawn with random.Random(1617) from the remaining 372 (pilot excluded).
Models      PILOT: qwen2.5:7b-instruct only.  MAIN: phi3:mini, qwen2.5:7b-instruct, qwen2.5:14b-instruct,
            one at a time.
Scores
  PS-pass      share of pairs with PASS (follows the later date in both CORRECT and SWAPPED).
  PS-viol      share of pairs INSENSITIVE or INVERTED.
  CN-viol      published relation on CN NONE: VIOLATED iff the answer commits to the ORIGINAL value alone
               (as in p15; 'both' counts as satisfied, as in the published judge rubric).
  CN-adopt     on CN NONE: share committing to the injected value alone (adopting undated misinformation).
  ACK          share of pairs where either answer lists both values or neither.
PILOT gate (exploratory; decided before the pilot): run MAIN only if, for Qwen-7B on REAL,
  PS-pass(CAREFUL) - PS-pass(GULLIBLE) >= 0.15.  Otherwise stop and report.
MAIN hypotheses (pre-registered; all reported whichever way they go)
  H5a  the valid relation separates the strategies: PS-pass(CAREFUL) - PS-pass(GULLIBLE) >= 0.15 on REAL for
       >= 2 of 3 models.
  H5b  the published relation does not separate them in the right direction: for >= 2 of 3 models,
       CN-viol(GULLIBLE) <= CN-viol(CAREFUL) + 0.05, i.e. the gullible strategy looks at least as good,
       while CN-adopt(GULLIBLE) - CN-adopt(CAREFUL) >= 0.15 (it adopts undated misinformation more).
  H5c  prompt-induced listing: for qwen2.5:14b, ACK(ONE) <= ACK(BASE) - 0.30 on REAL date-swap pairs.
  Adequacy: p15 adequacy carries over (same contexts); no new adequacy prompt.

    .venv/bin/python research/staleness/pretests/p16_strategies.py --dry-run
    .venv/bin/python research/staleness/pretests/p16_strategies.py --pilot
    .venv/bin/python research/staleness/pretests/p16_strategies.py --main
"""
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARGV = sys.argv[1:]
MODE = "pilot" if "--pilot" in ARGV else "main" if "--main" in ARGV else "dry"
sys.argv = sys.argv[:1] + ["--dry-run"]
import p15_provenance_swap as P15  # noqa: E402

P9, data, wilson, ps_label = P15.P9, P15.data, P15.wilson, P15.ps_label
OUT = data.CACHE / "p16"
TAIL_ONE = ("Fill in the blank (___) in the statement below with exactly one number taken from the context. "
            "Reply with only that single number. If the context does not contain it, reply: unknown\n\nStatement: {statement}")
TAIL_CAREFUL = ("Fill in the blank (___) in the statement below using the context. Some passages may conflict. "
                "If conflicting passages carry dates ('as of ...'), use the value from the passage with the most recent "
                "date and reply with only that number. If conflicting values have no dates, reply with all conflicting "
                "values separated by ' or '. If the context does not contain it, reply: unknown\n\nStatement: {statement}")
TAIL_GULLIBLE = ("Fill in the blank (___) in the statement below using the context. If passages conflict, the passage "
                 "listed later is a correction of the earlier one: use the value from the passage listed later. Reply "
                 "with only the number. If the context does not contain it, reply: unknown\n\nStatement: {statement}")


def template(strategy):
    if strategy == "BASE":
        return P9.TEMPLATE
    tail = {"ONE": TAIL_ONE, "CAREFUL": TAIL_CAREFUL, "GULLIBLE": TAIL_GULLIBLE}[strategy]
    return "Context:\n{context}\n\n" + tail


def prompts(c, strategy):
    lab = lambda t, d: f"(as of {d}) {t}" if d else t  # noqa: E731
    T = template(strategy)
    out = {}
    for src, orig, mut, conds in (("REAL", c["old_p"], c["new_p"], ("CORRECT", "SWAPPED")),
                                  ("CN", c["new_p"], c["cf_p"], ("NONE", "CORRECT", "SWAPPED"))):
        for dates in conds:
            do, dm = {"NONE": (None, None), "CORRECT": (c["d_early"], c["d_late"]), "SWAPPED": (c["d_late"], c["d_early"])}[dates]
            po, pm = lab(orig, do), lab(mut, dm)
            for order, ps in (("M-LAST", [po, pm, c["dis_p"]]), ("M-FIRST", [pm, po, c["dis_p"]])):
                ctx = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(ps))
                out[(src, dates, order)] = T.format(context=ctx, statement=c["statement"])
    return out


def units():
    cases, _ = P15.build()
    ids = sorted(c["q"] for c in cases)
    pilot = set(random.Random(1616).sample(ids, 20))
    rest = [i for i in ids if i not in pilot]
    main = set(random.Random(1617).sample(rest, 200))
    return cases, pilot, main


def scores(recs, gen, strategy):
    R = {(r["q"], r["src"], r["dates"], r["order"]): r["label"] for r in recs if r["gen"] == gen and r["strategy"] == strategy}
    qs = sorted({k[0] for k in R})
    out = {}
    for src in ("REAL", "CN"):
        cnt = Counter(ps_label(R[(q, src, "CORRECT", o)], R[(q, src, "SWAPPED", o)]) for q in qs for o in ("M-LAST", "M-FIRST"))
        n = sum(cnt.values())
        out[src] = {"pass": cnt["PASS"] / n, "viol": (cnt["INSENSITIVE"] + cnt["INVERTED"]) / n, "ack": cnt["ACK"] / n, "n": n,
                    "pass_k": cnt["PASS"], "viol_k": cnt["INSENSITIVE"] + cnt["INVERTED"]}
    none = [R[(q, "CN", "NONE", o)] for q in qs for o in ("M-LAST", "M-FIRST")]
    ml = [R[(q, "CN", "NONE", "M-LAST")] for q in qs]
    out["CN_viol"] = sum(l == "old" for l in ml) / len(ml)          # published relation, ASE placement (M-LAST)
    out["CN_adopt"] = sum(l == "new" for l in none) / len(none)     # adopts undated injected value (both orders)
    out["n_q"] = len(qs)
    return out


def run(cases, chosen, gens, strategies):
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for gen in gens:
        for c in cases:
            if c["q"] not in chosen:
                continue
            for s in strategies:
                for (src, dates, order), pr in prompts(c, s).items():
                    out = P9.generate(gen, pr)
                    orig, mut = (c["old"], c["new"]) if src == "REAL" else (c["new"], c["inj"])
                    recs.append({"gen": gen, "strategy": s, "q": c["q"], "page": c["page"], "src": src,
                                 "dates": dates, "order": order, "output": out, "label": P9.commit(mut, orig, out)})
        subprocess.run(["ollama", "stop", gen], check=False)
        print(f"{gen} done ({len(recs)} records)", flush=True)
    return recs


def report(recs, strategies):
    for gen in dict.fromkeys(r["gen"] for r in recs):
        print(f"\n===== {gen}")
        print(f"  {'strategy':<9} {'REAL PS-pass':>12} {'REAL PS-viol':>12} {'REAL ACK':>9} {'CN PS-pass':>11} {'CN-viol(pub)':>13} {'CN-adopt':>9}  n_q")
        for s in strategies:
            x = scores(recs, gen, s)
            print(f"  {s:<9} {x['REAL']['pass']:>12.3f} {x['REAL']['viol']:>12.3f} {x['REAL']['ack']:>9.3f} "
                  f"{x['CN']['pass']:>11.3f} {x['CN_viol']:>13.3f} {x['CN_adopt']:>9.3f}  {x['n_q']}")


def main():
    cases, pilot, mainq = units()
    strategies = ["BASE", "ONE", "CAREFUL", "GULLIBLE"]
    print(f"questions: pilot {len(pilot)}, main {len(mainq)} (disjoint: {not (pilot & mainq)}); "
          f"prompts per question per strategy: {len(prompts(cases[0], 'BASE'))}", flush=True)
    if MODE == "dry":
        c = next(c for c in cases if c["q"] in pilot)
        for s in strategies:
            k = ("REAL", "CORRECT", "M-LAST")
            print(f"\n### {s} {k}\n" + prompts(c, s)[k])
        return
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("script sha256", me, flush=True)
    if MODE == "pilot":
        recs = run(cases, pilot, ["qwen2.5:7b-instruct"], strategies)
        (OUT / "pilot_records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        report(recs, strategies)
        x_c, x_g = scores(recs, "qwen2.5:7b-instruct", "CAREFUL"), scores(recs, "qwen2.5:7b-instruct", "GULLIBLE")
        d = x_c["REAL"]["pass"] - x_g["REAL"]["pass"]
        print(f"\nPILOT GATE: PS-pass(CAREFUL) - PS-pass(GULLIBLE) on REAL = {d:+.3f} -> {'GO' if d >= 0.15 else 'STOP'}")
        return
    gens = ["phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]
    recs = run(cases, mainq, gens, strategies)
    (OUT / "main_records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    report(recs, strategies)
    S = {g: {s: scores(recs, g, s) for s in strategies} for g in gens}
    h5a = sum(S[g]["CAREFUL"]["REAL"]["pass"] - S[g]["GULLIBLE"]["REAL"]["pass"] >= 0.15 for g in gens)
    h5b = sum((S[g]["GULLIBLE"]["CN_viol"] <= S[g]["CAREFUL"]["CN_viol"] + 0.05)
              and (S[g]["GULLIBLE"]["CN_adopt"] - S[g]["CAREFUL"]["CN_adopt"] >= 0.15) for g in gens)
    a1, a0 = S["qwen2.5:14b-instruct"]["ONE"]["REAL"]["ack"], S["qwen2.5:14b-instruct"]["BASE"]["REAL"]["ack"]
    print(f"\nH5a valid relation separates CAREFUL from GULLIBLE (>=0.15) in {h5a}/3 models -> {'SUPPORTED' if h5a >= 2 else 'NOT supported'}")
    print(f"H5b published relation fails to penalise GULLIBLE while it adopts misinformation in {h5b}/3 models -> {'SUPPORTED' if h5b >= 2 else 'NOT supported'}")
    print(f"H5c 14B ACK on REAL: BASE {a0:.3f} -> ONE {a1:.3f} (drop {a0 - a1:+.3f}) -> {'SUPPORTED (listing is prompt-induced)' if a0 - a1 >= 0.30 else 'NOT supported'}")


if __name__ == "__main__":
    main()
