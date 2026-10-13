"""Pre-test 14 (EXPLORATORY, randomized): is the ASE 2026 metamorphic oracle for knowledge-conflict
mutations order-dependent?

Pre-registered 2026-10-13, before any p14 generation.  The sha256 of this file is recorded in
`p14_prereg.log` before the run; the first line of `p14_order_oracle.log` repeats it.  (The old p14 —
a prompt-only test — was dropped on 2026-10-13 after the critical review; this replaces it.)

Background (observational, post hoc, `verification_2026-10-13/g_aseoracle.py`): applying ASE's released
numeric oracle verbatim to p11/p12 records, the violation rate for Qwen-7B CN was 0.705 when the
mutation passage was retrieved BEFORE the original and 0.220 when AFTER.  ASE's post-chunk CN always
appends the counterfactual AFTER the original (mutation_operators.py L651).  Retrieval order was not
randomized, so this needs a randomized test.

Hypothesis H1: the ASE oracle's violation rate depends on the order of the two conflicting passages.
Units    120 distinct questions, seeded sample (random.Random(14)) of the 400-question p11 sample, keeping
         questions whose counterfactual (p12, qwen2.5:7b) has a fresh non-year value and did not use the
         x1.13 fallback.
Contexts constructed (ASE post-chunk scope; no retrieval): three passages
         REAL  orig = old-revision field passage,  mut = new-revision field passage
         CN    orig = current field passage,        mut = p12 counterfactual passage
         order M-LAST  = [orig, mut, distractor]   (ASE's post-chunk placement)
         order M-FIRST = [mut, orig, distractor]
         distractor = the first other numeric field passage of the same page (current revision), the
         same in every condition.  ADEQUACY control: [mut(REAL new), distractor].
Prompt   p9 TEMPLATE unchanged; temperature 0; num_predict 24 (as p11/p12).
Models   phi3:mini, qwen2.5:7b-instruct, one at a time (14B only if the user approves later).
Scoring  (a) ASE verbatim: parse_numeric_answer / numeric_answers_match (pipeline_config.py L752-789),
             semantics-altering branch: VIOLATED iff the answer matches the ORIGINAL value (tol 0.01);
         (b) P9.commit four-way label (new/old/both/neither) for description.
Primary  per model x source: delta = viol(M-FIRST) - viol(M-LAST); paired exact McNemar over questions.
Adequacy ADEQUACY control: P9.commit == 'new' (new value alone) >= 0.60 per model, else that model is VOID.
Decision H1 SUPPORTED if |delta| >= 0.10 with McNemar p < 0.01 for at least one model x source with valid
         adequacy; NOT SUPPORTED if |delta| < 0.05 for every model x source; otherwise INCONCLUSIVE.
         The order claim of the planned paper is dropped unless H1 is SUPPORTED.
All numbers reported whichever way they go; this is a pre-test, labelled exploratory.

AMENDMENT 1 (2026-10-13, BEFORE ANY p14 GENERATION; original hash 1fb7d9b8... in p14_prereg.log):
         The primary measure is changed from the ASE-parser verdict to the COMMITTED VALUE.
         Reason: ASE applies parse_numeric_answer only to a single JSON `computed_formula` value
         (pipeline_config.py L819-833), so applying it to our free-text answers, which may list
         both values, mixes a parser artefact (last number wins) with the order effect.
         PRIMARY:   orig_only = P9.commit(mut, orig, answer) == 'old' (commits to the ORIGINAL
                    value alone); delta = orig_only(M-FIRST) - orig_only(M-LAST); paired exact
                    McNemar over questions.  Decision thresholds unchanged (|delta| >= 0.10 with
                    p < 0.01 -> SUPPORTED; all |delta| < 0.05 -> NOT SUPPORTED; else INCONCLUSIVE).
         SECONDARY (descriptive only): the ASE-parser verdict, and the full four-way distribution.

    .venv/bin/python research/staleness/pretests/p14_order_oracle.py --dry-run
    .venv/bin/python research/staleness/pretests/p14_order_oracle.py
"""
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
DRY = "--dry-run" in sys.argv
sys.argv = sys.argv[:1] + ["stats"]
import p9_wiki_infobox as P9  # noqa: E402
import p12_counterfactual as P12  # noqa: E402
from research.staleness import data, facts  # noqa: E402

GENS = ["phi3:mini", "qwen2.5:7b-instruct"]
N_Q = 120
OUT = data.CACHE / "p14"
YEAR = re.compile(r"(19|20)\d\d")


# ---- ASE oracle, verbatim from github.com/dbr7/RAG-metamorphic-mutation pipeline_config.py L752-789
def parse_numeric_answer(text):
    if not text or text.strip().upper() in ("N/A", "NONE", "NULL", ""):
        return None
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    cleaned = cleaned.replace("million", "").replace("billion", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        pass
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if matches:
        return float(matches[-1])
    return None


def numeric_answers_match(pred, ground_truth, tolerance=0.01):
    p = parse_numeric_answer(pred)
    if p is None:
        return False
    if ground_truth == 0.0:
        return abs(p - ground_truth) < 1e-6
    return abs(p - ground_truth) / abs(ground_truth) <= tolerance


def ase_violated(out, orig_value):          # step3 _check_mr_numeric, semantics-altering branch
    if out.strip().upper() == "N/A":
        return False
    return numeric_answers_match(out, orig_value)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def build():
    frozen = data.CACHE / "p11_wiki_frozen"
    P9.RAW = frozen
    pages = [json.loads(p.read_text()) for p in sorted(frozen.glob("*.json")) if p.name != "pages.json"]
    pages = [p for p in pages if not p.get("skip") and p.get("text0") and p.get("text1")]
    _, allq = P9.build()
    qs = random.Random(0).sample(allq, min(P9.SAMPLE, len(allq)))
    cases = []
    for n, q in enumerate(qs):
        p = pages[q["doc"]]
        f0, f1 = P9.fields(p["text0"]), P9.fields(p["text1"])
        fld = lambda f, k: f"{p['title']} -- {k.replace('_', ' ')}: {f[k]}."  # noqa: E731
        fact = fld(f1, q["key"])
        cf, cfv = P12.counterfactual(fact, {facts.norm(a) for a in q["new"] + q["old"]})
        llm_first_line = P9.generate(P12.CF_MODEL, P12.CF_PROMPT.format(fact=fact)).strip().splitlines()[0]
        fallback = cf != llm_first_line
        inj = [x for x in facts.NUM.findall(cf.split(":", 1)[-1]) if len(x) >= 3 and not YEAR.fullmatch(x)
               and facts.norm(x) not in {facts.norm(a) for a in q["new"] + q["old"]}]
        dis = [k for k in sorted(f1) if k != q["key"] and facts.NUM.search(f1[k])
               and not ({facts.norm(x) for x in facts.NUM.findall(f1[k])} & {facts.norm(a) for a in q["new"] + q["old"] + inj})]
        if fallback or not inj or not dis:
            continue
        cases.append({"q": n, "page": q["doc"], "statement": q["statement"], "new": q["new"], "old": q["old"],
                      "inj": inj, "old_p": fld(f0, q["key"]), "new_p": fact, "cf_p": cf, "dis_p": fld(f1, dis[0])})
    return random.Random(14).sample(cases, min(N_Q, len(cases))), len(cases)


def prompts(c):
    ctx = lambda ps: "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(ps))  # noqa: E731
    T = lambda ps: P9.TEMPLATE.format(context=ctx(ps), statement=c["statement"])  # noqa: E731
    return {("REAL", "M-LAST"): T([c["old_p"], c["new_p"], c["dis_p"]]),
            ("REAL", "M-FIRST"): T([c["new_p"], c["old_p"], c["dis_p"]]),
            ("CN", "M-LAST"): T([c["new_p"], c["cf_p"], c["dis_p"]]),
            ("CN", "M-FIRST"): T([c["cf_p"], c["new_p"], c["dis_p"]]),
            ("ADEQ", "-"): T([c["new_p"], c["dis_p"]])}


def main():
    cases, pool = build()
    print(f"eligible questions {pool}; sampled {len(cases)}; prompts per model {5 * len(cases)}", flush=True)
    if DRY:
        for k, v in prompts(cases[0]).items():
            print("\n###", k, "\n" + v)
        return
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"script sha256 {me}", flush=True)
    recs = []
    for gen in GENS:
        for c in cases:
            for (src, order), pr in prompts(c).items():
                out = P9.generate(gen, pr)
                orig, mut = (c["old"], c["new"]) if src == "REAL" else (c["new"], c["inj"])
                recs.append({"gen": gen, "q": c["q"], "page": c["page"], "src": src, "order": order, "output": out,
                             "ase_violated": ase_violated(out, float(facts.norm(orig[0]))) if src != "ADEQ" else None,
                             "commit": P9.commit(mut, orig, out) if src != "ADEQ" else P9.commit(c["new"], c["old"], out)})
        subprocess.run(["ollama", "stop", gen], check=False)
        print(f"{gen} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    verdicts = []
    for gen in GENS:
        ad = [r for r in recs if r["gen"] == gen and r["src"] == "ADEQ"]
        a = sum(r["commit"] == "new" for r in ad) / len(ad)
        ok = a >= 0.60
        print(f"\n{gen:<22} ADEQUACY new-only {a:.2f} -> {'ok' if ok else 'VOID'}")
        for src in ("REAL", "CN"):
            by = {(r["q"], r["order"]): r for r in recs if r["gen"] == gen and r["src"] == src}
            qids = sorted({q for q, _ in by})
            vl = [by[(q, "M-LAST")]["commit"] == "old" for q in qids]; vf = [by[(q, "M-FIRST")]["commit"] == "old" for q in qids]
            al = sum(by[(q, "M-LAST")]["ase_violated"] for q in qids) / len(qids); af = sum(by[(q, "M-FIRST")]["ase_violated"] for q in qids) / len(qids)
            b = sum(f and not l for f, l in zip(vf, vl)); c = sum(l and not f for f, l in zip(vf, vl))
            d = sum(vf) / len(qids) - sum(vl) / len(qids); p = mcnemar(b, c)
            dist = {o: {k: round(sum(by[(q, o)]["commit"] == k for q in qids) / len(qids), 3)
                        for k in ("new", "old", "both", "neither")} for o in ("M-LAST", "M-FIRST")}
            print(f"  {src:<5} ORIG-ONLY M-LAST {sum(vl)/len(qids):.3f}  M-FIRST {sum(vf)/len(qids):.3f}  "
                  f"delta {d:+.3f}  McNemar b={b} c={c} p={p:.2g}  | secondary ASE-parser violation {al:.3f}/{af:.3f} | four-way (mut=new) {dist}", flush=True)
            if ok:
                verdicts.append((abs(d), p))
    sup = any(d >= 0.10 and p < 0.01 for d, p in verdicts)
    nos = verdicts and all(d < 0.05 for d, _ in verdicts)
    print("\nH1:", "SUPPORTED" if sup else "NOT SUPPORTED" if nos else "INCONCLUSIVE")


if __name__ == "__main__":
    main()
