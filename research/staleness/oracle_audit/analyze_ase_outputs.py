"""Impact of the CN-oracle defects on the ASE 2026 authors' OWN released outputs (GPT-5-mini).

PRE-REGISTERED 2026-10-13, before the authors' output archive was downloaded or opened.  The sha256
of this file is logged in `prereg.log` before the first real run.  Fixes to the *loader* only
(file names, field names) after unzipping are allowed and logged as amendments; analysis
definitions and thresholds are not changed after data are seen.

Data      Figshare archive of github.com/dbr7/RAG-metamorphic-mutation (commit 597fd6582b), output/:
          step2_baseline_rag_results*.json (ground truth), step3_mutation_results*.json,
          step3_metamorphic_verdicts*.json, step3_gt_validation*.json, and step4 / RQ3 repair files
          if present.  Context-mode runs (`*_context_*`), as in the paper.
Units     CN mutants: mutation_operator starting with "counterfactual_noise"; per dataset x scope
          (pre_chunk, post_chunk).  Unit of analysis = (question_id, scope).
Subsets   ALTERING = is_semantics_preserving is False.
          LOOKUP   = numeric datasets only: the ground-truth value (tolerance 1%) appears as a number in
                     the baseline retrieved content (a lookup, not a computed answer).  Value-alignment
                     measures are defined only on LOOKUP, because a computed answer that correctly
                     uses the counterfactual need not contain any counterfactual number.
Measures (each reported with Wilson 95% CI; all reported whichever way they go)
  M1  share of ALTERING VIOLATED verdicts whose mutated answer matches the ground truth
      (numeric: their numeric_answers_match, tol 1%; RepliQA: their gt_correct == "correct").
  M2  [LOOKUP, ALTERING, SATISFIED, non-abstaining] share whose answer matches NEITHER the ground
      truth NOR any number in the generated counterfactual text (tol 1%) = passes that cannot be
      explained as following the injected fact.
  M3  [pre_chunk, ALTERING, LOOKUP, counterfactual text and ground-truth value both present in the
      retrieved mutated context] VIOLATED rate when the counterfactual text starts BEFORE the
      first occurrence of the ground-truth value vs AFTER it.  The insertion position is random
      (mutation_operators.py L602), so this is a randomized comparison.  Fisher exact test.
  M4  [LOOKUP, ALTERING] value-aligned re-scoring: FOLLOWED (answer matches a counterfactual number
      and not the GT) / KEPT (matches GT only) / ACK (abstains, or matches both) / OTHER (neither).
      Their verdict vs ours: SATISFIED<->FOLLOWED|ACK, VIOLATED<->KEPT; report agreement and the
      cross-table, and the share of their SATISFIED that are OTHER.
  M5  if repair files exist: share of CN "repaired" cases (non-preserving) whose repaired answer
      matches neither the GT nor a counterfactual number (LOOKUP only).
Amendment 1 (2026-10-13, after seeing only file/field NAMES, before any result): M5 was defined above
          but not implemented; implemented exactly as defined.  No definition or threshold changed.
Decision rules (pre-registered)
  H-M2 "material hallucinated passes"  if the pooled M2 >= 0.10 and the lower CI > 0.05.
  H-M3 "placement-dependent verdicts"  if the pooled |delta| >= 0.05 with Fisher p < 0.05.
  H-M4 "verdicts change under a value-aligned oracle" if pooled disagreement >= 0.10.
  If n < 30 in a cell, that cell is reported as descriptive only.

    .venv/bin/python research/staleness/oracle_audit/analyze_ase_outputs.py --root ~/SemProject/ase-replication/output
"""
import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOL = 0.01
NUMERIC = ("ConvFinQA", "FinQA", "TAT-DQA", "VQAonBD")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_numeric_answer(text):            # verbatim ASE pipeline_config.py L752-771
    if not text or text.strip().upper() in ("N/A", "NONE", "NULL", ""):
        return None
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    cleaned = cleaned.replace("million", "").replace("billion", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        pass
    m = re.findall(r"-?\d+\.?\d*", cleaned)
    return float(m[-1]) if m else None


def close(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6 if b == 0 else abs(a - b) / abs(b) <= TOL


def numbers(text):
    out = []
    for x in NUM.findall(text or ""):
        try:
            out.append(float(x.replace(",", "")))
        except ValueError:
            pass
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def fisher(a, b, c, d):                    # two-sided Fisher exact for [[a,b],[c,d]]
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 + 1e-12))


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = json.load(open(f))
        recs += d if isinstance(d, list) else d.get("results", d.get("records", []))
    return recs


def first_pos(text, value):
    """character offset of the first number in text equal to value (tol 1%), else None"""
    for m in NUM.finditer(text or ""):
        try:
            if close(float(m.group().replace(",", "")), value):
                return m.start()
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    print("script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    pooled = defaultdict(lambda: Counter())
    m5 = Counter()
    for ds in ("repliqa",) + NUMERIC:
        base = {b["question_id"]: b for b in load(str(root / "**" / ds / "step2_baseline_rag_results*.json"))}
        for scope in ("pre_chunk", "post_chunk"):
            run = f"{root}/**/{ds}/{scope}_context*"
            muts = [m for m in load(run + "/step3_mutation_results*.json") if str(m.get("mutation_operator", "")).startswith("counterfactual_noise")]
            ver = {(v["question_id"], v["mutation_operator"]): v for v in load(run + "/step3_metamorphic_verdicts*.json")}
            gtv = {(g["question_id"], g["mutation_operator"]): g for g in load(run + "/step3_gt_validation*.json")}
            if not muts:
                print(f"\n[{ds} {scope}] no CN mutation records found under {run}")
                continue
            c = Counter()
            m3 = Counter()
            m4 = Counter()
            for m in muts:
                key = (m["question_id"], m["mutation_operator"])
                v = ver.get(key)
                if v is None or m.get("is_semantics_preserving") is not False:
                    c["preserving_or_unjudged"] += 1
                    continue
                c["altering"] += 1
                b = base.get(m["question_id"])
                verdict = v["verdict"]
                ans = m.get("mutated_answer", "")
                cf = " ".join(str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", []))
                if verdict == "VIOLATED":
                    c["violated"] += 1
                    if ds in NUMERIC and b:
                        c["M1_viol_gt"] += close(parse_numeric_answer(ans), float(b["ground_truth_short"]))
                    elif key in gtv:
                        c["M1_viol_gt"] += gtv[key].get("gt_correct") == "correct"
                if ds not in NUMERIC or not b:
                    continue
                gt = float(b["ground_truth_short"])
                ctx0 = " ".join(b.get("retrieved_content", []))
                if first_pos(ctx0, gt) is None:
                    continue
                c["lookup"] += 1
                pa = parse_numeric_answer(ans)
                abst = ans.strip().upper() == "N/A" or pa is None
                hit_gt = close(pa, gt)
                hit_cf = any(close(pa, x) for x in numbers(cf)) and not hit_gt
                cls = "ACK" if abst else "KEPT" if hit_gt else "FOLLOWED" if hit_cf else "OTHER"
                m4[(verdict, cls)] += 1
                if verdict == "SATISFIED" and not abst:
                    c["M2_n"] += 1
                    c["M2_other"] += cls == "OTHER"
                if scope == "pre_chunk":
                    ctx1 = " ".join(m.get("retrieved_content_mutated", []))
                    i_cf = ctx1.find(cf.strip()[:60]) if cf.strip() else -1
                    i_gt = first_pos(ctx1, gt)
                    if i_cf >= 0 and i_gt is not None:
                        side = "CF_before" if i_cf < i_gt else "CF_after"
                        m3[(side, verdict == "VIOLATED")] += 1
            # ---- M5 (pre-registered definition; implemented 2026-10-13 as Amendment 1, before any run)
            cfmap = {(m["question_id"], m["mutation_operator"]): " ".join(str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", [])) for m in muts}
            for r in load(run + "/step4_fault_repair*.json"):
                if not str(r.get("mutation_operator", "")).startswith("counterfactual_noise") or r.get("is_semantics_preserving") is not False:
                    continue
                b = base.get(r["question_id"])
                if ds not in NUMERIC or not b or not r.get("is_repaired"):
                    continue
                gt = float(b["ground_truth_short"])
                if first_pos(" ".join(b.get("retrieved_content", [])), gt) is None:
                    continue
                pa = parse_numeric_answer(r.get("repaired_answer", ""))
                if pa is None or str(r.get("repaired_answer", "")).strip().upper() == "N/A":
                    m5["repaired_abstain"] += 1
                    continue
                cfn = numbers(cfmap.get((r["question_id"], r["mutation_operator"]), ""))
                m5["repaired_n"] += 1
                m5["repaired_other"] += (not close(pa, gt)) and not any(close(pa, x) for x in cfn)
            pooled["c"].update(c)
            pooled["m3"].update(m3)
            pooled["m4"].update(m4)
            print(f"\n[{ds} {scope}] CN records {len(muts)}; {dict(c)}")
            if c["violated"]:
                print(f"  M1 violated answers matching GT: {c['M1_viol_gt']}/{c['violated']} {wilson(c['M1_viol_gt'], c['violated'])}")
            if c["M2_n"]:
                print(f"  M2 satisfied (lookup) matching neither GT nor CF: {c['M2_other']}/{c['M2_n']} {wilson(c['M2_other'], c['M2_n'])}")
            if m3:
                print(f"  M3 {dict(m3)}")
            if m4:
                print(f"  M4 {dict(m4)}")
    c, m3, m4 = pooled["c"], pooled["m3"], pooled["m4"]
    print("\n================ POOLED (numeric datasets for M2-M4)")
    if c["violated"]:
        print(f"M1 {c['M1_viol_gt']}/{c['violated']} = {c['M1_viol_gt']/c['violated']:.3f} {wilson(c['M1_viol_gt'], c['violated'])}")
    if c["M2_n"]:
        p, lo = c["M2_other"] / c["M2_n"], wilson(c["M2_other"], c["M2_n"])[0]
        print(f"M2 {c['M2_other']}/{c['M2_n']} = {p:.3f} {wilson(c['M2_other'], c['M2_n'])} -> H-M2 {'SUPPORTED' if p >= 0.10 and lo > 0.05 else 'not supported'}"
              + ("  (descriptive: n<30)" if c["M2_n"] < 30 else ""))
    nb, vb = m3[("CF_before", True)], m3[("CF_before", True)] + m3[("CF_before", False)]
    na, va = m3[("CF_after", True)], m3[("CF_after", True)] + m3[("CF_after", False)]
    if vb and va:
        d = nb / vb - na / va
        pf = fisher(nb, vb - nb, na, va - na)
        print(f"M3 violated: CF_before {nb}/{vb} = {nb/vb:.3f} {wilson(nb, vb)}  CF_after {na}/{va} = {na/va:.3f} {wilson(na, va)}  "
              f"delta {d:+.3f} Fisher p={pf:.3g} -> H-M3 {'SUPPORTED' if abs(d) >= 0.05 and pf < 0.05 else 'not supported'}"
              + ("  (descriptive: a cell n<30)" if min(vb, va) < 30 else ""))
    if m5["repaired_n"]:
        print(f"M5 CN 'repaired' (lookup, non-abstaining) matching neither GT nor CF: {m5['repaired_other']}/{m5['repaired_n']} = "
              f"{m5['repaired_other']/m5['repaired_n']:.3f} {wilson(m5['repaired_other'], m5['repaired_n'])}; abstentions counted as repaired: {m5['repaired_abstain']}")
    tot = sum(m4.values())
    if tot:
        agree = sum(v for (vd, cl), v in m4.items() if (vd == "SATISFIED" and cl in ("FOLLOWED", "ACK")) or (vd == "VIOLATED" and cl == "KEPT"))
        print(f"M4 agreement {agree}/{tot} = {agree/tot:.3f}; disagreement {1-agree/tot:.3f} -> H-M4 {'SUPPORTED' if 1-agree/tot >= 0.10 else 'not supported'}")
        print("   cross-table:", dict(m4))


if __name__ == "__main__":
    main()
