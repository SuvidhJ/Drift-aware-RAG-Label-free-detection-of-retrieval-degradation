"""POST HOC sanity checks on the authors' outputs (not pre-registered; descriptive).
(1) Reproduce Table 1 CN violation counts from their verdict files (loader completeness).
(2) Share of published CN violations that come from the semantics-altering branch.
(3) Print random samples of M2 'OTHER' passes and M5 'OTHER' repairs for hand reading."""
import random, sys
sys.argv = [sys.argv[0], "--root", "x"]
import importlib.util
spec = importlib.util.spec_from_file_location("a", "analyze_ase_outputs.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
from pathlib import Path
root = Path("~/SemProject/ase-replication/output").expanduser()
TABLE1 = {("repliqa","pre_chunk"):(25,217),("repliqa","post_chunk"):(54,300),("ConvFinQA","pre_chunk"):(45,228),("ConvFinQA","post_chunk"):(70,300),
          ("FinQA","pre_chunk"):(37,205),("FinQA","post_chunk"):(78,300),("TAT-DQA","pre_chunk"):(57,231),("TAT-DQA","post_chunk"):(67,300),
          ("VQAonBD","pre_chunk"):(80,290),("VQAonBD","post_chunk"):(81,300)}
oth2, oth5 = [], []
tv = ta = 0
for ds in ("repliqa",) + A.NUMERIC:
    base = {b["question_id"]: b for b in A.load(str(root / "**" / ds / "step2_baseline_rag_results*.json"))}
    for scope in ("pre_chunk", "post_chunk"):
        run = f"{root}/**/{ds}/{scope}_context*"
        muts = {(m["question_id"], m["mutation_operator"]): m for m in A.load(run + "/step3_mutation_results*.json") if m["mutation_operator"].startswith("counterfactual_noise")}
        ver = [v for v in A.load(run + "/step3_metamorphic_verdicts*.json") if v["mutation_operator"].startswith("counterfactual_noise")]
        nv = sum(v["verdict"] == "VIOLATED" for v in ver)
        alt = sum(v["verdict"] == "VIOLATED" and muts.get((v["question_id"], v["mutation_operator"]), {}).get("is_semantics_preserving") is False for v in ver)
        tv += nv; ta += alt
        t = TABLE1[(ds, scope)]
        print(f"{ds:<9} {scope:<10} recount {nv}/{len(ver)} vs Table 1 {t[0]}/{t[1]} -> {'MATCH' if (nv, len(ver)) == t else 'DIFF'};  violations from ALTERING branch {alt}/{nv}")
        if ds not in A.NUMERIC: continue
        vmap = {(v["question_id"], v["mutation_operator"]): v for v in ver}
        for k, m in muts.items():
            b = base.get(m["question_id"]); v = vmap.get(k)
            if not b or not v or m.get("is_semantics_preserving") is not False: continue
            gt = float(b["ground_truth_short"])
            if A.first_pos(" ".join(b.get("retrieved_content", [])), gt) is None: continue
            cf = " ".join(str(l.get("generated_counterfactual", "")) for l in m["mutation_log"])
            pa = A.parse_numeric_answer(m["mutated_answer"])
            if v["verdict"] == "SATISFIED" and pa is not None and not A.close(pa, gt) and not any(A.close(pa, x) for x in A.numbers(cf)):
                oth2.append((ds, scope, m["question"], gt, cf, m["mutated_answer"]))
        for r in A.load(run + "/step4_fault_repair*.json"):
            if not r["mutation_operator"].startswith("counterfactual_noise") or r.get("is_semantics_preserving") is not False or not r.get("is_repaired"): continue
            b = base.get(r["question_id"])
            if not b: continue
            gt = float(b["ground_truth_short"])
            if A.first_pos(" ".join(b.get("retrieved_content", [])), gt) is None: continue
            cf = " ".join(str(l.get("generated_counterfactual", "")) for l in muts.get((r["question_id"], r["mutation_operator"]), {}).get("mutation_log", []))
            pa = A.parse_numeric_answer(r["repaired_answer"])
            if pa is not None and not A.close(pa, gt) and not any(A.close(pa, x) for x in A.numbers(cf)):
                oth5.append((ds, scope, r["question_id"], gt, cf, r["repaired_answer"], r.get("repair_config")))
print(f"\nALL published CN violations: {tv}; from the altering branch: {ta} = {ta/tv:.3f}")
rng = random.Random(7)
print("\n#### M2 sample: their SATISFIED, answer matches neither GT nor CF")
for ds, sc, q, gt, cf, ans in rng.sample(oth2, min(12, len(oth2))):
    print(f"[{ds} {sc}] Q: {q[:150]}\n   GT={gt} | answer={ans!r}\n   CF: {cf[:260]}")
print("\n#### M5 sample: their 'repaired', answer matches neither GT nor CF")
for ds, sc, qid, gt, cf, ans, cfg in rng.sample(oth5, min(10, len(oth5))):
    print(f"[{ds} {sc}] cfg={cfg} GT={gt} | repaired answer={ans!r}\n   CF: {cf[:220]}")
