"""Verify, from the ASE authors' released outputs, how their own showcase examples (paper §6.1 ZenSpace Yoga; §6.3
InspirePay 'currency conversion') were scored by their pipeline.  Run with the analysis environment."""
import glob, json
ROOT = "/Users/aditisaxena/SemProject/ase-replication/output/repliqa"
def load(p):
    out = []
    for f in sorted(glob.glob(p)): out += json.load(open(f))
    return out
for scope, needle in (("pre_chunk", "ZenSpace"), ("post_chunk", "InspirePay")):
    run = f"{ROOT}/{scope}_context_llm_semantic"
    muts = {(m["question_id"], m["mutation_operator"]): m for m in load(run + "/step3_mutation_results*.json")}
    vers = {(v["question_id"], v["mutation_operator"]): v for v in load(run + "/step3_metamorphic_verdicts*.json")}
    gts = {(g["question_id"], g["mutation_operator"]): g for g in load(run + "/step3_gt_validation*.json")}
    reps = {(r["question_id"], r["mutation_operator"]): r for r in load(run + "/step4_fault_repair*.json")}
    for k, m in muts.items():
        if needle in m["question"] and m["mutation_operator"].startswith("counterfactual_noise") and vers.get(k, {}).get("verdict") == "VIOLATED":
            r = reps.get(k, {})
            print(f"[{scope}] {m['question'][:90]}")
            print(f"  altering={m.get('is_semantics_preserving') is False}  verdict={vers[k]['verdict']}  gt={gts.get(k, {}).get('gt_correct')}  class={gts.get(k, {}).get('mr_classification')}")
            print(f"  original={m['original_answer'][:90]!r}\n  mutated ={m['mutated_answer'][:90]!r}")
            if r: print(f"  repair[{r.get('repair_config')}] -> {str(r.get('repaired_answer'))[:110]!r}  repaired_gt={r.get('repaired_gt_correct')}  is_repaired={r.get('is_repaired')}")
