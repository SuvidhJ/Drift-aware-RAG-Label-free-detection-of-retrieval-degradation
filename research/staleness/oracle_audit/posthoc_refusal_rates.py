"""POST HOC: precision check of the refusal pattern + CN violation rates with refusal mutants excluded."""
import json, glob, random
exec(open("posthoc_cf_quality.py").read().split("tot = Counter()")[0])
rng = random.Random(3); hits = []; rows = []
for ds in ("repliqa", "ConvFinQA", "FinQA", "TAT-DQA", "VQAonBD"):
    for scope in ("pre_chunk", "post_chunk"):
        vers = {}
        for f in glob.glob(str(root / ds / f"{scope}_context*" / "step3_metamorphic_verdicts*.json")):
            for v in json.load(open(f)):
                vers[(v["question_id"], v["mutation_operator"])] = v["verdict"]
        n = viol = n_ok = viol_ok = 0
        for f in sorted(glob.glob(str(root / ds / f"{scope}_context*" / "step3_mutation_results*.json"))):
            for m in json.load(open(f)):
                if not m["mutation_operator"].startswith("counterfactual_noise"): continue
                v = vers.get((m["question_id"], m["mutation_operator"]))
                if v is None: continue
                cf = " ".join(str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", []))
                r = bool(REF.search(cf)); n += 1; viol += v == "VIOLATED"
                if r: hits.append((ds, scope, cf[:150]))
                else: n_ok += 1; viol_ok += v == "VIOLATED"
        rows.append((ds, scope, viol, n, viol_ok, n_ok))
for ds, sc, a, b, c, d in rows:
    print(f"{ds:<10}{sc:<11} published {a}/{b} = {a/b:.3f}   excluding refusals {c}/{d} = {c/d:.3f}   ({b-d} refusals)")
print("\n20 random pattern matches from numeric datasets (precision check):")
num = [h for h in hits if h[0] != "repliqa"]
for h in rng.sample(num, 20): print("  ", h)
