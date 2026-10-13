"""POST HOC (descriptive): quality of the authors' generated counterfactuals.
REFUSAL  = the LLM refused instead of writing a counterfactual (pattern list below; printed for checking).
FALLBACK = pre-chunk counterfactual longer than 50% of the source document (the heuristic fallback
           rewrites the whole document; LLM output is 1-2 sentences, mutation_operators.py L551-574)."""
import json, glob, re, sys
from collections import Counter
from pathlib import Path
root = Path("~/SemProject/ase-replication/output").expanduser()
REF = re.compile(r"\b(I can(?:no|')t|I cannot|I'm (?:sorry|unable)|I am (?:sorry|unable)|Sorry,|can't help|cannot help|unable to (?:create|generate|provide|help)|not able to (?:create|generate))", re.I)
tot = Counter(); ex = []
for ds in ("repliqa", "ConvFinQA", "FinQA", "TAT-DQA", "VQAonBD"):
    docs = {}
    for f in glob.glob(str(root / ds / "step2_filtered_entries*.json")):
        for e in json.load(open(f)):
            docs[e["question_id"]] = e.get("document_extracted", "")
    vers = {}
    for scope in ("pre_chunk", "post_chunk"):
        for f in glob.glob(str(root / ds / f"{scope}_context*" / "step3_metamorphic_verdicts*.json")):
            for v in json.load(open(f)):
                vers[(scope, v["question_id"], v["mutation_operator"])] = v["verdict"]
        for f in sorted(glob.glob(str(root / ds / f"{scope}_context*" / "step3_mutation_results*.json"))):
            for m in json.load(open(f)):
                if not m["mutation_operator"].startswith("counterfactual_noise"): continue
                cfs = [str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", [])]
                cf = " ".join(cfs)
                k = (ds, scope)
                tot[k + ("n",)] += 1
                ref = bool(REF.search(cf))
                fb = scope == "pre_chunk" and docs.get(m["question_id"]) and len(cf) > 0.5 * len(docs[m["question_id"]])
                tot[k + ("refusal",)] += ref
                tot[k + ("fallback",)] += bool(fb)
                v = vers.get((scope, m["question_id"], m["mutation_operator"]))
                if ref:
                    tot[k + ("refusal_altering",)] += m.get("is_semantics_preserving") is False
                    tot[k + ("refusal_violated",)] += v == "VIOLATED"
                    if len(ex) < 8: ex.append((ds, scope, cf[:160]))
print(f"{'dataset':<10}{'scope':<11}{'CN n':>6}{'refusal':>9}{' (altering)':>12}{' (violated)':>12}{'fallback':>10}")
for ds in ("repliqa", "ConvFinQA", "FinQA", "TAT-DQA", "VQAonBD"):
    for sc in ("pre_chunk", "post_chunk"):
        k = (ds, sc)
        print(f"{ds:<10}{sc:<11}{tot[k+('n',)]:>6}{tot[k+('refusal',)]:>9}{tot[k+('refusal_altering',)]:>12}{tot[k+('refusal_violated',)]:>12}{tot[k+('fallback',)]:>10}")
N = sum(v for k, v in tot.items() if k[-1] == "n"); R = sum(v for k, v in tot.items() if k[-1] == "refusal")
F = sum(v for k, v in tot.items() if k[-1] == "fallback")
print(f"\nTOTAL CN mutants {N}; refusals {R} ({R/N:.3f}); pre-chunk fallbacks {F}")
print("\nrefusal examples (check the pattern is right):")
for e in ex: print("  ", e)
