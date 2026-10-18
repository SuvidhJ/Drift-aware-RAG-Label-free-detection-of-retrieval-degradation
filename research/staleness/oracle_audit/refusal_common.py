"""Shared definition of the v2 refusal pattern (CLAIMS F5, corrected 2026-10-17). Also pinned in tests/test_oracle_audit.py."""
import glob, json, re
from pathlib import Path
ROOT = Path.home() / "SemProject/ase-replication/output"
A = "['’]"
REF2 = re.compile(rf"\b(I can(?:no|{A})t|I cannot|I{A}m (?:sorry|unable|not able)|I am (?:sorry|unable|not able)|Sorry[,.]|can{A}t (?:help|assist|create|provide|comply|do that|generate)|cannot (?:help|assist|create|provide|comply|generate)|unable to (?:create|generate|provide|help|assist|comply)|not able to (?:create|generate|provide|help)|I won{A}t|I will not)", re.I)
DATASETS = ("repliqa", "ConvFinQA", "FinQA", "TAT-DQA", "VQAonBD")

def cn_mutants(ds, scope):
    for f in sorted(glob.glob(str(ROOT / ds / f"{scope}_context*" / "step3_mutation_results*.json"))):
        for m in json.load(open(f)):
            if m["mutation_operator"].startswith("counterfactual_noise"):
                yield m, " ".join(str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", []))

def verdicts(ds, scope):
    v = {}
    for f in glob.glob(str(ROOT / ds / f"{scope}_context*" / "step3_metamorphic_verdicts*.json")):
        for x in json.load(open(f)): v[(x["question_id"], x["mutation_operator"])] = x["verdict"]
    return v
