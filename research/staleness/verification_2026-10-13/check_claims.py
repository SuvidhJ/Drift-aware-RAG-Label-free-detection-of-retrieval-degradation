"""Re-runnable check of every quote and code citation in CLAIMS_LOCKED_2026-10-13.md.

Run from the repository root:
    .venv/bin/python research/staleness/verification_2026-10-13/check_claims.py
It checks the saved copies of the ASE artefact (commit 597fd6582b) and the arXiv v1 full text,
exits non-zero if anything fails, and prints PASS/FAIL per item.
"""
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
A = D / "ase_artifact"
PAPER = re.sub(r"\s+", " ", (D / "lit" / "2607.26843.full.txt").read_text())

CODE = [  # (claim, file, line, exact substring expected on that line)
    ("A1 numeric MR: altering -> SATISFIED iff answers differ", "step3_mutation_campaign.py", 747,
     'verdict = "SATISFIED" if not answers_match else "VIOLATED"'),
    ("A1 numeric MR compares with the ORIGINAL answer", "step3_mutation_campaign.py", 719, "orig = result.original_answer"),
    ("A1 GT numeric check uses the dataset ground truth", "step3_mutation_campaign.py", 1064,
     "ground_truth = float(baseline.ground_truth_short)"),
    ("A1 GT LLM judge given the ORIGINAL ground truth", "step3_mutation_campaign.py", 1044,
     "ground_truth=baseline.ground_truth_short"),
    ("A1 meta-eval: system_good inverted for altering", "step3_mutation_campaign.py", 1199,
     'system_good = True if gt_correct == "idk" else (is_preserving == (gt_correct == "correct"))'),
    ("A2 abstention N/A satisfies altering", "step3_mutation_campaign.py", 729, 'verdict = "SATISFIED"'),
    ("A2 step2 keeps baseline only if it matches GT", "step2_baseline_rag_filter.py", 221,
     "is_correct = numeric_answers_match(answer, ground_truth_numeric, NUMERIC_TOLERANCE)"),
    ("A2 default semantic check is the LLM judge", "step3_mutation_campaign.py", 1256,
     'parser.add_argument("--semantic-check", type=str, choices=["manual", "llm"], default="llm")'),
    ("A2 judge told CN 'alters key facts' (pre-chunk)", "step3_mutation_campaign.py", 653,
     "injected a counterfactual passage that alters key facts relevant to the question"),
    ("A3 step4 repair rule", "step4_fault_repair.py", 395, 'fault_repaired = (gt_correct != "correct")'),
    ("A3 supplemental repair rule", "rq3_rerank_select_repair.py", 317, 'repaired = gt_correct != "correct"'),
    ("A5 post-chunk appends the counterfactual", "mutation_operators.py", 651, 'c.content = c.content + " " + contradiction'),
    ("A5 pre-chunk random position", "mutation_operators.py", 602, "pos = self._rng.randint(0, len(sentences))"),
    ("A6 counterfactual written to mislead (1/2)", "mutation_operators.py", 554, "should mislead a reader"),
    ("A6 counterfactual written to mislead (2/2)", "mutation_operators.py", 555, "into giving a wrong answer"),
    ("A8 LLM MR judge shown ORIGINAL context only", "step3_mutation_campaign.py", 814,
     "retrieved_content_original=result.retrieved_content_original"),
    ("A8 document mode uses the original document", "step3_mutation_campaign.py", 858,
     'doc_map = {e["question_id"]: e["document_extracted"] for e in entries}'),
    ("A8 LLM verdict threshold >= 3", "step3_mutation_campaign.py", 621, 'verdict = "SATISFIED" if score >= 3.0 else "VIOLATED"'),
    ("A8 judge: hedging rated 4-5", "pipeline_config.py", 607,
     "If the mutated answer abstains, acknowledges conflicting information, expresses uncertainty, or hedges its claims, rate 4 or 5."),
    ("B1 numeric answer is one JSON field", "pipeline_config.py", 830, 'return str(data.get("computed_formula", "N/A"))'),
    ("Setting: default top-k = 1", "pipeline_config.py", 29, "TOP_K_RETRIEVAL = 1"),
]
QUOTES = [  # (claim, exact text in arXiv v1)
    ("A1 spec: align with injected fact", "specifically where y^{\\prime} aligns with the injected fact"),
    ("A1 spec: GT check reflects counterfactual", "the GT check verifies whether y^{\\prime} reflects the injected counterfactual"),
    ("A4 RQ3 example", "persisted after repair: the stronger model still adopted the contradictory evidence, producing an answer that directly opposed the ground truth"),
    ("A4 RQ1 example", "ignoring the contradictory evidence now present in the retrieved context"),
    ("A4 fault definition", "A robust system must exhibit explicit conflict awareness, signalling or resolving the discrepancy"),
    ("A4 judge description in paper", "or appropriately acknowledges conflicting information, satisfying the"),
    ("A7 their admission", "MR validation via exact match coincides with meta-evaluation with GT"),
    ("Setting: top-k 1", "In the study, the default top- k is set to 1"),
]

fail = 0
for claim, f, line, sub in CODE:
    lines = (A / f).read_text().splitlines()
    ok = line <= len(lines) and sub in lines[line - 1]
    near = "" if ok else f"  (found at line(s) {[i + 1 for i, l in enumerate(lines) if sub in l]})"
    print(f"{'PASS' if ok else 'FAIL'}  {claim}  [{f}:{line}]{near}")
    fail += not ok
for claim, q in QUOTES:
    ok = re.sub(r"\s+", " ", q) in PAPER
    print(f"{'PASS' if ok else 'FAIL'}  {claim}  [quote]")
    fail += not ok
print(f"\n{len(CODE) + len(QUOTES) - fail}/{len(CODE) + len(QUOTES)} checks pass")
sys.exit(1 if fail else 0)
