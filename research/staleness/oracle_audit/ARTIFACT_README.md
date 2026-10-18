# Artefact: "When the Oracle Is the Fault"

This folder, with `../pretests/` and `../verification_2026-10-13/`, reproduces every number in the
manuscript (`paper/main.tex`). Each analysis was pre-registered: the script's SHA-256 was logged
before any data were generated or opened (`*prereg.log`).

## Environment
- **Project environment:** `.venv`, from `pyproject.toml` / `uv.lock`. Runs the experiments and
  tests.
- **Analysis environment:** for the authors' outputs and the figures, a Python ≥ 3.11
  environment with `pypdf` and `matplotlib`. We used `~/SemProject/ase-replication/.venv`.
- **Local LLMs:** Ollama 0.18.3 with `phi3:mini` (Q4_0, id 4f2222927938),
  `qwen2.5:7b-instruct` (Q4_K_M, 845dbda0ea48) and `qwen2.5:14b-instruct` (Q4_K_M,
  7cdf5a0187d5). Temperature 0, at most 24 output tokens. All answers are cached, so re-runs
  reuse them.
- **No paid APIs are used.**

## Inputs not redistributed
- **ASE artefact:** github.com/dbr7/RAG-metamorphic-mutation, commit `597fd6582b`, and the
  authors' Figshare output archive (unzipped to `~/SemProject/ase-replication/output`).
- **Published paper:** DOI 10.1145/3832783.3837543 (CC BY 4.0).
- **How to re-fetch both:** see `../verification_2026-10-13/THIRD_PARTY_NOT_INCLUDED.md`.

## Reproduce, by paper section

| Paper section | Command (from the repository root) | Output |
|---|---|---|
| §4 audit (29 code and quote checks) | `.venv/bin/python research/staleness/verification_2026-10-13/check_claims.py` | 29/29 PASS |
| §4 version of record | see CLAIMS_LOCKED §D3 (15/15 passages identical to arXiv v1) | — |
| §5 impact (M1–M5, pre-registered) | `.venv/bin/python research/staleness/oracle_audit/analyze_ase_outputs.py --root ~/SemProject/ase-replication/output` | `run1_ase_outputs.log` |
| §5 Table 1 reproduction, samples | `…/oracle_audit/posthoc_checks.py` | `posthoc_checks.log` |
| §5 refusals (corrected v2 pattern) | `<analysis-env>/python posthoc_refusal_v2.py` and `posthoc_refusal_v2_rates.py` (in this folder; pattern in `refusal_common.py`, pinned in `tests/test_oracle_audit.py`) | `posthoc_refusal_v2*.log` |
| §6 MR-Provenance-Swap (pre-registered) | `.venv/bin/python research/staleness/pretests/p15_provenance_swap.py` | `pretests/p15_final_report.log` |
| §6 clustered CIs | `…/oracle_audit/stats_p15_clustered.py` | `stats_p15_clustered.log` |
| §6 RQ5 strategies (pre-registered) | `…/pretests/p16_strategies.py --pilot` / `--main` | `pretests/p16_*.log` |
| Figures | `<analysis-env>/python research/staleness/oracle_audit/paper/make_figures.py` | `paper/fig_*.pdf` |
| Human validation | `research/staleness/human_validation/build_package.py`; after labelling, `score_agreement.py annotator_1.csv annotator_2.csv` | κ report |
| Tests | `DRIFT_MINILM_DIR=~/.cache/minilm-l6-v2 .venv/bin/python -m pytest` | all pass |

## Claim register
`../CLAIMS_LOCKED_2026-10-13.md` lists every claim with its evidence, its status (verified,
withdrawn, exploratory) and a change log. That includes the corrections we made, e.g. the
refusal count, corrected from 12.5% to 26.0%.
