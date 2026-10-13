# Pre-tests (exploratory — not results)

Cheap checks run on 2026-10-09 to decide whether journal-extension ideas are worth building.
Numbers from these scripts are recorded in `RESEARCH_HANDOFF.md` §4 and are **not** paper
results until re-implemented as a tested module. Run from the repo root with the project venv:

    .venv/bin/python research/staleness/pretests/p1_ideas.py      # margin, keep-vectors, mixed index, token sensitivity
    .venv/bin/python research/staleness/pretests/p2_geometry.py   # truncation, Δ decomposition, random-projection model
    .venv/bin/python research/staleness/pretests/p3_certified.py  # certified top-k invariance for numeric edits
    .venv/bin/python research/staleness/pretests/p4_dilution.py   # signed cos(q, Δ) on retained / new-text queries
    .venv/bin/python research/staleness/pretests/p5_chunked.py stats|run [chunk]  # which edits need re-embedding (chunked)
    .venv/bin/python research/staleness/pretests/e0_fact_rates.py fetch|analyse  # E0: fact- vs page-change rates on Wikipedia (network)

p1–p3 print the inside-window population separately (`data.visible`); edits past the
encoder window cannot move the vector and are not evidence of insensitivity.

Known caveats: bge is run at a 256-token window (its native window is 512); everything is
per document, no chunking; p1's token-perturbation sample is 150 docs per corpus.

## Later pre-tests (2026-10-10 .. 13): real-revision metamorphic testing direction

Each script's docstring holds its pre-registration; outputs are in the `*.log` file of the
same name. Results are summarised in `RESEARCH_HANDOFF.md` §0 and recorded in §7 (0b–0h).

| script | question | status |
|---|---|---|
| `p5_chunked.py` | does keeping old vectors for edited chunks save cost? | no-go |
| `p6_metamorphic_k1.py` | with both versions indexed, k=1, how often does only the old value reach the context? | pass |
| `p7_generator_conflict.py` | does the generator state the old value when both values are in context? | pass (17%) |
| `p8_version_labels.py` | do version labels fix it? | partial |
| `p9_wiki_infobox.py` | scale-up on Wikipedia infobox updates (fetch + build) | VOID ×2; superseded by p11 |
| `p10_freshqa_feasibility.py` | real human questions (FreshQA) with real revisions | 23 usable |
| `p11_field_passages.py` | **main result**: per-field passages, 605 cases (fixed sample), plus FreshQA | phi3 0.31, Qwen-7B 0.19 replicate; Qwen-14B 0.03 does not (lists both values 84%, post hoc) |
| `p12_counterfactual.py` | RQ2: ASE counterfactual operator vs real updates | phi3 +0.15 (CN overstates); Qwen-7B +0.04 (fair proxy) |
| `p13_monitors.py` | RQ3: which monitors see the fault | only the judge told the change sees it (agrees 0.989 with the diff oracle) |
| `e0_fact_rates.py` | harm-aware refresh scheduling (Wikipedia weekly snapshots) | no-go |

Scoring: use `p9_wiki_infobox.commit` (digit-boundary matching; new / old / both / neither).
`facts.NUM` misreads comma-joined answers. Fixes made on 2026-10-13: `RESEARCH_HANDOFF.md` §0.4.

Generators are local Ollama models (no paid APIs). Answers are cached in
`embeddings/cache/staleness/p9_generation/`, so rerunning a script resumes it.
