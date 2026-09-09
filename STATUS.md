# STATUS — what this project is, right now

**Updated 8 September 2026.** Read this first, then `README.md` to run it and
`FINDINGS.md` for the measurements behind every claim.

---

## 1. What the project set out to do

*Drift-Aware RAG: Label-Free Detection of Retrieval Degradation.* A RAG system's
retrieval quality decays as its corpus is revised and its query stream moves. Detect
that decay **without relevance labels**, from the geometry of the embedding space:
measure distributional drift on the query and corpus channels, classify which of four
regimes you are in, and trigger a proportionate response.

## 2. What the code is now

### The pipeline runs end to end

`python main.py run-all` completes on real revision data and writes a validation report.
This was not true before 8 September 2026: the pipeline could not execute at all (its
data source was broken and its dependencies unavailable), and would not have produced a
valid number if it had.

It was rebuilt on the results that survived validation. Each change traces to a
numbered finding in `FINDINGS.md` §3, and each module's docstring names the one it
addresses.

| Was | Now | Finding |
|---|---|---|
| Query stream was one constant string, so query drift was identically zero | Sentence-window queries from live document text, each with a known gold document | F3 |
| Corpus snapshots near-identical across windows | Real revision timestamps drive which revision is live at each t | F4 |
| Fixed threshold τ = 0.15 on a raw statistic | Per-window permutation null; z-scores and p-values | F13, and 12.8 |
| No false-alarm-rate writer existed anywhere | A real null slice, FAR with a Wilson interval and stated granularity | F2, F21 |
| D(t) dominated by a fixed query-vs-corpus genre gap (91.4% measured) | Genre gap reported as a diagnostic, removed from the alarm path | F5 |
| α + β ≡ 1, collapsing the 2×2 regime plane to a line; `STABLE` unreachable | Classification on two absolute z-scores; all four quadrants reachable | F11 |
| Policy engine a provable no-op — selected `a1_no_op` everywhere | Regret in compute-equivalent units; all five actions reachable | F6 |
| `a3_reindex` existed only in an enum | Real budgeted partial refresh with five selectors | F16 |
| MMD allocated 16.4 GB per window and OOMed | Gram identity: 32 MB, agrees to 4e-17 | F7 |
| W₁ CI measured Monte-Carlo error of the slicing | Bootstrap over the data; slicing error reported separately | F12 |
| CUSUM could not fire | Frozen burn-in reference, slack in σ units, ARL₀ calibration | F14 |
| RAGAS faithfulness as endpoint — blind to staleness, called with a 3-version-old schema, exception swallowed | recall@k / MRR against a known gold document; RAGAS optional and corrected | F10, F17, F23 |
| Invalid Lipschitz estimator | Removed, with the reason recorded | F15 |
| Needed torch, chromadb, datasets, ragas | Runs on numpy + scipy + sklearn + pandas; heavy stack optional | — |

**Two defects were found during the rebuild, by the pipeline's own null slice**
(`FINDINGS.md` §13.4): overlapping corpus windows break the permutation null's
exchangeability and pinned the corpus z-score near −5 regardless of what happened; and
drawing null queries per window-group made a "null" that alarmed 5 times out of 5. Both
produce confident nonsense rather than errors.

### The three code layers

**Layer 1 — the pipeline** (tracked). `main.py`, `configs/`,
`data/`, `embeddings/`, `drift/`, `retrieval/`, `policy/`, `evaluation/`, `streaming/`,
`dashboard/`, `tests/`. 126 tests. Runs.

**Layer 2 — `research/`** (tracked, second generation). Cleaner than the original, but
its 24 result files come from `research/dar/simulator.py` — synthetic data. Seven issues
in `FINDINGS.md` §7. `research/experiments/aux.py` was renamed to `auxiliary.py`, because
`AUX` is a reserved DOS device name and the repo would not clone on Windows without it.

**Layer 3 — `research/gates/`** (tracked). The validation scripts, corpora and results
behind `FINDINGS.md` §11–§12, including the original numpy MiniLM implementation that
`embeddings/backends.py` is derived from.

## 3. Where the science stands

Measured on real revisions with real embeddings across three corpora: Python docs 3.8→3.13
(1,751 docs, 81% revised), arXiv v1→latest (7,346, 25%), Wikipedia 2023→2026 (344, 89%).

**Four claims hold**

1. **Drift thresholds do not transfer across traffic volume.** Calibrate at n = 3000,
   deploy at n = 300 → FAR **1.000**; reverse → **0.000**, with every window drawn from
   the same distribution. Predicted score inflation √10 = 3.162, measured 3.072 and
   3.110. Reproduced on the rebuilt pipeline (§13.2); `python main.py transfer`.
2. **Power at realistic effect sizes sits near the false-alarm floor** — ≈ 0.15 for arXiv
   and ≈ 0.20 for Python docs at n = 1500, against a 5% floor (§12.9, corrected in §13.3);
   `python main.py power`.
3. **Displacement is spatially autocorrelated** — Moran's I +0.1184 (docs, z = +15.4),
   +0.0557 (arXiv, z = +15.0), +0.0311 (wiki, z = +1.8).
4. **Position is unpredictable; magnitude is predictable** — interpolating the
   displacement *vector* field is worse than doing nothing, 6 of 6 conditions.

**Two were refuted**

5. **The blind spot.** Corpus-side displacement predicts retrieval damage at AUC
   **0.769 / 0.782**, not blindly. Part 1 does not need it.
6. **probe-kNN as a distinct method.** It beats random, but given the *same* probe it
   ties or loses to cluster-threshold refresh — the closest deployed prior art — at every
   budget (§12.5).

**Therefore: a publishable negative-results paper.** Claims 1 and 2 say a corpus-side
drift detector cannot be pinned to a stable operating point and cannot see the phenomenon
where you would operate it. Claim 6 is why the selective-re-index method is not the
contribution it was taken to be.

## 4. Numbers that were published and are now withdrawn

Do not use these.

| Withdrawn | Replaced by | Why |
|---|---|---|
| "2.9× less compute than a full re-index" | nothing survives | simulator only |
| "probe-kNN beats random 3.00× at 20% budget" | **1.43×** at n = 7,346 | 3-seed artifact; 30 seeds gives 1.09×, p = 0.39 |
| "2–3× more recoverable quality than random" | **1.26–1.58×** | same |
| "33× signal ratio — corpus drift is blind to damage" | **AUC 0.77** | simulator modelled in-place revision far too aggressively |
| Moran's I = +0.0798 (arXiv) | **+0.0557** | small-sample inflation; n = 1,786 → 7,346 |
| η² 0.355 vs 0.066 across corpora | use Moran's I only | artifact of group count (302 vs 14) |
| "nothing reaches 50% power until displacement 0.765, larger than any real corpus" | see §13.3 | mixes per-moved and per-all displacement conventions |

The §13.6 bake-off table is **also not quotable**: its headroom is 8 queries in total, so
its resolution is 0.125 per query and it cannot separate the selectors.

## 5. Where everything lives

| | |
|---|---|
| How to run it | `README.md` |
| Audit of the original pipeline, 25 findings with file:line | `FINDINGS.md` §3 |
| Simulator-era results (largely superseded) | `FINDINGS.md` §4–§6 |
| First real-data results, with a correction notice | `FINDINGS.md` §11 |
| Three-corpus replication — the core results | `FINDINGS.md` §12 |
| **The rebuild, and what it found** | `FINDINGS.md` §13 |
| Validation scripts, corpora, raw results | `research/gates/` |

## 6. Next

1. **Second embedding model.** Everything is MiniLM-L6-v2. `EMBED_BACKEND` makes this a
   one-line change; it closes the most obvious reviewer objection. Highest value.
2. **Write the paper.** 0% done, now the critical path. TMLR, leading with the negative
   result.
3. **A properly powered bake-off** if the selector question matters: many more queries per
   window and 30 seeds (§13.6).
