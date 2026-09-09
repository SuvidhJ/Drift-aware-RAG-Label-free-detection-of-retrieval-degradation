# FINDINGS — independent teardown and executed gates

**Subject:** the pipeline at commit `8a85622`, before the rebuild recorded in §13
**Reference:** *Drift-Aware RAG: Label-Free Detection of Retrieval Degradation*, working paper v0.3, Jul 2026
**Date:** 5 September 2026, with later sections dated inline

This is the measurement record. It keeps superseded results in place with correction
notices rather than editing them, so a claim and its later refutation can both be read.
Numbers withdrawn during validation are listed in [`STATUS.md`](STATUS.md) §4 and must not
be quoted.

**Orientation:** [`STATUS.md`](STATUS.md) says what the project is and where everything lives.

---

## Contents

1. [Scope and method](#1-scope-and-method)
2. [Executive summary](#2-executive-summary)
3. [Audit findings — the shipped pipeline](#3-audit-findings--the-shipped-pipeline)
4. [Gate 1 — threshold transfer under volume shift](#4-gate-1--threshold-transfer-under-volume-shift)
5. [Gate 2 — selective re-index](#5-gate-2--selective-re-index)
6. [The blind-spot result](#6-the-blind-spot-result)
7. [Findings on `research/` itself](#7-findings-on-research-itself)
8. [Provenance of the audited code](#8-provenance-of-the-audited-code)
9. [Reproduction](#9-reproduction)
10. [Limitations and open questions](#10-limitations-and-open-questions)

---

## 1. Scope and method

Full read of all 81 tracked files, then **independent numerical re-derivation** of every claim
rather than acceptance of the existing audit. Verification ran in a clean virtual environment with
no repository dependencies: Python 3.12, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, POT
0.9.7.post1.

Two conventions used throughout:

| Marker | Meaning |
|---|---|
| **[M]** | Measured — I ran code and report the number it produced |
| **[R]** | Read — established by reading source, with file:line given |
| **[D]** | Derived — follows analytically from [M] or [R] |

Result files produced by this work are in [`research/gates/results/`](research/gates/results/);
the scripts that produced them are in [`research/gates/`](research/gates/).

---

## 2. Executive summary

**On the shipped pipeline.** It runs to completion and cannot produce a valid number. Ten blocking
defects, eight severe, seven moderate. The four headline metrics on page 8 of the working paper are
*targets*, and two of them are structurally uncomputable by the code as written. The paper's own
README labels the same four quantities "Validation Targets — report actual".

**On the science.** Two decisive experiments were designed and executed. Both pass, and a third
result emerged that reframes the project's central thesis.

| | Result | Status |
|---|---|---|
| Gate 1 | Raw drift thresholds do not transfer across query volume: FAR **5% → 100%**. Permutation calibration holds at 8.6%. | Passes |
| Gate 2 | Probe-based selective re-index: **2.9× less embedding compute** for a 3.9-point recall@10 shortfall. | Passes |
| Unplanned | The commonest form of index staleness costs **30 points of recall@10** at a corpus-side W₁ of **0.00097** — the null floor. | Reframes the thesis |

**Consequence for the plan.** D(t) is defensible as a *query-side* instrument and is not a
corpus-staleness detector. Corpus staleness needs a per-document instrument, which §5 supplies.
The θ*-based selector proposed earlier in this project's planning is **wrong** — it barely beats
random — and is superseded by the probe estimator in §5.2.

---

## 3. Audit findings — the shipped pipeline

### 3.1 Blocking

**F1 — Injected drift never reaches the detector. [R]**
`data/inject_drift.py:47` writes the labelled slice to `snapshots/{category}_labeled/`.
`main.py:135-136` and `main.py:226-227` read `snapshots/{category}/`. `main.py:295-299` then
computes AUROC by pairing **labels from the drifted slice against a signal computed on the
undrifted one**.

**F2 — The false-alarm rate is structurally uncomputable. [M]**
`main.py:301` guards on `results/null_drift_series.parquet`. `grep -rn "null_drift_series"
--include=*.py .` returns **one hit**, that line — no writer exists anywhere.
`build_null_corpus_slice` (`data/inject_drift.py:93-113`) returns a directory path and computes
nothing. The paper reports FAR = 3.1%.

**F3 — The query stream is a single constant string. [M]**
`data/download.py:57-69` requests config `qas_<category>` from `McAuley-Lab/Amazon-Reviews-2023`.
That config does not exist — the dataset ships only `raw_review_*` and `raw_meta_*`, and its card
states there is no question-answering split. The request raises; the bare `except` at line 67 falls
through to `_synthesise_qa_from_reviews`, which at line 88 emits the literal
`"What do users say about this product?"` for every row. **Every query embedding in every window is
bit-identical**, so W₁(Q_t, Q_{t−1}) ≡ 0 and the query-drift axis of the taxonomy has no data.

**F4 — Corpus snapshots are near-identical across windows. [R]**
`data/build_snapshots.py:127-130` treats a product as alive when its liveness bounds are `NaN`.
Liveness is reconstructed by joining `.head(20 000)` of 1.3 M meta rows against `.head(100 000)` of
20.8 M review rows — two independently ordered dumps whose ASIN overlap is small — so most products
get `NaN` and are alive in every window.

**F5 — D(t) is dominated by a constant level term. [M]**
`drift/statistic.py:93` uses `_centroid_dist(Q_t, C_t)`: the gap between two *text genres*, fixed at
t=0, not a change. Measured centroid share of D(t): **92.5%** with no genre gap, **99.7%** at a
realistic one. The repo's own run records `cgap_share_of_d_quiet = 0.8718`. **[D]** The alarm rule
reduces to ‖μ_Q − μ_C‖ > 0.30 — decided before any drift occurs.

**F6 — The cost-aware policy is a constant no-op. [M]**
`policy/engine.py:30` sets no-op regret to `d * (alpha + beta)`, and α̂ + β̂ ≡ 1 by construction, so
it is just `d`. Costs are compute multiples (1…110); regret is a faithfulness fraction (0…1); λ=1
mixes them uncalibrated.

```
sweep D ∈ [0, 0.14], α ∈ [0, 1], λ = 1  →  actions selected: {a1_no_op}
to ever select a2_rerank:  λ > 54 at D = 0.14,  or  D > 7.5 at λ = 1
```

The repo's own `research/results/policy.csv` shows `repo_regime` bit-identical to `always_no_op` at
every λ.

**F7 — MMD allocates an (n, m, d) tensor. [M]**
`drift/mmd.py:34`. At the shipped default `subsample=2000`, d=1024: **16.38 GB** per kernel matrix,
three of them. The Gram identity gives an identical answer (max abs difference **4.27e−15**) in
**32 MB**, **84× faster**. Already fixed correctly in `research/dar/stats.py:99`.

**F8 — The pipeline cannot complete at its own defaults. [M]**
Amazon Reviews 2023 spans May 1996 – Sep 2023 = **329 monthly windows**.
- 329 × 20 000 = **6 580 000** BGE-M3 forward passes on `device="cpu"` → **2.5–7.6 days**
- `w1_bootstrap_iters=1000` → 329 × 2 × 1001 = **658 658** sliced-W₁ calls → **3.8–15.2 days**
- MMD OOMs on window 1

**F9 — Download is broken against the lockfile. [R]**
`uv.lock` pins `datasets 5.0.1`. Loading scripts and `trust_remote_code` were removed in datasets
4.0. `data/download.py:25, 40, 61` all pass it. Phase 0 raises on the first call.

**F10 — RAGAS called with a schema three versions old, failure swallowed. [R]**
`evaluation/ragas_harness.py:24-36` builds `question`/`answer`/`contexts`; RAGAS renamed these to
`user_input`/`response`/`retrieved_contexts` in 0.2 and the lockfile pins 0.3.1. Line 85
(`float(result["faithfulness"])`) also assumes a scalar where ≥0.2 returns a list. The exception is
caught at line 89 and converted to `None`, so faithfulness is empty for every window and both ρ and
lead-time are skipped without the run failing.

### 3.2 Severe

**F11 — The "2×2 plane" is a one-dimensional partition. [M]**
`drift/decomposition.py:74-76` — α̂ and β̂ are complementary fractions.

```
α̂ + β̂ ≡ 1  →  corpus_drift α∈[0, 0.4]   joint α∈[0.4, 0.6]   query α∈[0.6, 1]
(W₁ᵠ, W₁ᶜ) = (0.9, 0.9)  and  (0.009, 0.009)  →  identical regime
decomposition.py:48  `return Regime.STABLE`  →  unreachable
```

"Joint drift" therefore means *the two channels are similar in magnitude*, not *both are large* —
the least informative case triggering the ×110 action. The classifier is scale-free and cannot
distinguish a large drift from a tiny one.

**F12 — The W₁ confidence interval measures Monte-Carlo error of the slicing. [M]**
`drift/wasserstein.py:60-65` re-runs the estimator with new projection seeds and never resamples X
or Y.

| n_projections | 100 | 500 (shipped) | 2000 | 8000 |
|---|---|---|---|---|
| Shipped CI width | 1.196e−4 | 4.739e−5 | 2.218e−5 | 1.080e−5 |
| True sampling width | 1.036e−4 | 5.083e−5 | 3.104e−5 | 2.941e−5 |

Width ∝ P^(−1/2) → 0, while the estimand's uncertainty plateaus. Coverage of the true value when
X, Y are drawn from the same law (so true W₁ = 0): **0 of 15 trials**, at a nominal 95%.

**F13 — No null baseline; the raw statistic partly measures traffic volume. [M]**

| n | 300 | 600 | 1200 | 2400 | 4800 |
|---|---|---|---|---|---|
| null sliced-W₁ | 0.001975 | 0.001416 | 0.001020 | 0.000726 | 0.000523 |

log–log slope **−0.47** against a theoretical −0.5. τ = 0.15 is **75×** the largest null value
observed.

**F14 — The streaming CUSUM can never fire. [R]**
`streaming/online_detector.py:54-61`. Three compounding faults: the reference mean is computed
*after* appending the current observation, so it chases the signal; slack `k = 0.5` is absolute,
not in σ units, against a statistic on the 0.1 scale, so `g_pos = max(0, g_pos − ~0.5)` is pinned at
zero and the alarm degenerates to `d > τ`; and the 10-window rolling reference absorbs gradual drift
by construction. `g_neg` is computed and never read. No ARL₀ calibration.

**F15 — The Lipschitz constant is the wrong object, estimated the wrong way. [M]**
Kantorovich–Rubinstein needs the Lipschitz constant of the *retrieval functional* in embedding
space; W₁ in D(t) is already measured there, so the encoder's constant is not the required quantity.
The estimator is separately invalid — the denominator is a normalised character edit distance,
which scales as 1/document-length:

| document | 5 words | 20 words | 80 words | 300 words |
|---|---|---|---|---|
| normalised edit distance | 0.17647 | 0.04317 | 0.01073 | 0.00286 |
| ratio multiplier | 5.7× | 23.2× | 93.2× | **349.8×** |

`L̂ = max` over 500 samples is won by the longest document. Corpus text is truncated at 2048 chars
and questions at 512, a built-in 4× systematic gap. The bootstrap CI on a maximum is inconsistent
for extreme order statistics. The paper's "L ≈ 1.2–2.1 for bge-m3" cannot come from this estimator
except by coincidence.

**F16 — Three of five remediation actions do not exist. [R]**
`a3_reindex`, `a4_finetune`, `a5_escalate` appear only in the enum, the cost table and
`dashboard/app.py`'s emoji map. `retrieval/rag_pipeline.py:53, 56` branches on `a2_rerank` only.
`main.py:242` calls `build_index(..., force=True)` per window — there is no incremental re-index,
only a full rebuild.

**F17 — The endpoint cannot see the target failure. [D]**
RAGAS faithfulness scores answer-to-context consistency. Stale-but-plausible context scores *well*
while being wrong. The oracle is blind to the failure mode the paper is about.

**F18 — The tests validate an unreachable regime. [R]**
`tests/test_decomposition.py:34-52` feeds D ∈ {0.95, 1.80} against an achievable maximum of 0.139.
`tests/test_policy.py:63-74` needs λ=1000, and its own comment works out the arithmetic proving the
engine is inert at the default λ=1. `test_wasserstein.py:16` and `test_mmd.py:15` compare `X` to
`X.copy()` — the degenerate null. **The test that would have caught F13 is the one not written.**
Zero tests on `statistic.py`, `evaluation/*`, `streaming/*`, `data/*`, `retrieval/*`.

### 3.3 Moderate

| # | Finding | Location |
|---|---|---|
| F19 | AUROC has exactly one positive window; code implements "delisted *at* t" while the docstring says "at or before t" | `evaluation/correlate.py:92-100` vs `:90` |
| F20 | Spearman ρ over ~12 autocorrelated windows of 10 LLM-judge samples each; no detrending, no block bootstrap, no CI | `main.py:245`, `evaluation/correlate.py:14-36` |
| F21 | FAR granularity is 1/T; at T≈12 the achievable values near target are 0% and 8.3%. The paper's 3.1% ≈ 1/32 requires ≥32 null windows to be expressible **[M]** | design |
| F22 | `min_window_samples` — declared as threat mitigation T4, read by nothing (`grep`: 0 hits outside settings) **[M]** | `configs/settings.py:22` |
| F23 | Three bare `except Exception` on the critical path convert hard failures into empty results; `run-all` prints success regardless | `download.py:67`, `rag_pipeline.py:81`, `ragas_harness.py:89` |
| F24 | `ds.to_pandas().head(n)` materialises a 20.8 M-row split before truncating; `head()` is not a sample | `data/download.py:27, 42, 63` |
| F25 | `_load_embeddings` is dead code with incorrect prefix mangling (`"queries"` → `"quers"`); `save_validation_report` formats `'N/A'` with `:.4f` → `TypeError` | `main.py:47-60`, `correlate.py:138-141` |

---

## 4. Gate 1 — threshold transfer under volume shift

**Question.** The repo's own bake-off shows a plain sliced-W₁ beating the calibrated detector on
AUROC (0.9995 vs 0.9668). The defence is that raw distances are volume-confounded, so their
*threshold* does not transfer, and AUROC — being threshold-free — cannot express that. Untested
until now.

**Protocol.** Calibrate each detector to a 5% false-alarm rate on stable windows at one query
volume; freeze the threshold; deploy on held-out stable windows at the same volume, then at a
different volume. d=256 unit-sphere samples, reference window fixed at n=2000, 70 windows per cell,
40 permutations for the calibrated statistic, injected drift 0.11 relative units.
Script: [`research/gates/e2_threshold_transfer.py`](research/gates/e2_threshold_transfer.py).

### 4.1 Mechanism — the score itself depends on sample size **[M]**

Stable windows only; **no drift exists**.

| Detector | mean score, n=400 | mean score, n=4000 | ratio |
|---|---|---|---|
| sliced-W₁ (mean) | 0.00209 | 0.00106 | 0.51 |
| sliced-W₁ (max) | 0.00448 | 0.00227 | 0.51 |
| centroid shift | 0.02630 | 0.01316 | 0.50 |
| MMD² | ~0 | ~0 | 0.03 |
| **permutation z** | −0.0687 | +0.0820 | no systematic scaling |

**[D]** With the reference fixed at n_ref=2000, two-sample error scales as
(n_cur^−½ + n_ref^−½). Going n_cur 400→4000 predicts a ratio of
(0.0500+0.0224)/(0.0158+0.0224) = **0.53**. Measured 0.50–0.51.

### 4.2 Calibrate at n=4000, deploy at n=400 **[M]**

| Detector | FAR in-domain (held out) | FAR after transfer | Power after transfer |
|---|---|---|---|
| sliced-W₁ (mean) | 0.057 | **1.000** | 1.000 |
| sliced-W₁ (max) | 0.057 | **1.000** | 1.000 |
| centroid shift | 0.057 | **1.000** | 1.000 |
| MMD² | 0.057 | 0.329 | 1.000 |
| **permutation z** | 0.057 | **0.086** | 1.000 |

Every raw distance detector fires on **100% of stable windows**. In deployment that is not a
degraded monitor; it is a monitor that must be switched off.

### 4.3 Calibrate at n=400, deploy at n=4000 **[M]**

| Detector | FAR in-domain (held out) | FAR after transfer | Power after transfer |
|---|---|---|---|
| sliced-W₁ (mean) | 0.014 | **0.000** | 1.000 |
| sliced-W₁ (max) | 0.086 | **0.000** | 1.000 |
| centroid shift | 0.029 | **0.000** | 1.000 |
| MMD² | 0.029 | **0.000** | 1.000 |
| **permutation z** | 0.014 | **0.043** | 1.000 |

The failure is asymmetric. Volume *increase* silences the detector — an alarm budget requested and
never delivered. Volume *decrease* is the operational catastrophe.

### 4.4 What this does not show

**Detection power was 1.000 in every cell, including after transfer.** The injected drift (0.11
relative units against a null centroid gap of 0.013) sits far above the detection floor, so this
design cannot demonstrate power collapse. **Do not claim power collapse** without a follow-up sweep
at effect sizes near the floor.

An earlier run at NW=40 gave a calibrated in-domain held-out FAR of 0.200; at NW=70 this resolves to
0.014–0.057. The earlier figure was small-sample noise, not a defect in the statistic.

---

## 5. Gate 2 — selective re-index

**Question.** Can the compute cost of re-embedding a stale vector index be reduced without
proportionate loss of retrieval quality? This is the technical-effect evidence required for both
*Alice* Example 47 (claim 3 shape) and Indian §3(k) under *AB Initio*.

### 5.1 Two modelling failures, both informative **[M]**

- **A rigid shift of a whole product line is invisible to cosine retrieval.** Relative geometry is
  preserved, so recall@10 measured **1.000 with a fully stale index**. This plausibly explains why
  corpus drift is the hard cell in the repo's confusion matrix (23/60) and why top-1 cosine scores
  below chance there.
- **Small per-document perturbation is also near-invisible.** With a move of 1.15 relative units the
  stale-index recall@10 was still 0.9986.

Staleness bites only when documents are **rewritten individually**: the page changes, its embedding
moves in its own direction, and a query about the new content no longer matches the old vector.
All results below use that model.

### 5.2 The proposed algorithm

```
PROBE-KNN SELECTIVE RE-INDEX

 1. re-embed a random probe S_p, |S_p| = 2% of N
      -> for j in S_p we learn  m_j = || c_new[j] - c_index[j] ||
 2. for each un-probed document i, predict movement FROM ITS STALE POSITION ALONE:
      m̂_i = Σ_j w_ij · m_j   over the 15 nearest probe docs,
              w_ij ∝ max(⟨c_index[i], c_index[j]⟩, 0)^8   (normalised)
 3. re-embed the top-k by m̂
 4. total spend = (2% + k/N) of a full re-index
 5. (optional) repeat 2–3 in batches, treating each newly re-embedded document
    as an additional observed displacement — an active-learning loop whose labels
    are a by-product of work already paid for
```

Rationale: document revision is regionally concentrated. A section under active revision produces
many large movements in one neighbourhood; a static section produces none. The probe samples that
field and kNN interpolates it. **No distributional shift is required** — which is why it survives the
regime in §6 that blinds every distributional detector.

### 5.3 Selector bake-off **[M]**

Share of the full-reindex gain recovered at a 20% budget (probe included), 3 seeds, N=6000
documents, 20 Zipf clusters, 2000 queries, recall@10.
Script: [`research/gates/e5b_selector_bakeoff.py`](research/gates/e5b_selector_bakeoff.py).

| Selector | in-place | mixed | migrating |
|---|---|---|---|
| **probe-kNN (proposed)** | **0.574** | **0.574** | **0.548** |
| low-confidence proximity | 0.347 | 0.316 | 0.104 |
| θ* drift direction | 0.173 | 0.232 | 0.257 |
| random | 0.211 | 0.211 | 0.174 |
| retrieval frequency | 0.206 | 0.123 | 0.057 |
| *oracle (infeasible ceiling)* | *0.760* | *0.799* | *0.777* |

- **θ\* barely beats random and loses to it in the in-place regime.** The earlier proposal to draw
  the claim around the max-slice argmax direction is **superseded**. θ* improves as documents
  migrate, exactly as its mechanism predicts, and is weakest in the common case.
- **"Refresh what gets retrieved most" is worse than random** in two of three regimes: the documents
  being served most are the healthy ones.
- probe-kNN is the only selector stable across all three regimes, at **70–74% of oracle**.

### 5.4 Headline figures **[M] / [D]**

In-place regime, stale recall@10 = 0.8342, full re-index = 1.0000.
Script: [`research/gates/e5c_iterative.py`](research/gates/e5c_iterative.py).

| Budget | probe-kNN recall | share of gain | random recall | share of gain | advantage |
|---|---|---|---|---|---|
| 10% | 0.8802 | 27.7% | 0.8520 | 10.8% | 2.6× |
| 20% | 0.9330 | 59.6% | 0.8667 | 19.6% | **3.0×** |
| 35% | 0.9607 | 76.3% | 0.8895 | 33.4% | 2.3× |

**Derived technical effect:**

- **2.9× less embedding compute than a full re-index** (35% vs 100%) for a **3.9-point recall@10
  shortfall** (0.9607 vs 1.0000)
- **2.2× less compute than random selection at matched quality** — random needs ~76% budget to reach
  0.9607
- Reproduces in the migrating regime: 0.9620 at 35%, 3.8-point shortfall

**Iterative variant:** helps where budget is tightest (27.7% → 32.0% of the gain at a 10% budget),
neutral above 20%. A 5% probe was **worse** than 2% — probe spend is wasted on random documents.

---

## 6. The blind-spot result

Not planned. Emerged from calibrating Gate 2, and it is the most consequential finding here.
Script: [`research/gates/e5a_staleness_calibration.py`](research/gates/e5a_staleness_calibration.py).

Same corpus, same query stream, same ~30% of documents rewritten. Only the *kind* of rewrite
differs. **[M]**

| Staleness regime | recall@10 stale | recall@10 fresh | damage | max-slice W₁(C₁,C₀) |
|---|---|---|---|---|
| **In-place** — document revised, stays in its semantic region | 0.6985 | 1.0000 | **0.3015** | **0.00097** |
| Mixed — half migrate | 0.7035 | 1.0000 | 0.2965 | 0.01279 |
| Migrating — document moves to a new region | 0.6865 | 1.0000 | 0.3135 | 0.03181 |

**The damage is identical to within noise (1.04× spread). The signal varies by 33×.**

In the in-place regime the corpus-side W₁ is **0.00097** — indistinguishable from the null level of
two i.i.d. samples of comparable size measured in F13 (0.00102 at n=1200). **A distributional drift
detector cannot see it, and it costs 30 points of recall@10.**

### Consequences

1. **Corpus-side distributional drift is neither necessary nor sufficient for retrieval
   degradation.** This is a direct empirical counterpart to the theoretical result that the
   corpus-side Kantorovich–Rubinstein bound cannot exist: retrieval is a *max* over documents, and a
   max is Θ(n·W₁)-sensitive to a single document. Moving one of n corpus points by ε changes W₁ by
   ε/n and changes the max by ε.
2. **D(t) should be scoped to the query side.** Query drift is genuinely distributional — users do
   start asking about new things, and that moves the distribution. Corpus staleness is per-document
   and needs a per-document instrument (§5.2).
3. **This is a stronger paper than the one on the poster.** "We set out to detect retrieval
   degradation from distributional drift and found that the commonest form of degradation is
   invisible to it — here is the theory that says why, and here is what to do instead."

---

## 7. Findings on `research/` itself

`research/` is materially better work than the pipeline: the permutation calibration, the
Gram-identity MMD, the max-slice statistic with its Johnson–Lindenstrauss justification, and the
frozen CUSUM with ARL₀ calibration are all correct and well-motivated. Seven issues remain.

| # | Issue | Location | Why it matters |
|---|---|---|---|
| R1 | **Thresholds selected in-sample.** The FAR-matched threshold is the 95th percentile of the same stable rows it is then evaluated on, so `FAR_empirical ≈ 0.05` by construction, not by estimation | `harness.py:165-178`, `gradual.py:57-58` | Would sink a submission on its own. Needs a held-out split |
| R2 | Fisher combination assumes independent p-values; query and corpus samples are coupled (queries are generated *from* the corpus), so χ² with df=4 is anticonservative | `detect.py:153-154` | Permute the combined statistic instead |
| R3 | Exchangeability null on overlapping samples: adjacent corpus snapshots share most of their points, so pooling and re-splitting destroys that dependence and makes the null too tight | `stats.py:169` | Inflated z. Condition on the persistent set |
| R4 | The factorial is saturated — AUROC ≈ 1, power = 1, median delay = 0 for most detectors | `harness.py:26-34` | No power to separate methods |
| R5 | Only the query half of Proposition 1 is checked; the corpus is held fixed | `theory.py:72` | The corpus half is the hard one, and per §6 it is false as stated |
| R6 | CUSUM delays compared at unmatched ARL₀ (frozen 149.6 vs rolling 49.0) | `results/cusum.json` | A detector alarming 3× more often will look faster |
| R7 | One simulator, one parameterisation | `simulator.py` | Honestly flagged in its README, but it bounds every claim |

**Note on scope:** the same limitation applies to the work in §4–§6 of this document. See §10.

---

## 8. Provenance of the audited code

The subject of this audit is the pipeline as it stood at commit `8a85622`, before the
rebuild recorded in §13. Section 3 cites that revision by file and line, so every finding
can be checked against it. The base pipeline predates this analysis and was written by
others; §13 records what was changed in response to the findings and why.

Results in §11-§13 were produced on real corpora collected by the scripts in
`research/gates/`, not on the simulator used in §4-§6. Where a later section overturns an
earlier one, the earlier section is kept with a correction notice rather than edited, so
the record of what was believed when stays legible.

---

## 9. Reproduction

```bash
# clean environment, no repository dependencies
uv venv --python 3.12 .venv-gates
uv pip install --python .venv-gates/Scripts/python.exe numpy scipy pandas scikit-learn POT

# audit verification (§3)
python research/gates/verify_policy_degeneracy.py    # F6, F11
python research/gates/verify_estimators.py           # F7, F12, F13
python research/gates/verify_dt_budget.py            # F5
python research/gates/verify_ci_coverage.py          # F12 coverage, F15

# gates (§4–§6)
python research/gates/e2_threshold_transfer.py       # ~5 min   -> e2_transfer.csv
python research/gates/e5a_staleness_calibration.py   # ~1 min
python research/gates/e5b_selector_bakeoff.py        # ~1 min   -> e5_final.csv
python research/gates/e5c_iterative.py               # ~1 min   -> e5_iter.csv
```

Raw outputs from the runs reported here are committed in
[`research/gates/results/`](research/gates/results/).

**Status of this code:** prototypes. Hard-coded constants, no CLI, no tests, seeds fixed inline.
They are sufficient to reproduce every number in §4–§6 and are not production code. §5.2 and §4 are
specified for reimplementation in `drift/calibrate.py` and `retrieval/selective_reindex.py`.

---

## 10. Limitations and open questions

**Everything in §4–§6 ran on a simulator written for this purpose**, with the same limitation as
`research/dar/simulator.py`. Parameters were chosen to make the phenomena visible; a different
parameterisation would move every number. What should transfer is the **ordering** of methods, the
**mechanisms**, and the **direction** of each failure — all of which follow from high-dimensional
geometry rather than from the chosen constants. Replicate before building on it.

Specific open questions, in priority order:

1. **Is document revision regionally concentrated on real corpora?** This is the load-bearing
   external-validity assumption behind probe-kNN (§5.2). If revision is spatially random, probe-kNN
   degrades to random selection and Gate 2 fails on real data. Directly testable on any corpus with
   revision timestamps. **Run this first** — it is cheap and it can invalidate the central claim.
2. **Does the volume-transfer failure reproduce on real BGE-M3 embeddings?** The n^(−½) mechanism is
   distribution-free, so it should, but it must be shown.
3. **Power near the detection floor.** §4.4 — the current design cannot show power collapse.
4. **Does the blind spot survive on real text?** In-place rewrite is modelled as a fresh draw within
   the same cluster. Real revisions vary in how far they move a document; the distribution of that
   movement on a real corpus is unknown and is what determines the size of the blind spot.
5. **The applicability boundary for §5.2.** Where a reliable per-document change log exists,
   selective re-indexing is trivial and uninteresting. The method's value rests on corpora that do
   not expose one — mirrored or scraped sources, and semantic staleness where the text is unchanged
   but its validity is not. State this condition rather than implying universality.
6. **Related work for §5.2 has not been systematically surveyed.** Adjacent methods to check:
   sampling-based estimation of embedding staleness; partial or incremental re-embedding of a vector
   index; selective recomputation driven by a sampled probe; and cache invalidation by sampled
   staleness estimation, which is where the closest method most plausibly sits.

---

## 11. E6 — open question #1 answered on real data **[M]**

> ### CORRECTION, 6 September 2026 02:45 — read before using anything in §11.4 or §11.5
>
> **The retrieval figures in §11.4 and §11.5 were computed with 3 seeds and do not survive
> replication. They are withdrawn.** Re-run with 30 seeds and a two-sample t-test on the same
> corpus, same queries, same code:
>
> | Budget | probe-kNN | random | ratio | 95% CI | p |
> |---|---|---|---|---|---|
> | 10% | 0.9190 ± 0.0035 | 0.9189 ± 0.0033 | 1.02× | [0.64, 1.63] | 0.94 |
> | 20% | 0.9246 ± 0.0035 | 0.9238 ± 0.0037 | **1.09×** | [0.89, 1.35] | 0.39 |
> | 35% | 0.9327 ± 0.0049 | 0.9306 ± 0.0052 | 1.14× | [0.98, 1.33] | 0.11 |
>
> The effect decays monotonically with seed count — **3 seeds: 3.00×, 12 seeds: 1.92× (p=0.055),
> 30 seeds: 1.09× (p=0.39)** — which is the signature of a low-seed artifact, not a real effect.
> Every confidence interval includes 1.0.
>
> On the Python-docs corpus (§12) it is worse: 1.08×, p = 0.74, and the cluster-threshold prior-art
> baseline ties probe-kNN at every budget.
>
> **What still stands** (all independently established, none affected by this): displacement is
> spatially autocorrelated on real corpora (Moran z = +10.9 arXiv, +15.4 docs); probe-kNN predicts
> displacement magnitude better than chance (Spearman +0.12…+0.23 against a null of ≈0); and
> position is unpredictable while magnitude is predictable.
>
> **What does not stand**: that this prediction produces a measurable retrieval improvement over
> random selection. That is the claim the method's practical value depends on, and at the scale
> tested it is **not demonstrated**. Do not put "2–3× better than random" in a paper, an abstract,
> or a slide.
>
> **RESOLVED — see §12.** The better-powered test ran on 7,346 arXiv documents at 30 seeds. The
> effect against random **is real** (1.26–1.58×, all p < 0.001), just far smaller than 3.00×. But
> §12.5 shows probe-kNN **ties or loses to the closest prior art** at every budget, which is the
> comparison that decides whether the method is a genuine advance. Read §12 before §11.

Added 6 September 2026. `FINDINGS.md` §10 listed this as the first thing to run because it
could invalidate the central claim. It has now been run.

**Verdict: the assumption survives, and is much weaker than the simulator implied.**

### 11.1 Setup

| | |
|---|---|
| Corpus | 1,786 arXiv papers, 14 categories, submitted 2023 H1 |
| Stale index | the **v1** abstract |
| Fresh index | the **latest** abstract as of Sep 2026 |
| Revised | 501 papers (28.1%) have a changed abstract; the rest are genuinely unrevised (m = 0) |
| Embedder | `all-MiniLM-L6-v2`, re-implemented in numpy from the checkpoint (no torch available) |

Scripts: [`e6_fetch_arxiv.py`](research/gates/e6_fetch_arxiv.py),
[`e6_embed.py`](research/gates/e6_embed.py),
[`e6_run_analysis.py`](research/gates/e6_run_analysis.py),
[`e6_run_retrieval.py`](research/gates/e6_run_retrieval.py).
Corpus preserved at [`results/ax_corpus.json`](research/gates/results/ax_corpus.json).

Two things were validated **before** the analysis touched real data, and both mattered:

- **The embedder** — vocabulary ids identical to the checkpoint's own `tokenizer.json` across all
  30,522 entries, deterministic, padding-invariant to 9.3e−9, batch-equals-solo to 1.5e−8, and 10/10
  on a semantic-ordering battery (mean cosine 0.685 related vs 0.008 unrelated). No bit-exact torch
  reference existed here, so this is validated **by properties, not by equality**.
- **The statistics** — on synthetic fields with known structure. Moran's I returned **+0.424,
  z = +43.4** on a positive control and **−0.012, z = −1.1, p = 0.86** on a shuffled null.

### 11.2 Displacement is spatially autocorrelated — decisively, but weakly

| Statistic | Value |
|---|---|
| Moran's I of displacement over kNN(stale), k=15 | **+0.0798** |
| permutation null (999) | mean −0.0006, sd 0.0073 |
| **z** | **+10.9** |
| p | 0.0010 |
| same test on the binary *was-it-revised* field | I = +0.0737, z = +9.8, p = 0.0010 |

The null is rejected without ambiguity. But **I = 0.08 against +0.42 for the synthetic positive
control** — the structure is real and small.

Churn is heterogeneous by category, an 8× spread in mean displacement:

| Category | mean m | % changed | | Category | mean m | % changed |
|---|---|---|---|---|---|---|
| econ.EM | 0.2277 | 52% | | cs.DB | 0.0708 | 22% |
| stat.ME | 0.1649 | 41% | | cs.CR | 0.0570 | 20% |
| math.PR | 0.1158 | 32% | | eess.SP | 0.0410 | 12% |
| math.NT | 0.1141 | 36% | | astro-ph.GA | 0.0270 | 14% |

Category explains only **η² = 0.066** of displacement variance. The variogram shows the
autocorrelation is **very short range** — E|Δm| = 0.126 at neighbour rank 1, rising to ~0.155 by
rank 4 and then flat. Only the immediate neighbourhood carries signal.

### 11.3 Position is not predictable; magnitude is **[M]**

This is the test that carries the novelty argument, and it comes out
cleanly in the expected direction.

| Probe | cos(predicted, fresh) | cos(stale, fresh) — do nothing | verdict | magnitude ρ |
|---|---|---|---|---|
| 2% | 0.9682 | **0.9769** | **no gain — it hurts** | +0.043 |
| 10% | 0.9716 | **0.9766** | **no gain — it hurts** | +0.159 |

Interpolating the displacement **vector** field is worse than leaving the stale vector alone.
Predicting the **magnitude** works, weakly. Drift-Adapter's global map exists for a model swap; under
content revision it does not, exactly as argued.

### 11.4 probe-kNN vs random on a real retrieval task **[M]**

495 targeted queries — sentence windows that exist only in the revised abstract. recall@10, 3 seeds.

Stale index **0.9152**, fresh index **0.9576**, so the entire headroom is **0.0424**.

| Budget | probe-kNN | random | oracle | change-log | share of gain | **vs random** |
|---|---|---|---|---|---|---|
| 5% | 0.9199 | 0.9172 | 0.9475 | 0.9212 | 0.111 | **2.33×** |
| 10% | 0.9226 | 0.9185 | 0.9515 | 0.9266 | 0.175 | **2.20×** |
| 20% | 0.9253 | 0.9185 | 0.9576 | 0.9455 | 0.238 | **3.00×** |
| 35% | 0.9354 | 0.9246 | 0.9576 | 0.9576 | 0.476 | 2.14× |
| 50% | 0.9448 | 0.9333 | 0.9576 | 0.9576 | 0.698 | 1.63× |

### 11.5 What replicated and what did not

**Replicated.** The *ratio* to random survives contact with real data, including the headline:
simulator 3.0× at a 20% budget, real data **3.00×**. The ordering of selectors holds. The
position-vs-magnitude asymmetry holds.

**Did not replicate.** The *absolute* claims do not, and the §5.4 figures must not be used:

| Quantity | Simulator (§5.4) | Real (§11.4) |
|---|---|---|
| Headroom (staleness cost) | 0.166 | **0.0424** |
| Share of gain recovered at 20% budget | 59.6% | **23.8%** |
| Moran's I of displacement | +0.42 (synthetic control) | **+0.0798** |
| probe-kNN Spearman | +0.44 (8-cluster control) | **+0.122** at a 2% probe |

**Consequences.**

- **"2.9× less embedding compute than a full re-index for a 3.9-point shortfall" does not hold on
  real data and must not appear in a paper or an abstract.** On arXiv, probe-kNN does not
  reach full-re-index quality even at a 50% budget.
- What survives is: *at equal compute budget, probe-kNN recovers more of the recoverable retrieval
  quality than random selection.* That is still a measurable compute-efficiency effect, but it is a
  different and weaker claim. **Superseded by §12.4: the ratio is 1.26–1.58×, not 2–3×.**
- The **change-log baseline reaches full quality at a 35% budget**, versus 47.6% of the gain for
  probe-kNN. This quantifies §3's applicability boundary precisely: where a change log exists, use
  it. The claim's value is confined to corpora that do not expose one.
- The gap between simulated and real is itself a result, and is worth reporting: a natural-looking
  simulator overstated the recoverable share by 2.5× and the effect size by 5×.

### 11.6 Caveat — arXiv is probably a conservative case

arXiv revisions are per-paper editorial events driven by peer review, which is close to independent
across papers. Corpora where a whole topic or product line is updated together — documentation,
wikis, product catalogues — should show stronger regional concentration. **The number here is
plausibly a lower bound, not a typical value**, and the Wikipedia collector
([`e6_fetch_wikipedia.py`](research/gates/e6_fetch_wikipedia.py)) exists to test that; it did not run
because the API rate-limited this host after an early attempt used six concurrent workers. Re-run it
before concluding that 0.08 is the general figure.

---

## 12. Overnight replication across three corpora **[M]**

Added 6 September 2026, 05:30. An autonomous driver collected, embedded and analysed three real
corpora spanning three revision regimes. Everything below is measured on real revisions with real
embeddings. **This section supersedes §11.4 and §11.5, and materially revises §6.**

### 12.1 The three corpora

| Corpus | Regime | n | groups | revised | mean displacement |
|---|---|---|---|---|---|
| **Python docs** 3.8 → 3.13, section level | documentation | 1,751 | 302 pages | 80.8% | 0.164 |
| **arXiv** v1 → latest, 24 categories, 2022–23 | independent per-paper editing | 7,346 | 24 | 25.2% | 0.080 |
| **Wikipedia** lead sections 2023 → 2026 | wiki | 344 | 20 hubs | 89.2% | 0.263 |

The Wikipedia collector recovered only 344 usable documents — enough for the autocorrelation test
but, as §12.4 shows, not enough for anything else.

### 12.2 Displacement is spatially autocorrelated in all three regimes **[M]**

Same k=15 kNN graph on the stale embeddings, same 999-permutation null, in every corpus.

| Corpus | Moran's I | z | p | binary *was-it-revised* field |
|---|---|---|---|---|
| Python docs | **+0.1184** | +15.4 | 0.001 | +0.1055 (z = +13.8) |
| arXiv (n=7,346) | **+0.0557** | +15.0 | 0.001 | +0.0529 (z = +14.6) |
| Wikipedia (n=344) | +0.0311 | +1.8 | 0.043 | +0.0222 (z = +1.4) |

**Ordering: documentation > arXiv > wiki.** §11.6 predicted arXiv would be a lower bound relative to
a corpus where whole topics are revised together; that is confirmed for documentation, and the wiki
corpus is too small to rank.

**A caution on §11's own number.** The small arXiv sample (n=1,786) gave I = +0.0798. With 4× the
documents the estimate falls to **+0.0557** while z rises from +10.9 to +15.0 — the significance
came from n, and the earlier effect size was inflated. Use +0.0557.

**Do not compare corpora using η².** Raw η² is 0.355 (docs, 302 groups) vs 0.033 (arXiv, 24 groups),
which is an artifact of group count. Permutation-tested against a shuffled null: docs 0.3546 vs null
0.1723 = **2.06×**; arXiv (n=1,786) 0.0662 vs null 0.0075 = **8.84×**. Relative to its own null the
arXiv category effect is the stronger one. Moran's I is the only defensible cross-corpus statistic
here.

### 12.3 Position is not predictable; magnitude is — 6 of 6 conditions **[M]**

| Corpus | probe | cos(predicted, fresh) | cos(stale, fresh) — do nothing | verdict | magnitude ρ |
|---|---|---|---|---|---|
| Python docs | 2% / 10% | 0.9563 / 0.9608 | **0.9680 / 0.9677** | no gain | +0.071 / +0.208 |
| arXiv | 2% / 10% | 0.9760 / 0.9789 | **0.9817 / 0.9818** | no gain | +0.083 / +0.107 |
| Wikipedia | 2% / 10% | 0.9154 / 0.9108 | **0.9438 / 0.9460** | no gain | +0.039 / −0.010 |

Interpolating the displacement **vector** field is worse than leaving the stale vector alone, in
every corpus at every probe size. Predicting **magnitude** works. This is the most robust finding of
the whole project.

probe-kNN's rank correlation with true displacement, against a spatial-shuffle null of ≈0:

| Corpus | 2% probe | 5% | 10% | 20% |
|---|---|---|---|---|
| Python docs | +0.117 | +0.125 | +0.200 | **+0.228** |
| arXiv | +0.076 | +0.086 | +0.120 | +0.145 |
| Wikipedia | +0.071 | +0.051 | +0.087 | +0.096 |

### 12.4 The retrieval gain is real, small, and needs a large corpus to see **[M]**

30 seeds, bootstrap CI on the share-of-gain ratio, two-sample t-test. Script: `significance.py`.

| Corpus | n | budget | probe-kNN | random | ratio | 95% CI | p |
|---|---|---|---|---|---|---|---|
| **arXiv** | **7,346** | 10% | 0.8912 | 0.8895 | **1.58×** | [1.24, 2.04] | **0.0006** |
| | | 20% | 0.8957 | 0.8930 | **1.43×** | [1.24, 1.68] | **<0.0001** |
| | | 35% | 0.9024 | 0.8992 | **1.26×** | [1.16, 1.37] | **<0.0001** |
| arXiv | 1,786 | 20% | 0.9246 | 0.9238 | 1.09× | [0.89, 1.35] | 0.39 |
| Python docs | 1,751 | 35% | 0.7706 | 0.7689 | 1.19× | [1.01, 1.43] | 0.043 |
| Wikipedia | 344 | — | 0.9933 | 0.9933 | — | headroom **0.0000** | vacuous |

**The effect on displacement-ranked selection is real** — three independent significant results at
n = 7,346, all CIs excluding 1.0. **It is roughly 1.3–1.6×, not the 2–3× reported in §11.5.**

The seed- and size-sensitivity is the lesson:

| | ratio at 20% budget | p |
|---|---|---|
| n=1,786, 3 seeds (as published in §11.4) | 3.00× | — |
| n=1,786, 12 seeds | 1.92× | 0.055 |
| n=1,786, 30 seeds | 1.09× | 0.39 |
| **n=7,346, 30 seeds** | **1.43×** | **<0.0001** |

§11.4's 3.00× was a 3-seed artifact. The true effect is ~1.4× and needs roughly 7,000 documents to
resolve. Wikipedia at n=344 has **zero headroom** — recall@10 is saturated at 0.9933 stale and fresh
alike — so no selective-re-index claim can be tested on a corpus that small.

### 12.5 probe-kNN does not beat the closest prior art — the decisive comparison **[M]**

Beating random is necessary but not sufficient. The comparison that decides novelty is
against **cluster-threshold adaptive refresh** (cluster the stale index, re-embed
the clusters whose probe-measured drift is highest). Both methods are given the **same 2% probe**, so
this isolates the claimed mechanism — kNN interpolation — from the probe itself.
30 seeds, n = 7,346. Script: `pk_vs_prior.py`.

| Budget | probe-kNN | cluster-threshold | difference | t | p | verdict |
|---|---|---|---|---|---|---|
| 10% | 0.8912 ± 0.0020 | **0.8923 ± 0.0026** | −0.0010 | −1.72 | 0.091 | tie |
| 20% | 0.8957 ± 0.0021 | **0.8965 ± 0.0025** | −0.0008 | −1.35 | 0.182 | tie |
| 35% | 0.9024 ± 0.0023 | **0.9027 ± 0.0024** | −0.0003 | −0.57 | 0.574 | tie |

**probe-kNN never beats the prior art, and is numerically behind at every budget.** Given a probe,
aggregating by cluster is as good as kNN interpolation. The specific claimed mechanism adds nothing
measurable.

The change-log oracle is far ahead of both — 0.9003 / 0.9158 / 0.9250, reaching full re-index quality
at a 35% budget.

**Consequence.** A reviewer assessing novelty would find cluster-threshold refresh in the
practitioner literature and ask what probe-kNN adds. On this evidence the answer is *nothing
measurable*. §5.2's algorithm should **not** be presented as a contribution in its own right.

### 12.6 The blind spot does not reproduce on real revisions **[M]**

§6 is Part 1's headline and had only ever been shown on a simulator, where in-place and migrating
revisions caused identical damage (0.3015 / 0.3135) at a **33× difference in corpus-side W₁**.

The binary in-place/migrating split cannot be reproduced on real data: only **~2% of real revisions
move a document out of its own neighbourhood** (35 of 1,852 in arXiv), which is itself a finding but
leaves that comparison underpowered.

Tested directly instead, per document, on every revised document — does displacement magnitude (what
drives corpus-side W₁/MMD/centroid) predict actual retrieval damage? Script: `blindspot2.py`.

| | arXiv (1,826 queries) | Python docs (1,270 queries) |
|---|---|---|
| documents damaged (fresh finds, stale misses) | 77 (4.2%) | 50 (3.9%) |
| mean displacement, damaged | 0.5440 | 0.3996 |
| mean displacement, undamaged | 0.3105 | 0.1986 |
| **AUC of displacement predicting damage** | **0.769** | **0.782** |
| point-biserial r | +0.224 (p = 4e−22) | +0.199 (p = 8e−13) |
| centroid shift, damaging vs non-damaging (matched size) | **1.68×** | **1.84×** |

Damage rate rises monotonically with displacement (arXiv quintiles: 0.000 → 0.019 → 0.036 → 0.036 →
0.120).

**On real revisions, displacement is a decent predictor of retrieval damage (AUC ≈ 0.77–0.78), and
damaging documents generate ~1.7–1.8× more corpus-side signal than non-damaging ones — not equal
signal.** The simulator's 33× blind spot is an artifact of how in-place revision was modelled there
(a fresh draw within the same cluster), which is far more aggressive than what real revisions do.

**§6 must be substantially weakened.** The honest claim is that corpus-side signal is an *imperfect*
predictor of retrieval damage (AUC 0.77), not a blind one.

### 12.7 Where this leaves the project

| Claim | Status after tonight |
|---|---|
| Displacement is spatially autocorrelated on real corpora | **Holds** — z = +15.4, +15.0; three regimes |
| Position unpredictable, magnitude predictable | **Holds** — 6 of 6 conditions |
| probe-kNN predicts displacement above chance | **Holds** — ρ +0.05…+0.23 vs null ≈ 0 |
| Displacement-ranked selection beats random | **Holds, but small** — 1.26–1.58×, needs n ≈ 7,000 |
| probe-kNN beats the closest prior art | **Refuted** — ties or loses at every budget |
| Corpus-side drift is blind to retrieval damage | **Refuted** — AUC 0.77, not blind |
| Thresholds fail to transfer across traffic volume | **Holds on real embeddings** — FAR 1.000 / 0.000 (§12.8) |
| Detection power at realistic effect sizes | **At the false-alarm floor** (§12.9) |

Two of the six above are refuted, the central selective-re-index claim among them. Part 1, however,
survives on §12.8 and §12.9, which are independent of the refuted blind-spot claim — see §12.10.


### 12.8 Gate 1 reproduces on real embeddings — open question #2 answered **[M]**

Open question #2 in §10: does the calibrated-threshold failure under traffic-volume shift hold on
real vectors, or was it a property of the simulator? Pool = 7,346 real arXiv embeddings, 60 trials,
96 projections, both windows drawn from the *same* distribution so every alarm is false.
Script: `gate1_fast.py`.

| Statistic | threshold (95th pct at n=4,000) | mean score at n=400 | FAR at n=4,000 | **FAR deployed at n=400** |
|---|---|---|---|---|
| sliced-W₁ mean | 0.00149 | 0.00331 | 0.050 | **1.000** |
| sliced-W₁ max | 0.00352 | 0.00731 | 0.050 | **1.000** |
| centroid shift | 0.02392 | 0.04962 | 0.050 | **1.000** |

Reverse direction — calibrate at n=400, deploy at n=4,000 — gives **FAR = 0.000** for all three: the
detector goes completely silent.

**The mechanism is confirmed.** Predicted score inflation from (n_cur^(−½) + n_ref^(−½)):
**2.081**. Measured: 2.417 (sliced-W₁ mean), 2.426 (max), 2.349 (centroid). The prediction is
approximate — measured is ~15% higher — but the scaling law is the right one and the direction and
magnitude are correct.

This is the one result of the night that lands unambiguously in the project's favour, and it is
Part 1's strongest remaining claim: **a drift threshold calibrated at one traffic volume is
worthless at another**, on real embeddings, for every statistic tested.


### 12.9 Detection power at realistic effect sizes is zero — open question #3 answered **[M]**

Open question #3 in §10 said the existing design saturated power at 1.000 in every cell, so power
collapse could not be claimed. Re-run on real embeddings with effect sizes swept **down through the
detection floor**: 7,346-vector pool, n=800 per window, threshold calibrated to a 5% false-alarm
rate from the null, 60 trials per cell. Script: `power_floor.py`.

Power at a calibrated 5% FAR, 30% of documents revised:

| perturbation scale | mean displacement induced | sliced-W₁ mean | sliced-W₁ max | centroid | MMD |
|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.033 | 0.000 | 0.033 | 0.033 |
| 0.10 | 0.100 | 0.033 | 0.017 | 0.033 | 0.050 |
| 0.20 | 0.197 | 0.083 | 0.050 | 0.100 | 0.100 |
| 0.35 | 0.335 | 0.050 | 0.000 | 0.033 | 0.033 |
| 0.60 | 0.532 | 0.133 | 0.150 | 0.100 | 0.133 |
| 1.00 | 0.765 | **0.483** | 0.133 | **0.483** | **0.567** |

Nothing reaches 50% power until the induced displacement is **0.765** — larger than any real corpus
measured here. Mapped onto the three corpora:

| Corpus | mean displacement over all documents | fraction revised | **power** |
|---|---|---|---|
| arXiv (n=7,346) | 0.0803 | 25% | **0.000 – 0.083** |
| Python docs | 0.1641 | 81% | **0.050 – 0.100** |
| Wikipedia | 0.2628 | 89% | **0.000 – 0.050** |

**At every real corpus's actual effect size, power is indistinguishable from the 5% false-alarm
rate.** These detectors are not merely badly calibrated — against realistic corpus revision at this
window size they carry no information at all.

This is a stronger result than the one §10 asked for. It does not say power *collapses near a
floor*; it says the realistic operating regime **is** the floor.

**Limitation, stated plainly:** this is at n=800 documents per window, and power grows with window
size. A larger window would detect more. But §12.8 shows that enlarging the window changes the score
distribution and invalidates the calibrated threshold — so "just use more data" is not available
without recalibrating, which is exactly the failure mode §12.8 documents. The two results compound
rather than cancel.

### 12.10 Revised standing of Part 1

§12.6 withdrew the blind-spot claim. Part 1 does not depend on it, and now rests on two independent
results, both measured on real embeddings:

1. **Thresholds do not transfer across traffic volume.** Calibrate at n=4,000 → FAR **1.000** at
   n=400; calibrate at n=400 → FAR **0.000** at n=4,000. 200 trials, three statistics, no exceptions
   (§12.8).
2. **Power at realistic effect sizes is at the false-alarm floor** (§12.9).

Together: a corpus-side distributional drift detector for RAG cannot be calibrated to a stable
operating point, and at the point where you would operate it, it cannot see the phenomenon anyway.
That is a complete and defensible negative result, and it no longer needs the blind-spot claim.

## 13. Pipeline rebuild — 8 September 2026 **[M]**

The shipped pipeline was rebuilt on the results that survived §12, and now runs end to
end (`python main.py run-all`). Every number below is from that run: 7,346 arXiv
documents, 12 windows, 300 queries per window, real revision timestamps, MiniLM-L6-v2
evaluated in numpy. §2 of `STATUS.md` maps each change to the finding that forced it.

### 13.1 The calibration is correct on real data

Over the 11 comparison windows of the real slice, the null-standardised corpus channel
has **mean −0.295, sd 0.955** — a standard normal to within sampling error, which is
what a correctly built permutation null must give when nothing is detectable. The query
channel is comparable (mean −0.303, sd 0.664). Zero alarms in 11 windows.

The raw statistic those z-scores come from is **W₁ = 0.0034**. The legacy fixed
threshold τ = 0.15 therefore sits **44×** above anything the corpus produces, and on the
null slice it scored FAR = **0.000** — a *vacuous* pass of the ≤5% target, achieved by
never firing at all. The calibrated path on the same slice scored 0.091 (1 alarm in 11),
with granularity also 1/11 = 0.091, so at this window count the 5% target is not
expressible. That is F21 stated by the code rather than discovered later.

`w1_c` interval width averaged **6.6e−4** against a slicing error of **7.5e−5** — a
ratio of **8.8×**, confirming on real data that the interval now tracks sampling
uncertainty rather than shrinking with projection count (F12).

Measured share of the *legacy* D(t) contributed by the query-to-corpus genre gap:
**91.4%**, on real embeddings. F5 confirmed outside the simulator.

### 13.2 §12.8 reproduces on the rebuilt pipeline

Real arXiv embeddings, calibrate at n = 3000, deploy at n = 300, 60 trials, all windows
drawn from the same distribution so every alarm is false:

| statistic | null mean @3000 | null mean @300 | inflation | FAR big→small | FAR small→big | native FAR |
|---|---|---|---|---|---|---|
| sliced-W₁ | 0.001595 | 0.004901 | **3.072** | **1.000** | **0.000** | 0.050 |
| centroid  | 0.024815 | 0.077180 | **3.110** | **1.000** | **0.000** | 0.050 |
| MMD       | −0.000004 | −0.000007 | undefined | 0.350 | 0.000 | 0.050 |

Predicted inflation from √(3000/300) is **3.162**; measured 3.072 and 3.110. The
per-window permutation null, run at the same two sizes, stayed near nominal (0.100 at
n = 3000, 0.033 at n = 300, against 0.05 with a granularity of 1/30).

MMD's inflation ratio is undefined rather than missing: the unbiased MMD² estimator is
centred on zero by construction, so a ratio of null levels carries no information. Its
transfer failure is still visible in the FAR columns.

### 13.3 Correction — §12.9 mixes two displacement conventions **[correction]**

`e6_power_floor.py` sweeps a perturbation `scale` and never records a displacement; the
"mean displacement induced" column in §12.9 (0.050 … 0.765) was added at write-up and is
the displacement of the documents that **moved**. Perturbing a fraction *f* of documents
by a step of size *s* gives a mean of *s* over the moved documents and *f·s* over all of
them. The corpus figures §12.9 compares against — arXiv 0.0803, Python docs 0.1641,
Wikipedia 0.2628 — are means over **all** documents. The two are mixed on one axis.

The rebuilt sweep reports both columns. At f = 0.30 the moved-docs column reproduces
§12.9's axis exactly, which settles the diagnosis:

| scale | displacement, all docs | displacement, moved docs | §12.9's column | power (sliced-W₁) |
|---|---|---|---|---|
| 0.05 | 0.015 | 0.050 | 0.050 | 0.025 |
| 0.10 | 0.030 | 0.100 | 0.100 | 0.025 |
| 0.20 | 0.059 | 0.197 | 0.197 | 0.125 |
| 0.35 | 0.100 | 0.335 | 0.335 | 0.150 |
| 0.60 | 0.160 | 0.534 | 0.532 | 0.200 |
| 1.00 | 0.230 | **0.765** | **0.765** | 0.900 |

Measured on this build, arXiv sits at **0.080 over all documents and 0.330 over the 24%
that moved**. Read on the matching axis its power is ≈ 0.15 either way (all-docs 0.080
falls between the 0.125 and 0.150 rows; moved-docs 0.330 lands on the 0.150 row). §12.9
read 0.0803 against the *moved* axis, which understates arXiv by roughly 5× — power
≈ 0.03 rather than ≈ 0.15.

**The conclusion is unchanged**: 0.15 is still far below 50% and close to the 5% floor,
and the same holds for Python docs (≈ 0.20). The two conventions happen to agree here
because arXiv's moved fraction, 24%, is close to the sweep's 30%. But the sentence
"nothing reaches 50% power until the induced displacement is 0.765 — larger than any
real corpus measured here" compares a per-moved sweep value against per-all corpus
values and **should not be quoted in that form**.

### 13.4 Two defects the rebuilt pipeline's own null slice caught

Both produce confident nonsense rather than errors, which is why they are worth
recording.

**Overlapping windows break exchangeability.** The permutation null assumes the two
windows are independent samples. When consecutive corpus windows contain the same
documents, the observed split pairs each document with itself and is more balanced than
any reshuffle of the pooled set, so the observed statistic sits below the entire null.
Measured: **z_c pinned near −5 in every window**, including windows with real revisions,
while the query channel — whose windows are drawn independently — behaved correctly at
z ~ N(0,1). Fixed by giving the detector a disjoint sample view, separate from the
paired view retrieval uses.

**A null slice must vary only by sampling.** Drawing each null window's queries from
that window's own document group made successive query windows samples of *different
content* — a real distributional difference, not a null. It alarmed in **5 of 5**
windows at z up to +4.9. Fixed by drawing null queries from one fixed pool.

### 13.5 Retrieval headroom requires targeted queries

Queries drawn from any live sentence are almost all answerable from the stale index,
because most of a revised document is unchanged. On the 7,346-document build this gave
stale recall@10 = full-re-index recall@10 = **0.9433, headroom 0.0000** — every
selective-re-index comparison vacuous, the same "no headroom" condition §12 recorded for
Wikipedia. The bake-off in §12.4–12.5 used *targeted* queries, drawn from sentences
present only in the revision; that is now a first-class setting
(`DATA_QUERY_TARGETED_FRAC`) rather than a property of one script.

This is a limitation of the measurement, not of the method: it means the retrieval
effect is only observable when queries actually concern the changed text, which is worth
stating plainly in any write-up.


### 13.6 The bake-off runs, and is far too underpowered to rank the selectors

With targeted queries the comparison is no longer vacuous: stale recall@10 = 0.9467,
full re-index = 0.9733, **headroom +0.0267**. Share of recoverable recall captured,
n = 7,346, 3 seeds:

| budget | random | probe-kNN | cluster-threshold | change log | oracle |
|---|---|---|---|---|---|
| 5%  | −0.042 | 0.000 | 0.125 | 0.167 | 0.50 |
| 10% | 0.167 | 0.000 | 0.167 | 0.458 | 0.75 |
| 20% | 0.208 | 0.208 | 0.208 | 0.750 | 1.00 |
| 35% | 0.292 | 0.417 | 0.375 | 1.000 | 1.00 |
| 50% | 0.500 | 0.583 | 0.500 | 1.000 | 1.00 |

The qualitative ordering is right — oracle ≥ change log ≫ probe-kNN ≈ cluster-threshold
≥ random — and consistent with §12.5 and §12.4. **No number in this table should be
quoted.** The headroom of 0.0267 over 300 queries is **8 queries in total**, so one
query changing hands moves share-of-gain by 0.125. Every visible difference between
probe-kNN and cluster-threshold is smaller than that quantum, and there are only 3
seeds.

This is the same failure mode as the withdrawn 3.00× headline: a difference read off a
table whose resolution cannot support it. Deciding probe-kNN versus cluster-threshold
needs §12.5's design — far more queries per window and 30 seeds — not this table. What
this run establishes is that the machinery is correct and instrumented, not what the
answer is.


---

*Findings recorded 5 September 2026; §11 added 6 September 2026; §12 added and completed
6 September 2026 05:30-07:10; §13 added 8 September 2026 after the pipeline rebuild,
including a correction to §12.9's displacement convention. Not legal advice.*
