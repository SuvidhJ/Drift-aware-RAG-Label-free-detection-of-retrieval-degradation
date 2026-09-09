# `research/` — the missing science for drift-aware-rag

Part I of the audit (`../FINDINGS.md` §3) found that the repo's four validation
metrics are computed on data with no injected drift: `inject-drift` writes its
labeled and null slices to directories that `compute-drift` / `run-rag` /
`evaluate` never read. This directory builds the pieces that were missing and
**runs** the experiment programme, against a controllable embedding-space world
with known ground truth.

Self-contained: `numpy`, `scipy`, `pandas`, `scikit-learn`, `POT`. No embedding
model, no LLM, no network. POT is the only one not in the base install, so it has
its own extra:

```
uv pip install -e ".[research]"         # adds POT, for exact optimal transport

python -m research.run_all              # full programme  (~40 min, 1 core)
python -m research.run_all --fast       # smaller grid    (~10 min)
python  research/aggregate.py           # results/*.{csv,json} -> results/report_data.json
```

## Layout

| path | what |
|---|---|
| `dar/simulator.py`  | embedding-space world: Zipf-clustered catalogue on S²⁵⁵, questions = product + relative noise, rolling "seen" memory, two injectable drift channels (whole-line delisting; held-out-line query shift), lognormal query-volume seasonality |
| `dar/stats.py`      | sliced-W₁ (mean **and** max), Gram-identity RBF-MMD (fixes the repo's 16 GB allocation), **`permutation_null`** — the per-window calibration the repo never had; data bootstrap, not projection-seed bootstrap |
| `dar/detect.py`     | `d_original_series` (faithful repo reproduction), `d_calibrated_series` (z-scored components, difference-based centroid, two-test 2×2 regime), and 7 baselines |
| `dar/streaming.py`  | `FrozenCUSUM` with frozen (μ₀,σ₀) + `arl0_calibrate`; `rolling_cusum_repo` for comparison |
| `dar/theory.py`     | 1-Lipschitz retrieval functional `rho_C` + empirical Kantorovich–Rubinstein bound check (exact vs sliced W₁) |
| `dar/policy.py`     | harm model + true-argmin oracle + fixed-policy baselines |
| `experiments/harness.py` | the 2×2 factorial and its analysis (AUROC, FAR/power, detection delay, regime confusion, seasonality confound) |
| `experiments/auxiliary.py` | null-vs-sample-size, CUSUM ARL₀, KR bound, policy sweep |
| `run_all.py`        | orchestrates everything, writes `results/` incrementally |
| `aggregate.py`      | collapses `results/` into one JSON for the write-up |

## What each experiment answers

- **`null_vs_sample_size`** (S1) — raw sliced-W₁ between two i.i.d. samples decays
  as `n^-1/2` with *no drift present*; the permutation z-score does not. Raw D(t)
  partly measures question volume.
- **`factorial`** — the crossed query×corpus injection. Regime confusion matrix
  for the original α̂/β̂ classifier vs the two-test classifier; per-channel AUROC;
  detection power at a matched 5 % false-alarm rate; detection delay.
- **`cusum`** (S3) — in-control ARL₀ of the repo's rolling-reference CUSUM vs a
  frozen-reference CUSUM at matched slack.
- **`kr_bound`** (theory) — `|ΔR| ≤ W₁` holds for exact W₁, is violated for
  mean-sliced W₁ (a lower bound on W₁).
- **`policy`** — realised cost of the calibrated policy, the repo policy, and four
  fixed policies, vs a true-argmin oracle, swept over the regret weight λ.

## Caveat

This is a synthetic testbed. Absolute effect sizes depend on its parameters. The
transferable results are the **comparisons** — original vs calibrated pipeline,
and each detector vs the cheap baselines — and the **mechanisms**, all of which
are properties of high-dimensional embedding geometry rather than artefacts of
the toy.
