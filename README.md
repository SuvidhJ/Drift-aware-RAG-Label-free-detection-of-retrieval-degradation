# Drift-Aware RAG

Label-free detection of retrieval degradation, from the geometry of the embedding
space.

A RAG system's retrieval quality decays as its corpus is revised and its query stream
moves. This measures that decay **without relevance labels**: compare consecutive
windows of query and corpus embeddings, calibrate every comparison against its own
permutation null, classify which of four regimes you are in, and choose a remediation
action by expected cost.

> **Read [STATUS.md](STATUS.md) first** for what is established and what is not.
> [FINDINGS.md](FINDINGS.md) holds every measurement, including the ones that
> overturned earlier claims. Numbers withdrawn during validation are listed in
> STATUS.md §4 and must not be quoted.

---

## What this actually shows

Two results hold up on real revision data and are reproducible from this repository:

1. **A drift threshold does not transfer across traffic volume.** Calibrate at
   n = 3000 and deploy at n = 300 and the false-alarm rate is 1.000; reverse it and
   the detector goes silent at 0.000. Both windows are drawn from the same
   distribution throughout, so every alarm is false. `python main.py transfer`
2. **At realistic effect sizes, detection power sits near the false-alarm floor.**
   Power is ≈ 0.15 on arXiv and ≈ 0.20 on Python docs at n = 1500, against a 5%
   false-alarm floor. `python main.py power`

Together: a corpus-side distributional drift detector cannot be pinned to a stable
operating point, and at the point where you would operate it, it cannot see the
phenomenon anyway. This is why the pipeline reports z-scores and p-values rather than
thresholded alarms on a raw statistic.

Two structural results also hold — displacement is spatially autocorrelated, and its
*magnitude* is predictable from a probe while its *direction* is not — and they are
what make selective re-index possible at all. One claimed advantage did **not** hold:
given the same probe, probe-kNN selection ties the deployed prior art. See
`retrieval/select.py` and `python main.py bakeoff`.

---

## Install and run

Requires Python 3.11+. The default path needs no torch, no vector database and no LLM.

```bash
uv venv --python 3.11
uv pip install -e ".[dev,viz]"

# one-time: fetch the embedding checkpoint (~88 MB)
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

python main.py run-all
python main.py dashboard
```

`run-all` builds snapshots, embeds, computes calibrated drift, classifies regimes,
selects actions, runs the closed retrieval loop and writes a validation report into
`results/`.

### Data

The default corpus is a **real revision corpus**: documents observed at two real
timepoints with real edit timestamps.

| `DATA_CORPUS` | documents | revised | source |
|---|---|---|---|
| `arxiv` | 1,786 (7,346 with the big build) | 25–28% | v1 vs latest abstracts, 24 categories |
| `python_docs` | 1,751 | 81% | Python 3.8 vs 3.13, section level |
| `wikipedia` | 344 | 89% | lead sections, 2023 vs 2026 |

They live in `research/gates/results/`; point `DATA_CORPUS_DIR` elsewhere to use
another build. The collectors are `research/gates/e6_*fetch*.py`.

The Amazon Reviews 2023 path is still present (`DATA_SOURCE=amazon`) with its defects
repaired, but it is not recommended — it ships no question-answering split, so it has
no query stream.

---

## Commands

```bash
# pipeline
python main.py build-snapshots      # C(t), Q(t), disjoint drift-detection view, null slice
python main.py embed                # each distinct text encoded once, cached by content
python main.py compute-drift        # D(t) with a per-window permutation null
python main.py classify-regimes     # the (z_q, z_c) plane
python main.py run-policy           # argmin cost + lambda * regret
python main.py run-rag              # stale index -> action -> recall@k
python main.py evaluate             # rho, lead time, AUROC, FAR
python main.py run-all

# experiments
python main.py bakeoff              # selective re-index vs the prior art
python main.py transfer             # threshold transfer under volume shift
python main.py power                # power swept through the floor
python main.py audit                # structural self-checks
```

`python main.py audit` re-checks the properties this rewrite was built to guarantee:
all four regimes reachable, more than one policy action selectable, the null FAR
computable, and the confidence interval reflecting sampling rather than slicing error.

---

## How it works

**Snapshots.** Each document is represented by its earlier revision until its real
revision timestamp lands, then by its later one. Two views are written: `corpus_t*`
holds every document (the paired view an index has, used for retrieval), and
`sample_t*` holds a *disjoint* subset per window (the independent-sample view the
detector needs — overlapping windows break the permutation null's exchangeability and
drive the corpus z-score to about −5 regardless of what happened).

**Queries** are sentence windows drawn from live document text, so each has a known
gold document, and the stream drifts both because content is revised and because the
topic mixture moves.

**Drift.** Sliced Wasserstein-1 and RBF-MMD between consecutive windows, each
calibrated against a permutation null built at the window's own size. The reported
statistic is a z-score, which is comparable across window sizes; the raw distance is
not.

**Regimes.** The two absolute z-scores form a genuine 2×2 plane, gated by persistence.

**Policy.** `argmin_a [ cost(a) + λ · regret_scale · residual_loss(a) ]`, with regret
converted into compute-equivalent units so the comparison is dimensionally coherent.

**Actions.** `a3_reindex` performs a real budgeted partial refresh through
`NumpyIndex.reindex_rows`, choosing rows with one of five selectors.

**Endpoint.** recall@k and MRR against the known gold document. Not RAGAS
faithfulness, which scores a stale-but-plausible context well and is therefore blind
to the failure mode being measured. RAGAS remains available behind the `heavy` extra.

---

## Layout

```
main.py                CLI
configs/settings.py    all configuration, env-overridable
data/                  corpora.py, windows.py (+ legacy Amazon path)
embeddings/            backends.py (numpy MiniLM, optional sentence-transformers), encoder.py
drift/                 wasserstein.py, mmd.py, null.py, statistic.py, decomposition.py
retrieval/             index.py, select.py, rerank.py, rag_pipeline.py
policy/                costs.py, engine.py
evaluation/            retrieval_quality.py, correlate.py, null_corpus_test.py,
                       transfer.py, power.py, ragas_harness.py (optional)
streaming/             online_detector.py (CUSUM with ARL0 calibration)
dashboard/app.py       Streamlit
tests/                 126 tests
research/              second-generation simulator work
research/gates/        validation scripts, corpora and results behind FINDINGS.md
```

---

## Configuration

Every setting is overridable by environment variable, prefixed by section.

```bash
DATA_CORPUS=python_docs          # arxiv | python_docs | wikipedia
DATA_N_WINDOWS=12
DATA_QUERIES_PER_WINDOW=300
EMBED_BACKEND=minilm-numpy       # or sentence-transformers (needs the heavy extra)
DRIFT_N_PERM=199                 # permutations per window per channel
DRIFT_ALPHA_LEVEL=0.05
DRIFT_TAU_Z_QUERY=2.0
RETRIEVAL_BACKEND=numpy          # or chroma
RETRIEVAL_SELECTOR=probe_knn     # probe_knn | cluster_threshold | random | oracle | changelog
POLICY_LAM=1.0
OPENAI_ENABLED=false             # answer generation is off by default
```

## Tests

```bash
python -m pytest tests -m "not slow"   # ~35 s
python -m pytest tests                 # everything
```

Tests that need the embedding checkpoint are marked `needs_model` and skip cleanly
without it. Several encode a specific past defect — the ordering of bootstrap CI
variants, the reachability of all four regimes, the non-degeneracy of the policy
engine — so a regression fails loudly rather than quietly returning a plausible
number.

## Provenance

The audit that motivated this rewrite is `FINDINGS.md` §3: 25 defects with file and
line. The load-bearing ones were a constant query string, a policy engine that was a
provable no-op, a 2×2 regime plane that was one-dimensional, a confidence interval
measuring the wrong quantity, an MMD that allocated 16 GB per window, and a
false-alarm rate with no writer anywhere in the repository. Each module's docstring
names the finding it addresses.

## Attribution

The base pipeline was written by **Aryan Garg**. **Aditi Saxena** owned the repository
this work was forked from and ran the first analysis pass. The audit, the validation
programme under `research/`, and the rebuild recorded in `FINDINGS.md` §13 were built on
top of both. Modules carrying their own authorship notice keep it.

## Licence

MIT — see [LICENSE](LICENSE). [NOTICE.md](NOTICE.md) records the provenance above and the
terms of the third-party corpora redistributed under `research/gates/results/` (Wikipedia
is CC BY-SA 4.0; the Python documentation is under the PSF licence).
