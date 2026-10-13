# Paper basis: stale RAG indexes fail at the answer, not at retrieval

This is the source document for a later paper draft. It records what exists, what was
found, what is and is not claimed, and why the work was needed. Every number below is
taken from a results file in this repository (listed in §9); nothing is rounded up.

---

## 1. The finding in one paragraph

When a document collection is revised and the retrieval index is not refreshed, the
stale index still **retrieves the right document as often as a refreshed index does**:
on questions whose answer was changed by the revision, the retrieval gap is −0.029 to
+0.010 (hit@10, two models, two corpora). But the stale copy it retrieves **holds the
outdated fact**, so the correct answer is present in the retrieved context far less
often (gap +0.52 to +0.79). A real generator then states the outdated value: on arXiv,
81% of the answers a 1.5B model committed to from the stale index were the replaced
number, against 0% from the fresh index (measured before the 2026-10-09 corpus cleanup,
§4.3). On facts that did not change, stale and fresh
agree at every level. **Staleness is a retrieval success on outdated content, so
retrieval-level metrics cannot measure it.**

---

## 2. Why this was needed

### 2.1 The original project and why it failed

The repository began as a *label-free drift detector for RAG*: monitor the distribution
of embeddings (queries and corpus) and raise an alarm when retrieval degrades, without
relevance labels. An independent audit and a series of executed experiments
(`FINDINGS.md`) ended in negative results:

- **Thresholds do not transfer across traffic volume.** Calibrated at n = 4,000 queries,
  the false-alarm rate at n = 400 is 1.000; calibrated at n = 400, it is 0.000 at
  n = 4,000 (`FINDINGS.md` §12.8, real embeddings, 200 trials, three statistics).
- **Detection power at realistic effect sizes is near the false-alarm floor.** At a
  calibrated 5% false-alarm rate, power is ≈ 0.15 at arXiv's real revision size and
  ≈ 0.20 at Python docs' (`FINDINGS.md` §12.9 as corrected in §13.3; the uncorrected
  §12.9 table understated arXiv and must not be quoted).
- The selective re-index method (probe-kNN) did not beat the closest prior art, and the
  "blind-spot" claim did not reproduce on real revisions (§12.5, §12.6).

### 2.2 The question those results left open

All of those detectors were validated against **retrieval degradation** (recall@k
changes). That assumes staleness shows up as retrieval degradation. Nobody in the
project, and as far as we know few evaluations in practice, checked that assumption.
This work checks it, and finds that it is largely false: a fully stale index costs
about one point of hit@1 or less (§4.1). That explains the failed detectors without
needing any detector-specific defect: they were trying to detect a signal that is
almost absent at the level they measured.

### 2.3 Why it matters outside this repository

Common RAG tooling does not remove superseded document versions by default. Read from
source during this work (versions must be pinned and re-checked before the paper cites
them): LangChain's `add_documents` assigns fresh `uuid4` IDs, and its indexing API
defaults to `cleanup=None`; LlamaIndex `insert()` adds a second copy, and
`refresh_ref_docs` is not called automatically; Chroma `add` with an existing ID logs a
warning and drops the write, while `upsert` updates. So stale or duplicated versions
are a realistic state of a deployed index, and if teams monitor it with retrieval
metrics they will see almost nothing.

---

## 3. What we have

| Item | Where | Notes |
|---|---|---|
| arXiv corpus, two versions per paper | `research/gates/results/ax_big_corpus.json` | 6,835 rows = 6,247 papers (cross-listed papers appear once per category; `data.load_corpus` keeps the first row per `pageid`), v1 vs latest version, from the arXiv API; 1,571 differ in raw text, 1,568 after whitespace/case canonicalisation |
| Python docs corpus | `research/gates/results/pd_corpus.json` | 1,751 Python 3.8 → 3.13 documentation sections; `data.load_corpus` cuts the Sphinx sidebar and footer, after which 1,346 are revised (1,383 before the cut: 37 differed only in page chrome) |
| Two embedding models | `embeddings/backends.py` | all-MiniLM-L6-v2 (mean pooling), bge-small-en-v1.5 (CLS pooling), run in the repo's numpy BERT, validated against HuggingFace |
| Query pools | `research/staleness/queries.py` | `new`, `retained`, `unchanged` text, gold document known |
| Fact-edit definitions | `research/staleness/facts.py` | text-only, no embeddings, numbers compared by value |
| Retrieval-level cost | `research/staleness/measure.py`, `run_headline.py` | stratified bootstrap over queries |
| Answer-level evaluation | `research/staleness/answer_level.py` | same questions, plus a control set |
| Generator confirmation | `research/staleness/generation.py`, `generate_hf.py` | Qwen2.5-1.5B-Instruct, greedy decoding, top-3 context |
| Tests | `tests/test_staleness.py` | pins every definition the numbers depend on |
| Text primitives | `drift/freshness.py` | `canonical`, `shingles`, `jaccard`; the version registry and `drift/topics.py` are in `quarantine/` (§6) |

Wikipedia (`wp_corpus.json`) was used in the earlier `FINDINGS.md` work but not in the
answer-level experiments.

---

## 4. Results

All comparisons are a **fully stale index** (old versions only) against a **fully
fresh index** (new versions only), on the same questions.

### 4.1 Retrieval level: a stale index costs about one point or less

Net gain of refreshing, fresh minus stale, over a realistic query mix (weights =
each pool's share of the corpus), 95% bootstrap CI (`results_headline.json`):

| model / corpus | hit@1 | hit@10 | MRR |
|---|---|---|---|
| MiniLM / arXiv | +0.0050 [+0.0017, +0.0084] | +0.0034 [+0.0011, +0.0058] | +0.0039 [+0.0019, +0.0060] |
| bge / arXiv | +0.0015 [−0.0017, +0.0044] | +0.0016 [−0.0005, +0.0037] | +0.0028 [+0.0008, +0.0047] |
| MiniLM / Python docs | **−0.0112 [−0.0207, −0.0015]** | −0.0106 [−0.0184, −0.0029] | −0.0094 [−0.0159, −0.0030] |
| bge / Python docs | −0.0042 [−0.0128, +0.0046] | −0.0072 [−0.0148, +0.0006] | −0.0060 [−0.0118, −0.0000] |

On Python docs, refreshing makes retrieval slightly **worse**. The per-pool breakdown
gives the reason: refreshing helps queries about newly added text (hit@1 +0.046 to
+0.074) and hurts queries about retained text (−0.014 to −0.021 in 3 of 4 settings,
CIs excluding zero; MiniLM/arXiv +0.001, not significant). Retained text is 67% of
Python-docs query traffic and 19% of arXiv's. On MiniLM / Python docs the
unchanged-document pool also moves (hit@1 −0.011 [−0.022, −0.003]): those documents
are identical in both indexes, so this is competitor documents being refreshed around
them, not a change in the documents themselves. Our explanation is that added text
dilutes a whole-document embedding; this is consistent with the per-pool signs, but it
was not isolated by a dedicated experiment.

### 4.2 Answer level: the same questions, a large gap

Questions built from sentences whose number changed in the revision (§5).
"Answer@10" = gold document in the top 10 **and** the retrieved version contains a
correct value. Wilson CIs are in `results_answer_level.json`.

| model / corpus | questions | retrieval@10 stale / fresh (gap) | answer@10 stale / fresh (gap) |
|---|---|---|---|
| MiniLM / arXiv | 70 | 0.829 / 0.829 (0.000) | 0.043 / 0.829 (+0.786) |
| bge / arXiv | 70 | 0.800 / 0.800 (0.000) | 0.043 / 0.800 (+0.757) |
| MiniLM / Python docs | 103 | 0.748 / 0.718 (−0.029) | 0.204 / 0.718 (+0.515) |
| bge / Python docs | 103 | 0.728 / 0.738 (+0.010) | 0.175 / 0.738 (+0.563) |

Before the 2026-10-09 cleanup this table had 76 arXiv questions (cross-listed papers
counted twice) and 294 Python-docs questions, 187 of which came from the page footer
("3.8.20 Documentation", "© 2001-2025"). The stale copy could never hold the new
footer's version string, so those questions inflated the gap; removing them raised the
stale answer@10 on Python docs from 0.065–0.075 to 0.175–0.204. Six further Python-docs
questions were the same sentence repeated within one document and are now counted once.

**Control:** numbers in sentences identical in both versions (152 arXiv, 531 Python
docs questions). Stale and fresh agree; all four gaps lie within −0.013 to +0.007, at
retrieval and at answer level. In this control set answer@10 equals retrieval@10 by
construction (the value is in both versions), so it can only detect a retrieval bias
toward fresh documents; it does not test the containment metric itself.

**Question quality on Python docs.** Even after the footer cut, many Python-docs
questions are not facts a reader would ask about: 21 of the 103 are version notes
("Changed in version 3.12: ..."), and others come from code samples, ISBNs and RFC
numbers. arXiv is the cleaner evidence.

Answer@k is a **ceiling**: it says the correct value is available in the retrieved
text, not that a generator uses it. §4.3 tests the generator.

### 4.3 A real generator states the outdated value

Top-3 retrieved documents passed to Qwen2.5-1.5B-Instruct, asked to fill the blanked
number from the context only (`results_generation.json`). **These numbers predate the
2026-10-09 corpus cleanup** (arXiv duplicates, Python-docs footer) and must be
regenerated before they are cited: the Python-docs rows include footer questions. Shares are among answers
where the model committed to a number; Wilson 95% CIs.

| corpus | index | committed / asked | outdated value | correct value |
|---|---|---|---|---|
| arXiv | stale | 32 / 76 | **0.81** [0.65, 0.91] | 0.06 [0.02, 0.20] |
| arXiv | fresh | 42 / 76 | 0.00 [0.00, 0.08] | **0.95** [0.84, 0.99] |
| arXiv | control, stale / fresh | 38 / 40 of 120 | 0.00 / 0.00 | 0.92 / 0.93 (Fisher p = 1.0) |
| Python docs | stale | 22 / 294 | **0.32** [0.16, 0.53] | 0.23 [0.10, 0.43] |
| Python docs | fresh | 29 / 294 | 0.00 [0.00, 0.12] | **0.55** [0.38, 0.72] |
| Python docs | control, stale / fresh | 36 / 38 of 120 | 0.00 / 0.00 | 0.75 / 0.71 (Fisher p = 0.80) |

Stale vs fresh, Fisher exact: arXiv p = 1.4×10⁻¹⁴ (outdated), 4.8×10⁻¹⁶ (correct);
Python docs p = 1.5×10⁻³ (outdated), 0.025 (correct).

Read with care:
- The model abstains often: on 90–92% of Python-docs changed-fact questions, so that row
  describes 22 and 29 answers. arXiv is the stronger evidence.
- Gold-document retrieval in these prompts is the same for stale and fresh (arXiv 0.776
  both; Python docs 0.483 / 0.476), so the difference in answers is not a retrieval
  difference.
- The generator and prompt were accepted on **control questions only**, before any
  changed-fact question was generated (a 0.5B model was rejected at that stage).

### 4.4 How common fact-changing revisions are

Share of revised documents (1,568 arXiv, 1,346 Python docs) with at least one numeric
fact edit, numbers compared by value (`facts.py`):

| definition | arXiv | Python docs |
|---|---|---|
| strict: only a number changed in the sentence | 1.5% | 2.2% |
| aligned: closest old sentence (shingle Jaccard ≥ 0.5) has different numbers | 8.9% | 13.6% |
| the document's set of number values differs at all | 29.0% | 43.8% |

The paper uses *aligned*. Only numeric facts are counted, so this is a **lower bound**
on fact-bearing revisions.

---

## 5. Definitions the results depend on

- **Fact question:** a revised sentence whose closest old counterpart (Jaccard ≥ 0.5 on
  character 5-gram shingles) carries a different set of numbers, with all numbers removed so the
  question cannot contain its answer. The answer is the set of numbers, of at least 3
  characters, new in the revision. A sentence repeated within one document gives one
  question. Short integers are excluded because they occur in
  old documents by coincidence.
- **Numbers are compared by value:** `84.3%` = `84.3`, `20,000` = `20000`,
  `450K` = `450,000`, `1.20` = `1.2`, "1.5 million" = `1.5M`. Version strings such as
  `3.8.0` are kept verbatim.
- **Query pools:** two-sentence windows with a known gold document, drawn from `new`,
  `retained` or `unchanged` text.
- **Statistics:** bootstrap over queries (not over seeds), Wilson intervals for
  proportions, Fisher exact for stale vs fresh counts.

---

## 6. What is claimed, and what is not

**Claimed:**
1. On questions whose answer changed, retrieval metrics barely distinguish a stale
   index from a fresh one, while answer availability and generated answers differ
   sharply. Replicated on two embedding models and two corpora, with passing controls.
2. Therefore retrieval-level metrics are an invalid target for evaluating or detecting
   corpus staleness; it has to be measured at the fact or answer level.
3. This explains why the label-free retrieval-drift detectors in this project failed
   (§2.2).

**Not claimed:**
- That stale RAG gives wrong answers is not new on its own. Temporal QA benchmarks
  already show it. The contribution is the **paired measurement**: the same questions,
  the same retrieval success, a different answer. That makes the blind spot of
  retrieval metrics explicit and quantified.
- No new detector is proposed. The content-hash version registry (formerly in
  `drift/freshness.py`, now in `quarantine/`) is engineering. Detecting that indexed text changed is exact by construction, which is a
  property of hashing, not a result. Against independent source metadata, the registry
  found 56.8% / 79.0% / 89.2% of edits (arXiv / Python docs / Wikipedia) at 100%
  precision. On arXiv the misses are revisions that left the abstract untouched (43% of
  revisions), so there was nothing in the index to detect. This was a one-off check and
  is not packaged as a results file.
- `drift/topics.py` (query topic-mixture reporting, now in `quarantine/`) is
  unevaluated as research and is not part of the paper.
- The dilution explanation (§4.1) is consistent with the data but not isolated.

---

## 7. Limitations (state all of these in the paper)

- **Numeric facts only.** Changed dates written as words, entities, negations and
  qualitative claims are not covered.
- **Two corpora**, both with abstract- or section-length documents; no chunking in the
  main results.
- **Synthetic questions:** sentence windows with numbers removed, not real user queries.
- **Small sample on arXiv** (70 changed-fact questions), large CIs on the generator rows.
- **Python-docs questions are noisy** (§4.2): version notes, code samples and reference
  numbers make up much of the set.
- **One small generator** (1.5B, local) that abstains often. A larger model might
  abstain less, and might also resist outdated context differently.
- **Extreme conditions:** fully stale vs fully fresh. Partially refreshed or duplicated
  indexes (both versions present) were not evaluated at answer level.
- **Framework defaults** (§2.3) were read from source and not version-pinned.

---

## 8. Publishability: honest assessment

**Suitable for:** a workshop paper, a short paper, or an evaluation / reproducibility
track at an IR or NLP venue. The result is clean, replicated, controlled and
reproducible, and it gives practitioners a concrete recommendation.

**Not yet suitable for** a main-track full paper or a strong journal without more work:
the sample sizes are modest, there is one generator, one fact type and synthetic
queries.

**What would make it stronger, in order of value:**
1. A related-work check, done properly before writing. Position against temporal QA
   benchmarks (e.g. StreamingQA, RealTimeQA, FreshQA), knowledge-conflict studies
   (context vs parametric memory) and RAG evaluation frameworks. We have **not**
   completed this, so the novelty claim in §6 is provisional.
2. A second, larger generator (7B+ or an API model) on the same prompts.
3. Answer-level evaluation of a **mixed** index containing both versions, the realistic
   failure under the default framework behaviour in §2.3.
4. Non-numeric fact edits (entities, dates), to raise the 70-question arXiv set.
5. A third corpus (Wikipedia revisions are already available).

**Authorship:** the motivating negative results (`FINDINGS.md` §3–§13) were produced by
Suvidh Jain, building on Aryan Garg's base pipeline. Authorship and credit must be
agreed with them before submission.

---

## 9. Reproduce and number sources

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/.cache/minilm-l6-v2
huggingface-cli download BAAI/bge-small-en-v1.5             --local-dir ~/.cache/bge-small-en-v1.5

python -m research.staleness.run_headline     # §4.1  -> results_headline.json
python -m research.staleness.answer_level     # §4.2  -> results_answer_level.json
python -m research.staleness.generation build # §4.3  prompts
<venv with torch, transformers> python research/staleness/generate_hf.py \
    ~/.cache/qwen2.5-1.5b-instruct            # §4.3  outputs (greedy, resumable)
python -m research.staleness.generation score # §4.3  -> results_generation.json
python -m pytest tests/test_staleness.py
```

Corpora are cleaned at load time (`data.load_corpus`); the JSON files are the raw
collector output. Embeddings are cached under a digest of the exact texts, so changing a
corpus or a question definition recomputes them.

| Section | Source |
|---|---|
| §2.1 | `FINDINGS.md` §12.5, §12.6, §12.8, §12.9 with the §13.3 correction |
| §4.1 | `research/staleness/results_headline.json` |
| §4.2 | `research/staleness/results_answer_level.json` |
| §4.3 | `research/staleness/results_generation.json` |
| §4.4 | `research/staleness/facts.py` (computed, not stored in a results file) |
| §6 edit recall | one-off check, not stored, so re-run before citing |

**Validation already done:** both numpy encoders match HuggingFace `transformers`
(tokenisation identical on 30/30 texts, embeddings within 2×10⁻⁷, minimum cosine
0.99999988). The query pools regenerate every cached array exactly. Three scoring bugs
found during the work (spelling instead of value, dropped magnitude suffixes,
formatting changes counted as facts) are fixed and covered by tests; fixing them moved
the headline numbers by at most 0.004. The §4.4 prevalence figures were recomputed with
the fixed code for this document (earlier drafts quoted 9.7% / 32.8% / 56.7%, from
before the value comparison was fixed).
