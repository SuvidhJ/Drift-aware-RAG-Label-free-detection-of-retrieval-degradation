# `research/staleness/` — what corpus staleness actually costs

Everything else in this repository asked whether staleness can be **detected**. This
directory asks what it **costs**, and finds that the answer depends entirely on where
you measure it.

## The result

On questions whose answer was changed by a revision, a fully stale index retrieves the
right document **as often as** a fully refreshed one, while carrying the correct answer
far less often:

| model / corpus | questions | retrieval@10 stale / fresh | answer@10 stale / fresh |
|---|---|---|---|
| MiniLM / arXiv, 6,835 docs | 76 | 0.803 / 0.803 (**0.000**) | 0.039 / 0.803 (**+0.763**) |
| bge / arXiv, 6,835 docs | 76 | 0.776 / 0.776 (**0.000**) | 0.039 / 0.776 (**+0.737**) |
| MiniLM / Python docs 3.8→3.13 | 294 | 0.626 / 0.609 (**−0.017**) | 0.075 / 0.609 (**+0.534**) |
| bge / Python docs 3.8→3.13 | 294 | 0.707 / 0.697 (**−0.010**) | 0.065 / 0.697 (**+0.632**) |
| control: facts that did *not* change (all four) | 169–543 | gaps −0.006 to +0.006 | identical to retrieval |

MiniLM-L6-v2 (mean pooling) and bge-small-en-v1.5 (CLS pooling), both validated against
the HuggingFace reference. Full numbers in `results_answer_level.json`.

### A real generator states the outdated fact

Top-3 retrieved documents → Qwen2.5-1.5B-Instruct, asked to fill the blanked number from
context only (`generation.py`, `results_generation.json`). Shares are of answers where the
model committed to a number; it abstains on the rest.

| corpus | index | committed | states the **outdated** number | correct |
|---|---|---|---|---|
| arXiv | stale | 32 | **0.81** [0.65, 0.91] | 0.06 [0.02, 0.20] |
| arXiv | fresh | 42 | 0.00 [0.00, 0.08] | **0.95** [0.84, 0.99] |
| arXiv | control, stale / fresh | 38 / 40 | 0.00 / 0.00 | 0.92 / 0.93 (p = 1.0) |
| Python docs | stale | 22 | **0.32** [0.16, 0.53] | 0.23 [0.10, 0.43] |
| Python docs | fresh | 29 | 0.00 [0.00, 0.12] | **0.55** [0.38, 0.72] |
| Python docs | control, stale / fresh | 36 / 38 | 0.00 / 0.00 | 0.75 / 0.71 (p = 0.80) |

Stale vs fresh, Fisher exact: arXiv p = 1.4×10⁻¹⁴ (outdated), 4.8×10⁻¹⁶ (correct);
Python docs p = 1.5×10⁻³ and 0.025. The model abstains on 90–92% of Python-docs
questions, so that row describes the minority it chose to answer. The generator was
judged adequate on control questions alone (94% precision when committing), and the
prompt was fixed before any changed-fact question was generated.

### Retrieval level alone

At retrieval level alone (`results_headline.json`), a fully stale index costs between
−1.2 and +0.5 points of hit@1 across both models and both corpora; on Python docs a full
refresh makes retrieval significantly *worse* (bge hit@1 −0.0120 [−0.0211, −0.0032]),
because refreshing dilutes each document's embedding with added text and costs queries
about the text it retained.

A stale index succeeds at retrieval and fails at the answer. Retrieval-level metrics
therefore cannot see the harm staleness does — which is why every label-free detector
in this repository, validated against retrieval degradation, was validated against the
wrong target. Measured at retrieval level, a fully stale index costs under one recall
point (`results_headline.json`); that is the symptom, not the finding.

## Definitions (all text-only — nothing here depends on embeddings)

**Query pools** (`queries.py`). Two-sentence windows with a known gold document:
`new` (text only in the revision), `retained` (text in both versions of a revised
document), `unchanged` (documents never revised). A realistic query stream mixes all
three in the corpus's own proportions (`measure.pool_weights`).

**Fact edits** (`facts.py`). A revised sentence whose closest old counterpart (shingle
Jaccard ≥ 0.5) carries a different set of numbers. A *fact question* is that sentence
with its numbers removed, so it cannot contain its own answer; the answer is the set of
distinctive (≥ 3 character) numbers new in the revision. Short integers are excluded
because they appear in old documents by coincidence. Numbers are compared by **value**
(`84.3%` = `84.3`, `20,000` = `20000`): comparing spellings counted dropped percent
signs as changed facts, which an earlier version of this module did.

**Answer-level** (`answer_level.py`). The gold document is in the top-k *and* the
retrieved version contains a correct answer. This is a ceiling on answer correctness —
a faithful generator cannot state a number absent from its context — measured by string
containment, with no LLM.

## Reproduce

```bash
# models: both run through the repo's own numpy BERT forward pass
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/.cache/minilm-l6-v2
huggingface-cli download BAAI/bge-small-en-v1.5             --local-dir ~/.cache/bge-small-en-v1.5

python -m research.staleness.run_headline    # retrieval-level cost of total staleness
python -m research.staleness.answer_level    # retrieval-level vs answer-level, + control
python -m pytest tests/test_staleness.py
```

Embeddings are cached under `embeddings/cache/staleness/<model>/` (gitignored), keyed by
model, so two models can never share a file.

## Validation

- Both numpy encoders match the HuggingFace `transformers` reference: tokenisation
  identical on 30/30 texts, embeddings within 2×10⁻⁷, minimum cosine 0.99999988.
- The query pools regenerate the arrays behind every earlier number exactly
  (identical gold indices on all eleven pools).
- `tests/test_staleness.py` pins the definitions above.

## Limits

- Two corpora with retrieval headroom, abstract- and section-length documents, and
  sentence-window queries rather than real user questions.
- Answer-level is an availability ceiling; the generator confirmation uses one small
  (1.5B) local model that abstains often, especially on Python docs.
- Fact edits are numeric; entity, date and negation changes are not yet covered, so the
  measured prevalence (8.7% of arXiv revisions, 24.7% of Python-docs revisions) is a
  lower bound on fact-bearing edits.
