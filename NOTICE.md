# NOTICE

## Provenance

This repository has three layers of authorship:

| Layer | Author |
|---|---|
| The base pipeline — the original drift-detection, retrieval and policy code | Aryan Garg |
| The first analysis pass and the repository this work was forked from | Aditi Saxena |
| The audit, the validation programme under `research/`, and the rebuild recorded in `FINDINGS.md` §13 | Suvidh Jain |

Individual modules that carry their own authorship notice keep it. `FINDINGS.md` §8
records which revision of the pipeline each audit finding was written against.

## Licence status

The `LICENSE` file applies the MIT License to this repository. Contributors who have
not yet confirmed that grant retain their own copyright in the portions they wrote; if
you intend to redistribute or build on this code commercially, confirm the position
with all three authors named above first.

## Third-party data

`research/gates/results/` ships derived text corpora used to reproduce the results.
They are redistributed here for reproducibility and remain under their original terms:

| File | Source | Terms |
|---|---|---|
| `wp_corpus.json` | Wikipedia lead sections, 2023 vs 2026 | CC BY-SA 4.0 — attribution and share-alike apply |
| `pd_corpus.json` | Python documentation, 3.8 vs 3.13 | PSF License Agreement |
| `ax_corpus.json`, `ax_corpus_big_manifest.json` | arXiv abstracts, v1 vs latest | per-article terms; most are arXiv's non-exclusive licence to distribute |

The collectors that produced them are `research/gates/e6_*fetch*.py`, so every file can
be regenerated from its source rather than taken from this repository.

## Models

The default embedding path uses `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0).
The checkpoint is not vendored here; `README.md` gives the one-line download.
