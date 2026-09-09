"""
data - the data foundation
==========================

``corpora.py`` - **the default source**
    Real revision corpora: documents observed at two real timepoints, with real edit
    timestamps and revision ids. Nothing is simulated - every pair is a real document
    and a real later revision of that same document. arXiv (v1 vs latest), Python docs
    (3.8 vs 3.13) and Wikipedia lead sections (2023 vs 2026).

    Python docs stores version labels rather than timestamps; that is reported as
    ``real_timestamps=False`` and surfaced in the manifest, instead of being coerced
    into a fake date.

``windows.py`` - two views of the corpus, deliberately
    ``corpus_t*.parquet`` holds every document at its live revision as of t. This is
    the paired view an index has, and it is what retrieval and selective re-index use.

    ``sample_t*.parquet`` holds a *disjoint* subset per window. The drift detector
    needs this one, because the permutation null assumes the two windows are
    independent samples. With overlapping windows the observed split pairs each
    document with itself and sits below the entire null distribution - measured, the
    corpus z-score was pinned near -5 in every window, including windows with real
    revisions.

    Queries are sentence windows from live document text, so each has a known gold
    document and the stream drifts both because content is revised and because the
    topic mixture moves. The previous pipeline emitted one constant string for every
    row, so query-side drift was identically zero by construction (F3).

    ``build_null_windows`` writes a slice where nothing drifts, of arbitrary length.
    Its queries come from one fixed pool: drawing them per window group made
    successive query windows samples of different content, which is a real difference
    rather than a null, and it alarmed in 5 of 5 windows.

**Legacy Amazon path** (``DATA_SOURCE=amazon``, needs the ``heavy`` extra):
``download.py``, ``build_snapshots.py``, ``inject_drift.py``. Repaired but not
recommended - the dataset ships no question-answering split, so it has no query
stream at all (F3, F4, F9, F24), and product liveness cannot be reconstructed from
two truncated dumps, which made snapshots near-identical across windows.
"""
