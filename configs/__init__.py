"""
configs - configuration
=======================

Single source of truth for every tunable parameter. Built on ``pydantic-settings``:
each section is a ``BaseSettings`` with its own environment prefix (``DATA_``,
``EMBED_``, ``DRIFT_``, ``POLICY_``, ``RETRIEVAL_``, ``OPENAI_``, ``LOG_``), and a
``.env`` file is read if present.

The singleton is importable directly::

    from configs.settings import settings

Choices that are configuration rather than code:

- ``EMBED_BACKEND`` selects the encoder. ``minilm-numpy`` (default) needs no torch;
  ``sentence-transformers`` needs the ``heavy`` extra. Replicating a result on a
  second embedding model is a one-line change.
- ``RETRIEVAL_BACKEND`` selects ``numpy`` (exact) or ``chroma``.
- ``DATA_SOURCE`` selects the real revision corpora or the legacy Amazon path.
- ``OPENAI_ENABLED`` turns answer generation on; it is off by default, because the
  endpoint that matters here is retrieval quality.

Two constraints that used to live in this schema are gone, deliberately:

- ``DRIFT_W1_N_PROJECTIONS >= 500`` was enforced as a threat mitigation, on the theory
  that more projections meant a tighter confidence interval. They tighten the wrong
  thing: projections reduce Monte-Carlo error of the slicing, not sampling
  uncertainty, and the interval that shrank with them was measuring the wrong quantity
  (FINDINGS.md F12). The two are now reported separately and the floor is 32.
- ``DRIFT_THRESHOLD_TAU`` no longer drives any alarm. It survives only so
  ``evaluation/transfer.py`` can demonstrate that a fixed threshold does not survive a
  change in window size. Alarms key off ``DRIFT_TAU_Z_*``, which are z-scores against
  a per-window permutation null.

``DATA_MIN_WINDOW_SAMPLES`` is now actually read, by ``data/windows.py``. It was
declared as a threat mitigation and referenced by nothing (F22).
"""
