"""
evaluation - validation harness
===============================

Computes the numbers the empirical claim rests on, from actual pipeline output. Where
the data cannot support a statistic, these functions say so explicitly rather than
returning a plausible-looking value.

``retrieval_quality.py`` - **the primary endpoint**
    recall@k and MRR against a known gold document. This replaced RAGAS faithfulness
    as the endpoint because faithfulness scores answer-to-context consistency, so a
    stale-but-plausible context scores *well* - the model faithfully summarises the
    outdated document it was handed. It was blind to the failure mode this project
    measures (FINDINGS.md F17).

``correlate.py`` - Spearman rho, lead time, AUROC
    rho is reported with a moving-block bootstrap CI, because a drift series is
    autocorrelated and resampling individual windows would assume an independence it
    does not have (F20). Lead time keys off the null-calibrated alarm rather than a
    fixed threshold. AUROC takes its labels from the caller with the convention
    recorded, and ``auroc_from_churn`` derives a balanced label from per-window churn -
    the previous label produced exactly one positive window, and the code and its
    docstring disagreed about what it meant (F19).

``null_corpus_test.py`` - false-alarm rate
    Counts calibrated alarms over a slice where nothing drifts, with a Wilson interval
    and an explicit note of the granularity (1/T). The previous pipeline reported
    FAR = 3.1% and could not have computed it: nothing in the repository wrote the null
    series the check guarded on, and at T = 12 the achievable rates near 5% are 0% and
    8.3% anyway (F2, F21).

``transfer.py`` - threshold transfer under a shift in traffic volume
    Reproduces FINDINGS.md 12.8 as a command. Both windows come from the same
    distribution, so the correct false-alarm rate is alpha everywhere; a fixed
    threshold moved across window sizes departs from it in both directions.

``power.py`` - detection power swept through the false-alarm floor
    Reproduces FINDINGS.md 12.9. Reports where the corpus actually sits, so the
    operating point can be read off the sweep rather than assumed.

``ragas_harness.py`` - OPTIONAL secondary signal
    Needs the ``heavy`` extra and generated answers. Kept, with its schema corrected to
    RAGAS >= 0.2 (``user_input`` / ``response`` / ``retrieved_contexts``) and its
    exception no longer swallowed (F10, F23).
"""
