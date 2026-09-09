"""
policy - cost-aware action selection
====================================

Turns a drift signal into a remediation action by expected cost, rather than raising
an alarm for a human to interpret.

``costs.py``
    The action space and its compute multiples: no-op x1, re-rank x4, re-index x22,
    fine-tune x110, escalate x1000. ``a3_reindex`` scales with the fraction of the
    corpus actually re-embedded, since that is where its cost comes from. Escalation
    was previously 1e9, which is a sentinel rather than a cost - at that magnitude it
    dominates every float comparison and makes the argmin numerically meaningless.

``engine.py``
    ``argmin_a [ cost(a) + lambda * regret_scale * residual_loss(a) ]``.

    The previous engine computed no-op regret as ``d * (alpha + beta)`` where
    ``alpha + beta == 1`` by construction, so it reduced to ``d`` - then compared that
    faithfulness fraction in [0, 1] against compute multiples in [1, 110] at
    lambda = 1. The two quantities shared no unit, and over its entire reachable input
    range the engine selected ``a1_no_op`` and nothing else; reaching ``a2_rerank``
    needed lambda > 54 or D > 7.5 against an achievable maximum D of 0.139. The repo's
    own policy.csv showed the "repo regime" row bit-identical to "always no-op" at
    every lambda (F6).

    ``regret_scale`` now converts lost retrieval quality into compute-equivalent
    units, so the comparison is dimensionally coherent and lambda is a dimensionless
    preference rather than a unit conversion smuggled in as a hyperparameter.

    ``sweep_lambda()`` is the degeneracy check the old engine would have failed, and
    ``tests/test_policy.py`` asserts on it. ``fit_loss_model()`` regresses observed
    recall loss on the drift z-scores: until it has been run and its R^2 inspected,
    any cost saving quoted from this engine is a model output, not a measurement.
"""
