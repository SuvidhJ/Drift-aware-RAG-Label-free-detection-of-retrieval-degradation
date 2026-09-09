"""
tests — Test suite
===================

**What it's for:**
Unit tests for the three mathematical components that must be correct
for the paper's claims to hold. Tests use synthetic data with known
properties so correctness is verifiable without running the full pipeline.

**Test files:**

``test_wasserstein.py`` — Sliced W1 sanity checks
    Tests the ``sliced_w1`` estimator against known-distribution properties:
    - W1(P, P) ≈ 0 for identical distributions
    - W1 is large when distributions are separated (mean shift of 5 units)
    - W1 is monotone: larger shift → larger W1
    - CI bounds contain the point estimate
    - Raises ``ValueError`` if n_projections < 500 (T4 enforcement)
    Runtime: ~3 minutes (1000 bootstrap iterations × 500 projections each)

``test_mmd.py`` — MMD sanity checks
    Tests the ``mmd_rbf`` estimator:
    - MMD(P, P) ≈ 0 for identical distributions
    - MMD is large for well-separated distributions
    - MMD is monotone in distribution shift magnitude
    - sigma is correctly reported and respected when specified manually

``test_decomposition.py`` — Regime classifier correctness
    Tests all four quadrants using synthetic (w1_q, w1_c) pairs:
    - Stable: D < τ → always stable regardless of components
    - Query drift: high w1_q, low w1_c → α̂ > τ_α
    - Corpus drift: low w1_q, high w1_c → β̂ > τ_β
    - Joint drift: equal high w1_q + w1_c → both α̂ > τ_α and β̂ > τ_β
    - Persistence gate: 1-window spike returns None (not gated)
    - Persistence gate: 3+ windows fire the action

``test_policy.py`` — Policy engine cost table + solver
    Tests:
    - Exact cost ratios: rerank/no_op = 4.0, reindex/no_op = 22.0, etc.
    - Zero drift → no-op selected (no regret to trade away)
    - High λ (1000) + high drift → no-op loses to cheaper action
    - Escalation never selected for moderate drift (compliance cost = 1e9)
    - ``total_cost`` field is populated for all windows
"""
