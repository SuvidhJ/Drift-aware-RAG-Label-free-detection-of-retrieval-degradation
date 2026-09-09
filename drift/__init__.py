"""
drift - drift detection
=======================

The mathematical core. Takes per-window embedding arrays and produces D(t), its
decomposition into query and corpus components, and a regime label.

``wasserstein.py`` - sliced Wasserstein-1
    Pure numpy; POT is not required. Reports two uncertainties *separately*, because
    conflating them was a real defect: ``ci_lo``/``ci_hi`` is a bootstrap over the
    data (sampling uncertainty, which does not shrink with more projections) and
    ``mc_se`` is the Monte-Carlo error of the slicing (which does). The previous
    version reported the second and called it the first, so the interval narrowed
    without bound while the estimand's uncertainty plateaued (F12).

``mmd.py`` - RBF-kernel MMD through the Gram identity
    ``||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>`` turns an (n, m, d) tensor into one
    matmul: 32 MB instead of 16.4 GB at the shipped defaults, agreeing to 4e-17 (F7).
    ``mmd_sq`` is reported unclipped, because the unbiased estimator is legitimately
    negative under the null and a permutation calibration needs that left tail.

``null.py`` - **the piece everything else keys off**
    Every two-sample statistic here carries an n^(-1/2) level term even when both
    windows come from the same distribution, so a fixed threshold is partly a
    measurement of traffic volume (F13) and does not survive a change in window size
    (12.8). Instead each comparison is calibrated against a permutation null built at
    its own window sizes, and what gets reported is a z-score and a p-value, which are
    comparable across windows. This is calibration, not a cure: 12.9 shows power at a
    correctly calibrated 5% false-alarm rate is 0.000-0.100 at realistic effect sizes.

``statistic.py`` - the D(t) compositor
    ``D(t) = w_q * z_q + w_c * z_c``. The old third term, the query-to-corpus centroid
    gap, was the distance between two text *genres* - fixed at t=0, not a change over
    time - and it accounted for 87-99% of D(t), so the alarm rule was decided before
    any drift occurred (F5). It is still computed and reported as ``qc_genre_gap``,
    but it no longer enters the alarm path.

``decomposition.py`` - the regime plane
    Classifies on the two absolute z-scores. The previous version normalised them into
    shares summing to 1, which collapsed a 2x2 taxonomy to a 1-D partition, gave
    (0.9, 0.9) and (0.009, 0.009) the same label, and left ``STABLE`` unreachable on
    its final branch (F11).
"""
