"""
streaming - online drift detection
==================================

Adapts the batch D(t) computation to a live setting where windows arrive one at a
time, rather than being replayed from stored snapshots.

``online_detector.py`` - two-sided CUSUM on the standardised statistic

The previous detector could not fire at all, for three compounding reasons (F14):

  1. the reference mean was recomputed *after* appending the current observation, so
     the baseline chased the very signal it was meant to detect;
  2. the slack ``k = 0.5`` was absolute, applied to a statistic living on the 0.1
     scale, so ``g_pos = max(0, g_pos - ~0.5)`` was pinned at zero and the alarm
     degenerated to the plain threshold test ``d > tau``;
  3. the 10-window rolling reference absorbed gradual drift by construction - the
     exact failure mode the detector exists to catch.

``g_neg`` was computed and never read, and ``h`` was an arbitrary constant with no
operating characteristic attached.

Now: the reference is frozen on a burn-in prefix, observations are standardised, slack
is expressed in sigma units, both arms are live, and ``calibrate_h(target_arl0)``
chooses the decision interval for a target in-control average run length by
simulation. A burn-in with no variability is reported rather than silently papered
over - falling back to sd = 1.0 there used to blind the detector completely.
"""
