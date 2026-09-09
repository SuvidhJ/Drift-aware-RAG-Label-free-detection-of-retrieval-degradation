"""
Cost-aware remediation policy -- evaluated, not just tabulated.

The repo has a cost table (1:4:22:110:inf) and an argmin solver with hand-coded
recovery coefficients (0.4/0.8/0.9) that appear nowhere in settings and are never
evaluated against an alternative.  Here we:

  1. give the world an explicit, if stylised, harm model tied to the injected
     drift magnitudes (so "regret" is a realised quantity, not a proxy);
  2. run several policies over each factorial run;
  3. report realised total cost and regret-vs-oracle.

Harm model (per window, from t_event on):
    harm_query_t  = KQ * r           (users asking about unindexed line)
    harm_corpus_t = KC * f           (stale context served)
An action neutralises the harm channel(s) it addresses, from the window it is
taken onward, once:
    a2_rerank   -> removes  eta_r * harm_query
    a3_reindex  -> removes  eta_i * harm_corpus
    a4_finetune -> removes  harm_query + harm_corpus
    a5_escalate -> removes everything, cost = huge
Per-window loss = lambda * residual_harm + action_cost_this_window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COST = {"a1_no_op": 1.0, "a2_rerank": 4.0, "a3_reindex": 22.0,
        "a4_finetune": 110.0, "a5_escalate": 4000.0}   # escalate: large but finite

# harm per post-event window = lambda * (KQ*r + KC*f_eff), with
#   f_eff ~ 0.02 * corpus_delist_lines  (unanswerable share from the corpus side)
KQ, KC = 250.0, 1500.0
F_PER_LINE = 0.02
ETA_R, ETA_I = 0.55, 0.85    # how well rerank / reindex actually fix their own channel


@dataclass
class PolicyOutcome:
    name: str
    total_cost: float
    action_cost: float
    harm_cost: float
    actions: list[str]


def _simulate(spec, actions: list[str], lam: float = 1.0) -> PolicyOutcome:
    T = spec.T
    r = spec.query_newline_frac
    f = F_PER_LINE * spec.corpus_delist_lines + spec.corpus_delist_frac
    q_fixed = 0.0        # cumulative fraction of query-harm channel removed
    c_fixed = 0.0
    action_cost = 0.0
    harm_cost = 0.0
    for t in range(T):
        a = actions[t]
        action_cost += COST[a]
        if a == "a2_rerank":
            q_fixed = max(q_fixed, ETA_R)
        elif a == "a3_reindex":
            c_fixed = max(c_fixed, ETA_I)
        elif a in ("a4_finetune", "a5_escalate"):
            q_fixed = c_fixed = 1.0
        if t >= spec.t_event:
            harm_cost += lam * (KQ * r * (1.0 - q_fixed) + KC * f * (1.0 - c_fixed))
    return PolicyOutcome("", action_cost + harm_cost, action_cost, harm_cost, list(actions))


# ------- policies -------------------------------------------------------- #
def policy_constant(spec, action: str) -> list[str]:
    return [action] * spec.T


def policy_oracle(spec, lam: float = 1.0) -> list[str]:
    """True argmin over the realistic policy class 'take at most one action, at the
    event window' -- enumerates all 5 single actions + pure no-op and picks the
    cheapest realised total.  Guarantees regret_vs_oracle >= 0 for every other
    policy in that class."""
    cands = [["a1_no_op"] * spec.T]
    for a in ("a2_rerank", "a3_reindex", "a4_finetune", "a5_escalate"):
        acts = ["a1_no_op"] * spec.T
        acts[spec.t_event] = a
        cands.append(acts)
    best = min(cands, key=lambda acts: _simulate(spec, acts, lam).total_cost)
    return best


def policy_threshold_only(spec, scores: np.ndarray, tau_z: float) -> list[str]:
    """Magnitude only: reindex whenever the (calibrated) score crosses tau. No regime."""
    acts = ["a1_no_op"] * spec.T
    done = False
    for i, s in enumerate(scores):          # scores index window t = i+1
        t = i + 1
        if s > tau_z and not done:
            acts[t] = "a3_reindex"
            done = True
    return acts


def policy_from_regime(spec, regimes: list[str], sig: list[bool]) -> list[str]:
    """Take the matching action at the first window whose regime fires; then hold."""
    acts = ["a1_no_op"] * spec.T
    fire = {"query_drift": "a2_rerank", "corpus_drift": "a3_reindex",
            "joint_drift": "a4_finetune", "stable": "a1_no_op"}
    done = False
    for i, (rg, sg) in enumerate(zip(regimes, sig)):
        t = i + 1
        if sg and rg != "stable" and not done:
            acts[t] = fire[rg]
            done = True
    return acts


def evaluate_policies(spec, cal_regimes, cal_sig, cal_scores, orig_regimes,
                      tau_z: float, lam: float = 1.0) -> list[PolicyOutcome]:
    runs = {
        "oracle": policy_oracle(spec, lam),
        "always_no_op": policy_constant(spec, "a1_no_op"),
        "always_rerank": policy_constant(spec, "a2_rerank"),
        "always_reindex": policy_constant(spec, "a3_reindex"),
        "threshold_only": policy_threshold_only(spec, cal_scores, tau_z),
        "repo_regime": policy_from_regime(spec, orig_regimes, [r != "stable" for r in orig_regimes]),
        "calibrated_regime": policy_from_regime(spec, cal_regimes, cal_sig),
    }
    outs = []
    for name, acts in runs.items():
        o = _simulate(spec, acts, lam)
        o.name = name
        outs.append(o)
    return outs
