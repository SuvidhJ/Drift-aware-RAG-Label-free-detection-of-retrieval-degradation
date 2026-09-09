"""Action space and cost table.

Costs are compute multiples relative to serving a window with no intervention:

    a1 no-op      x1     do nothing
    a2 re-rank    x4     MMR re-rank a wider candidate set at query time
    a3 re-index   x22    re-embed a budgeted subset of the corpus
    a4 fine-tune  x110   retrain the encoder
    a5 escalate   x1000  hand the window to a human

a5 was previously 1e9, which is not a cost but a sentinel; at that magnitude it
dominates every float comparison in the objective and makes the argmin numerically
meaningless. 1000 keeps it firmly last-resort while remaining a number.
"""
from __future__ import annotations

from enum import Enum

from configs.settings import settings


class Action(str, Enum):
    NO_OP = "a1_no_op"
    RERANK = "a2_rerank"
    REINDEX = "a3_reindex"
    FINETUNE = "a4_finetune"
    ESCALATE = "a5_escalate"


def action_cost(action: Action, budget: float | None = None) -> float:
    """Compute cost of an action.

    ``a3_reindex`` scales with the fraction of the corpus actually re-embedded, since
    that is where its cost comes from. The others are flat.
    """
    cfg = settings.policy
    if action is Action.REINDEX:
        budget = cfg.reindex_budget if budget is None else budget
        return cfg.cost_no_op + (cfg.cost_reindex - cfg.cost_no_op) * float(budget)
    return {
        Action.NO_OP: cfg.cost_no_op,
        Action.RERANK: cfg.cost_rerank,
        Action.FINETUNE: cfg.cost_finetune,
        Action.ESCALATE: cfg.cost_escalate,
    }[action]


REGIME_TO_ACTION: dict[str, Action] = {a.value: a for a in Action}
