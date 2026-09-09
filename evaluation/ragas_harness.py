"""OPTIONAL: RAGAS faithfulness as a secondary signal.

Not the primary endpoint.  Faithfulness scores whether an answer is entailed by the
context it was given, so a stale-but-plausible context scores *well* -- the model
faithfully summarises an outdated document.  It is blind to the failure mode this
project measures (FINDINGS.md F17).  ``evaluation/retrieval_quality.py`` is the
endpoint that moves when the index goes stale.

Repairs, for anyone who does want it
------------------------------------
F10 The harness built ``question`` / ``answer`` / ``contexts``.  RAGAS renamed these to
    ``user_input`` / ``response`` / ``retrieved_contexts`` in 0.2, and the lockfile pins
    0.3.1, so every call was malformed.  It also did ``float(result["faithfulness"])``,
    assuming a scalar where >= 0.2 returns a list.

F23 The resulting exception was caught and converted to ``None``, so faithfulness was
    empty for every window, both rho and lead-time were skipped, and the run still
    reported success.  Failures now propagate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def _require_ragas():
    try:
        import ragas
        from ragas import evaluate
        from ragas.metrics import faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS is optional and not installed:\n"
            "  uv pip install -e .[heavy]\n"
            "The default endpoint (retrieval recall@k) needs none of it."
        ) from exc
    return ragas, evaluate, faithfulness


def run_ragas_harness(
    per_window_results: dict[int, list[dict]],
    results_dir: Path,
    max_per_window: int = 10,
) -> pd.DataFrame:
    """Score faithfulness per window. Requires answers, so generation must be enabled."""
    if not per_window_results:
        raise ValueError(
            "no RAG results to score. Run `python main.py run-rag` with OPENAI_ENABLED=true first."
        )

    _, evaluate, faithfulness = _require_ragas()
    from datasets import Dataset

    rows: list[dict] = []
    for window, items in sorted(per_window_results.items()):
        usable = [
            it for it in items
            if it.get("answer") and it.get("contexts") and not it.get("error")
        ][:max_per_window]
        if not usable:
            logger.warning(f"t={window}: no answered queries to score")
            continue

        # RAGAS >= 0.2 schema
        ds = Dataset.from_dict(
            {
                "user_input": [it["question"] for it in usable],
                "response": [it["answer"] for it in usable],
                "retrieved_contexts": [list(it["contexts"]) for it in usable],
            }
        )
        result = evaluate(ds, metrics=[faithfulness])

        # >= 0.2 returns one score per sample, not a scalar
        scores = result["faithfulness"]
        series = pd.Series(scores if hasattr(scores, "__len__") else [scores], dtype="float64")

        rows.append(
            {
                "window": window,
                "faithfulness_score": float(series.mean()),
                "faithfulness_sd": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "n_scored": int(series.notna().sum()),
            }
        )
        logger.info(f"t={window}: faithfulness {rows[-1]['faithfulness_score']:.4f} (n={rows[-1]['n_scored']})")

    df = pd.DataFrame(rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_dir / "faithfulness.parquet", index=False)
    return df
