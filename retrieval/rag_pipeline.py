"""Retrieve, optionally re-rank, optionally generate.

Two changes from the previous version.

Generation is optional and off by default.  The measurement this project needs is
whether the index still returns the right document; an LLM in the loop adds cost and
variance without adding signal, and the endpoint it enabled (RAGAS faithfulness) was
blind to staleness anyway (FINDINGS.md F17).  Set ``OPENAI_ENABLED=true`` to turn it
on.

Failures are raised, not swallowed.  Three bare ``except Exception`` handlers on the
critical path used to convert hard failures into empty results while ``run-all``
printed success regardless (F23).  A generation error now propagates unless the
caller explicitly passes ``on_error="record"``, which records it in the result so a
batch run can continue and still report honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from configs.settings import settings
from retrieval.rerank import mmr_rerank


@dataclass
class RAGResult:
    question: str
    contexts: list[str]
    context_rows: list[int]
    action_applied: str
    answer: str | None = None
    error: str | None = None
    scores: list[float] = field(default_factory=list)


def retrieve_for_query(
    index,
    query_embedding: np.ndarray,
    action: str = "a1_no_op",
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and scores for one query, widening the candidate set when re-ranking."""
    cfg = settings.retrieval
    top_k = top_k or cfg.top_k
    fetch_k = cfg.mmr_fetch_k if action == "a2_rerank" else top_k

    res = index.search(query_embedding[None, :], top_k=fetch_k)
    rows, scores = res.indices[0], res.scores[0]

    if action == "a2_rerank" and len(rows) > top_k:
        order = mmr_rerank(query_embedding, index.embeddings[rows], top_k=top_k)
        rows, scores = rows[order], scores[order]

    return rows[:top_k], scores[:top_k]


def generate_answer(
    question: str,
    query_embedding: np.ndarray,
    index,
    action: str = "a1_no_op",
    top_k: int | None = None,
    on_error: Literal["raise", "record"] = "raise",
) -> RAGResult:
    """Full cycle for one query. Generation runs only when it is enabled."""
    rows, scores = retrieve_for_query(index, query_embedding, action=action, top_k=top_k)
    contexts = [index.texts[int(r)] for r in rows]

    result = RAGResult(
        question=question,
        contexts=contexts,
        context_rows=[int(r) for r in rows],
        action_applied=action,
        scores=[float(s) for s in scores],
    )

    cfg = settings.generation
    if not cfg.enabled:
        return result
    if not contexts:
        result.error = "no context retrieved"
        if on_error == "raise":
            raise RuntimeError(f"no context retrieved for question: {question[:80]!r}")
        return result

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "generation is enabled but the openai package is not installed:\n"
            "  uv pip install -e .[heavy]\n"
            "or set OPENAI_ENABLED=false to use the retrieval-only endpoint"
        ) from exc

    prompt = _build_prompt(question, contexts)
    try:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s)
        response = client.chat.completions.create(
            model=cfg.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        result.answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error(f"generation failed against {cfg.base_url}: {exc}")
        if on_error == "raise":
            raise
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _build_prompt(question: str, contexts: list[str]) -> str:
    block = "\n\n".join(f"[Doc {i + 1}]: {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the provided documents. "
        "If they do not contain the answer, say so.\n\n"
        f"Documents:\n{block}\n\nQuestion: {question}\n\nAnswer:"
    )


def run_window(
    index,
    query_embs: np.ndarray,
    questions: list[str],
    action: str = "a1_no_op",
    max_questions: int | None = None,
    on_error: Literal["raise", "record"] = "raise",
) -> list[RAGResult]:
    """Run a whole window of queries."""
    n = len(questions) if max_questions is None else min(max_questions, len(questions))
    out: list[RAGResult] = []
    for i in range(n):
        out.append(
            generate_answer(questions[i], query_embs[i], index, action=action, on_error=on_error)
        )
    return out
