"""LEGACY: download Amazon Reviews 2023 meta + reviews for a category.

This is not the default data path and is not recommended.  It is kept, with its
defects repaired, so the original experiment remains reproducible for anyone who
wants it.  Use ``DATA_SOURCE=revision`` (the default) for the real revision corpora.

Repairs
-------
F9  ``uv.lock`` pins datasets 5.0.1, in which loading scripts and ``trust_remote_code``
    were removed (datasets 4.0).  All three call sites passed it, so phase 0 raised on
    the first call.  Removed.

F3  The Q&A config ``qas_<category>`` does not exist -- the dataset ships only
    ``raw_review_*`` and ``raw_meta_*``, and its card states there is no
    question-answering split.  The request therefore always raised, a bare ``except``
    caught it, and the fallback emitted the literal string "What do users say about
    this product?" for *every* row.  Every query embedding in every window was
    bit-identical, so W1(Q_t, Q_{t-1}) was identically zero and the query axis of the
    taxonomy had no data at all.  The fallback is deleted; a missing query source is
    now a hard error naming the problem.

F24 ``ds.to_pandas().head(n)`` materialised a 20.8M-row split before truncating, and
    ``head()`` is a prefix, not a sample.  Now streams and takes, with an explicit note
    that a prefix of a time-ordered dump is not a random sample.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
from loguru import logger

from configs.settings import settings


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "the Amazon data path needs the heavy extra:\n"
            "  uv pip install -e .[heavy]\n"
            "The default DATA_SOURCE=revision path needs none of it."
        ) from exc
    return load_dataset


def _stream_take(repo: str, config: str, max_rows: int) -> pd.DataFrame:
    """Take the first ``max_rows`` rows without materialising the whole split.

    NOTE: this is a prefix of a time-ordered dump, not a random sample. Any statistic
    computed from it inherits that ordering; do not treat it as representative.
    """
    load_dataset = _require_datasets()
    ds = load_dataset(repo, config, split="full", streaming=True)
    rows = list(itertools.islice(ds, max_rows))
    if not rows:
        raise RuntimeError(f"{repo}:{config} yielded no rows")
    return pd.DataFrame(rows)


def download(
    category: str | None = None,
    null_category: str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    category = category or settings.data.category
    null_category = null_category or settings.data.null_category
    raw_dir = settings.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for cat in [category, null_category]:
        meta_path = raw_dir / f"meta_{cat}.parquet"
        reviews_path = raw_dir / f"reviews_{cat}.parquet"

        if not force and meta_path.exists() and reviews_path.exists():
            logger.info(f"{cat}: already cached, skipping (use --force to re-download)")
            paths[cat] = raw_dir
            continue

        logger.info(f"{cat}: downloading metadata (max {settings.data.max_products})")
        meta = _stream_take(settings.data.hf_repo, f"raw_meta_{cat}", settings.data.max_products)
        meta.to_parquet(meta_path, index=False)

        logger.info(f"{cat}: downloading reviews (max {settings.data.max_reviews})")
        reviews = _stream_take(settings.data.hf_repo, f"raw_review_{cat}", settings.data.max_reviews)
        reviews.to_parquet(reviews_path, index=False)

        logger.info(f"{cat}: meta {len(meta)} rows, reviews {len(reviews)} rows")
        paths[cat] = raw_dir

    logger.warning(
        "Amazon Reviews 2023 ships no question-answering split, so this path provides "
        "no query stream. Supply one explicitly before computing query-side drift, or "
        "use DATA_SOURCE=revision, whose queries are real sentences with known gold "
        "documents."
    )
    return paths


if __name__ == "__main__":
    download()
