"""
retrieval - index, selective refresh, and the RAG loop
======================================================

``index.py``
    ``NumpyIndex`` does exact cosine search in one matmul. At 10^3-10^4 documents an
    approximate index buys nothing and costs a heavyweight dependency, and exactness
    removes recall noise that would otherwise be confounded with the drift effect
    being measured. ``ChromaIndex`` is available behind the ``heavy`` extra.

    ``reindex_rows(rows, fresh)`` is the capability that was missing: the previous
    pipeline could only ``build_index(force=True)``, rebuilding the whole collection,
    so three of five remediation actions existed only in an enum (F16).

``select.py`` - which documents to re-embed under a budget
    Five selectors sharing one contract: re-embed a small probe (2% by default),
    observe how far those moved, then choose the rest blind.

    Two structural results make this possible at all, and both replicated on real
    revisions: displacement is spatially autocorrelated (Moran's I +0.1184 / +0.0557 /
    +0.0311), and its *magnitude* is predictable from a probe while its *direction* is
    not - interpolating the displacement vector field and moving documents there is
    worse than leaving them alone, in 6 of 6 conditions (12.2, 12.3).

    One claimed advantage did not replicate. probe-kNN beats random by 1.26-1.58x, but
    given the *same* probe it ties or loses to ``cluster_threshold`` - the closest
    deployed prior art - at every budget tested (12.5). Neither approaches a change
    log. Read that before choosing a selector.

``rerank.py``
    MMR re-ranking, action a2. Returns positions into the candidate array so callers
    can carry their own row indices and scores alongside.

``rag_pipeline.py``
    Retrieve, optionally re-rank, optionally generate. Generation is off by default:
    the endpoint that matters is whether the index still returns the right document,
    and an LLM in the loop adds cost and variance without adding signal. Failures
    propagate rather than becoming empty results - three bare ``except Exception``
    handlers on the critical path used to convert hard failures into silence while the
    run reported success (F23).
"""
