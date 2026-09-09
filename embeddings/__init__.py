"""
embeddings - encoding pipeline
==============================

Turns text into L2-normalised float32 vectors, behind a backend interface so the
choice of model is configuration rather than code.

``backends.py``
    ``MiniLMNumpyBackend`` (default) evaluates all-MiniLM-L6-v2 in pure numpy,
    straight from the safetensors checkpoint - no torch, no transformers, no
    tokenizers package. It is what makes the pipeline runnable on a machine without a
    deep-learning install, and it produced every result in FINDINGS.md 11-12.

    ``SentenceTransformerBackend`` is the original path (BGE-M3 and friends). It needs
    the ``heavy`` extra and is imported lazily, so its absence costs nothing.

    Select with ``EMBED_BACKEND``.

``encoder.py``
    ``Encoder.encode_cached(texts, name)`` is content-addressed per *text*: the cache
    key is a hash of the text plus the backend name. Identical text therefore always
    maps to the vector the current backend produced for it, slices that overlap
    (the null slice's documents are exactly the stale revisions of the real slice)
    cost nothing to re-encode, and switching backends switches cache file rather than
    silently reusing vectors from a different model.

    ``Encoder.encode_snapshot_dir(dir, prefix)`` additionally deduplicates across
    windows. Over T windows a document takes only two distinct values - its earlier
    and later revision - so the union of texts is about 2n rather than T*n. On the
    12-window arXiv build that is 8,329 forward passes instead of 88,152.

**Removed:** the Lipschitz estimator. It measured the wrong object (the
Kantorovich-Rubinstein bound needs the Lipschitz constant of the *retrieval
functional*, and W1 in D(t) is already measured in embedding space), and its
estimator was invalid independently of that - the denominator was a normalised
character edit distance, which scales as 1/document-length, so ``L_hat = max`` was
always won by the longest document. See FINDINGS.md F15. Nothing consumed it.
"""
