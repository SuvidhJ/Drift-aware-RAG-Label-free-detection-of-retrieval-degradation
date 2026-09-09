"""Shared fixtures.

Most tests are pure numerics and run in milliseconds. The few that need the MiniLM
checkpoint are marked ``needs_model`` and skip cleanly when it is absent, so the
suite passes on a machine that has never downloaded it.
"""
from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_model: requires the all-MiniLM-L6-v2 checkpoint")
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260908)


@pytest.fixture(scope="session")
def same_dist(rng):
    """Two windows from one distribution: everything here must read as 'no drift'."""
    return rng.standard_normal((300, 32)), rng.standard_normal((300, 32))


@pytest.fixture(scope="session")
def shifted(rng):
    """Two windows that genuinely differ."""
    return rng.standard_normal((300, 32)), rng.standard_normal((300, 32)) + 0.8


@pytest.fixture(scope="session")
def unit_pool(rng):
    """A pool of unit-norm vectors, like a real index."""
    x = rng.standard_normal((1200, 32))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="session")
def model():
    pytest.importorskip("scipy")
    from embeddings.backends import MiniLMNumpyBackend

    try:
        return MiniLMNumpyBackend()
    except FileNotFoundError as exc:
        pytest.skip(f"MiniLM checkpoint unavailable: {exc}")
