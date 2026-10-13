"""Configuration.

Every field is overridable by environment variable using the section prefix, e.g.
``DRIFT_ALPHA_LEVEL=0.01``.  A ``.env`` file is read if present.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DriftConfig(BaseSettings):
    """Defaults for the drift statistics and their calibration."""

    model_config = SettingsConfigDict(env_prefix="DRIFT_", env_file=".env", extra="ignore")

    # estimators
    w1_n_projections: int = Field(default=256, ge=32)
    w1_bootstrap_iters: int = Field(default=200, ge=0)
    mmd_subsample: int = Field(default=800, ge=50)
    mmd_sigma: float | None = None

    # calibration
    alpha_level: float = Field(default=0.05, gt=0.0, lt=0.5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    drift: DriftConfig = Field(default_factory=DriftConfig)


settings = Settings()
