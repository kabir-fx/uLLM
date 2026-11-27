"""Shared pytest setup.

Hypothesis profiles: ``dev`` (default) keeps local runs fast; ``ci`` runs more
examples and prints a reproduction blob on failure. Select with
``HYPOTHESIS_PROFILE=ci``.
"""

import os
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, settings
from tests.plans import Plans

from ullm.models import DEFAULT_MODEL

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(scope="session")
def mlx_loaded() -> "tuple[MLXRunner, MLXTokenizer]":
    """The default model through ``load_mlx``, loaded once per session.

    Skips without the model, unless ``ULLM_REQUIRE_MODEL=1`` (set in CI), where
    a missing model is a failure.
    """
    from scripts.fetch_models import local_path

    from ullm.runner.mlx_runner import load_mlx

    if local_path(DEFAULT_MODEL) is None:
        if os.environ.get("ULLM_REQUIRE_MODEL") == "1":
            pytest.fail(f"{DEFAULT_MODEL.repo} is not downloaded and ULLM_REQUIRE_MODEL=1")
        pytest.skip(f"{DEFAULT_MODEL.repo} is not downloaded: run `make models`")
    return load_mlx(DEFAULT_MODEL)


@pytest.fixture
def plans() -> Plans:
    """C3 v2 plans with block tables, one set of tables per test."""
    return Plans()
