"""Shared pytest setup.

Hypothesis profiles: ``dev`` (default) keeps local runs fast; ``ci`` runs more
examples and prints a reproduction blob on failure. Select with
``HYPOTHESIS_PROFILE=ci``.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
