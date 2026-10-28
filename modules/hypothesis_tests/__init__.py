"""Legacy statistical workflow (superseded by ``hypothesis_tests_comprehensive``).

Kept for reference and for instructors following the "add your own test" guide.
This package mirrors the previous single-file module's public API so existing
imports such as ``from modules.hypothesis_tests import run_hypothesis_tests``
continue to work unchanged.
"""

from .catalog import (
    HYPOTHESIS_CATALOG,
    NONPARAMETRIC_TEST_GROUPS,
    PARAMETRIC_TEST_GROUPS,
)
from .core import run_hypothesis_tests

__all__ = [
    "run_hypothesis_tests",
    "HYPOTHESIS_CATALOG",
    "PARAMETRIC_TEST_GROUPS",
    "NONPARAMETRIC_TEST_GROUPS",
]
