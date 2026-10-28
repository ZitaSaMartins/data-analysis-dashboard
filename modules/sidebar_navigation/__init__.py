"""Hierarchical sidebar navigation and statistical-test guidance.

This package mirrors the previous single-file module's public API so existing
imports such as ``from modules.sidebar_navigation import build_main_navigation``
continue to work unchanged.

- ``page_sets``: page-id sets used to decide which sidebar section auto-expands.
- ``_shared``: page switching and active-state helpers.
- ``main_nav``: the hierarchical sidebar navigation builder.
- ``decision_tree``: the interactive "choose a test" decision guide.
- ``status``: the sidebar footer showing the current dataset status.
"""

from .decision_tree import show_test_decision_tree
from .main_nav import build_main_navigation
from .status import show_sidebar_status

__all__ = [
    "build_main_navigation",
    "show_sidebar_status",
    "show_test_decision_tree",
]
