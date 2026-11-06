"""Top bar + section-scoped sidebar navigation and statistical-test guidance.

- ``page_sets``: page-id sets and the ``SECTIONS`` list used to decide which
  section is active and which sidebar submenu to render.
- ``_shared``: page switching and active-section helpers.
- ``top_nav``: the horizontal top bar used to switch between sections.
- ``main_nav``: renders only the active section's submenu in the sidebar.
- ``decision_tree``: the interactive "choose a test" decision guide.
- ``status``: the sidebar footer showing the current dataset status.
"""

from .decision_tree import show_test_decision_tree
from .main_nav import build_main_navigation
from .status import show_sidebar_status
from .top_nav import build_top_navigation

__all__ = [
    "build_main_navigation",
    "build_top_navigation",
    "show_sidebar_status",
    "show_test_decision_tree",
]
