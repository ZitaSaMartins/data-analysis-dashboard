"""Horizontal top bar used to switch between the app's main sections.

Selecting a section here changes which submenu ``build_main_navigation``
renders in the sidebar (see ``main_nav.py``).
"""

import streamlit as st

from ._shared import _active_section, _navigate
from .page_sets import SECTIONS


def build_top_navigation() -> None:
    """Render the horizontal section bar at the top of the main area."""
    active_section = _active_section(SECTIONS)

    # Remember the last page visited within each section so returning to a
    # section (instead of clicking Home first) resumes where the user left off.
    last_page_by_section = st.session_state.setdefault("section_last_page", {})
    last_page_by_section[active_section["id"]] = st.session_state.get("page", "home")

    with st.container(key="top_nav"):
        columns = st.columns(len(SECTIONS))
        for column, section in zip(columns, SECTIONS, strict=False):
            with column:
                is_active = section["id"] == active_section["id"]
                if st.button(
                    f"{section['icon']} {section['label']}",
                    key=f"topnav_{section['id']}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                ) and not is_active:
                    target_page = last_page_by_section.get(
                        section["id"], section["default_page"]
                    )
                    _navigate(target_page)
