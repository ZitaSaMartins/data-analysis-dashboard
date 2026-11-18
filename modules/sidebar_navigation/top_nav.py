"""Horizontal top bar used to switch between the app's main sections.

Selecting a section here changes which submenu ``build_main_navigation``
renders in the sidebar (see ``main_nav.py``).
"""

import streamlit as st

from ._shared import _active_section, _navigate, icon_data_uri
from .page_sets import SECTIONS


def _icon_css() -> str:
    """CSS that places each section's SVG icon before its button label."""
    rules = []
    for section in SECTIONS:
        uri = icon_data_uri(section["icon"])
        if not uri:
            continue
        rules.append(
            f".st-key-topnav_{section['id']} button p::before {{"
            f' content: ""; display: inline-block; width: 1.4rem; height: 1.4rem;'
            f' margin-right: 0.45rem; vertical-align: middle;'
            f' background: url("{uri}") center / contain no-repeat; }}'
        )
    return "<style>" + "\n".join(rules) + "</style>"


def build_top_navigation() -> None:
    """Render the horizontal section bar at the top of the main area."""
    active_section = _active_section(SECTIONS)

    # Remember the last page visited within each section so returning to a
    # section (instead of clicking Home first) resumes where the user left off.
    last_page_by_section = st.session_state.setdefault("section_last_page", {})
    last_page_by_section[active_section["id"]] = st.session_state.get("page", "home")

    st.markdown(_icon_css(), unsafe_allow_html=True)

    with st.container(key="top_nav"):
        columns = st.columns(len(SECTIONS))
        for column, section in zip(columns, SECTIONS, strict=False):
            with column:
                is_active = section["id"] == active_section["id"]
                if st.button(
                    section["label"],
                    key=f"topnav_{section['id']}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                ) and not is_active:
                    target_page = last_page_by_section.get(
                        section["id"], section["default_page"]
                    )
                    _navigate(target_page)
