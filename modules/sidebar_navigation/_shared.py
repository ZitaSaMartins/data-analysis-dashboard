"""Shared navigation primitives: page switching and active-state helpers."""

import base64
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import streamlit as st

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "images"


@cache
def icon_data_uri(filename: str) -> str:
    """Return a data URI for an SVG in ``images/`` (empty string if missing)."""
    path = IMAGES_DIR / filename
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _navigate(page: str) -> None:
    """Set the active page and immediately rerun the app."""
    st.session_state.page = page
    st.rerun()


def _nav_button(label: str, page: str, key: str) -> None:
    """Render a full-width navigation button with active-state styling."""
    is_active = st.session_state.get("page", "home") == page
    if st.button(
        label,
        key=key,
        width="stretch",
        type="primary" if is_active else "secondary",
    ):
        _navigate(page)


def _is_active(pages: Iterable[str]) -> bool:
    return st.session_state.get("page", "home") in pages


def _active_section(sections: list[dict]) -> dict:
    """Return the section whose ``pages`` set contains the current page.

    Falls back to the first entry (expected to be "home") if the current
    page does not belong to any known section.
    """
    current_page = st.session_state.get("page", "home")
    for section in sections:
        if current_page in section["pages"]:
            return section
    return sections[0]

