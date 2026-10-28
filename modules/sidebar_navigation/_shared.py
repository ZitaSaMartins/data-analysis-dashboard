"""Shared navigation primitives: page switching and active-state helpers."""

from collections.abc import Iterable

import streamlit as st


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

