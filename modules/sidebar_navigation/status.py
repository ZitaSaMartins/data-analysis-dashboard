"""Sidebar footer showing the current dataset status."""

import streamlit as st


def show_sidebar_status() -> None:
    """Display the current dataset status at the bottom of the sidebar."""
    with st.sidebar:
        st.markdown("---")
        df = st.session_state.get("df")
        if df is not None:
            filename = st.session_state.get("uploaded_filename") or "Not available"
            st.success(
                f"✅ **File name:** {filename}\n\n"
                f"**Data loaded:** {df.shape[0]} rows × {df.shape[1]} columns"
            )
        else:
            st.info("👉 Upload data to get started")

