import hashlib

import pandas as pd
import streamlit as st

from app_pages.state import set_page


def render() -> None:
    st.markdown("## 📤 Upload Your Data")
    st.markdown("Choose an Excel or CSV file with your data")

    if st.session_state.uploaded_filename:
        st.caption(f"📄 Currently loaded file: **{st.session_state.uploaded_filename}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📎 Select file:",
            type=["xlsx", "xls", "csv", "xlsm"],
            help="Supported formats: Excel (.xlsx, .xls, .xlsm) or CSV"
        )

    if uploaded_file is not None:
        try:
            dataset_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            dataset_changed = dataset_signature != st.session_state.dataset_signature
            if dataset_changed:
                st.session_state.distribution_test_result = None
                st.session_state.variance_test_result = None
                st.session_state.dataset_signature = dataset_signature

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.session_state.uploaded_filename = uploaded_file.name
            if dataset_changed:
                st.rerun()
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")

                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("📋 Total Columns", df.shape[1])
                with col3:
                    st.metric("⚠️ Missing Values", df.isnull().sum().sum())

                # Preview
                st.markdown("### Preview of Your Data")
                st.dataframe(df.head(10), use_container_width=True)

                st.info("✨ Great! Your data is uploaded. Next step: Rename variables (optional) →")

            if st.button("✏️ Rename Variables", type="primary"):
                set_page("rename")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Try saving your file as CSV or Excel format")
    else:
        st.info("👆 Click above to select your file")

        with st.expander("ℹ️ What file should I use?"):
            st.markdown("""
            **Accepted formats:**
            - Excel: .xlsx, .xls
            - Spreadsheet: .csv
    
            **Requirements:**
            - First row: Column names (headers)
            - Data below: Your actual data
            - No blank rows in the middle
    
            **Example:**
            | Product | Score | Temperature |
            |---------|-------|-------------|
            | A | 7.5 | 23 |
            | B | 8.2 | 24 |
            """)
