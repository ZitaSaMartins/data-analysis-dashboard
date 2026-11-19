import pandas as pd
import streamlit as st

from app_pages.state import set_page


def render() -> None:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## ✏️ Rename Variables")
        st.markdown("Give your columns clearer names — accents, special characters, and any language script are supported")

        df = st.session_state.df

        st.caption("Type any characters, accents, or language script — e.g. 'Température', '甜み', 'Préférence (n=50)'.")
        rename_table = pd.DataFrame({"Current Name": df.columns, "New Name": df.columns})
        edited_names = st.data_editor(
            rename_table,
            column_config={
                "Current Name": st.column_config.TextColumn(disabled=True),
                "New Name": st.column_config.TextColumn(),
            },
            hide_index=True,
            width="stretch",
            key="rename_editor"
        )

        if st.button("✅ Apply Renaming", type="primary"):
            new_names = [str(n).strip() for n in edited_names["New Name"]]
            if any(not n for n in new_names):
                st.error("❌ Column names cannot be empty.")
            elif len(set(new_names)) != len(new_names):
                st.error("❌ Column names must be unique.")
            else:
                rename_map = {old: new for old, new in zip(df.columns, new_names, strict=False) if old != new}
                if rename_map:
                    df = df.rename(columns=rename_map)
                    st.session_state.df = df
                    st.session_state.distribution_test_result = None
                    st.session_state.variance_test_result = None
                    st.session_state.dataset_signature = None
                    for state_key in (
                        "dependent_vars",
                        "independent_vars",
                        "eda_dependent_vars",
                        "eda_independent_vars",
                        "eda_dependent_selection",
                        "eda_independent_selection",
                        "assumption_variables",
                    ):
                        if state_key in st.session_state:
                            st.session_state[state_key] = [rename_map.get(v, v) for v in st.session_state[state_key]]
                    st.success(f"✅ Renamed {len(rename_map)} column(s)! The new names are used everywhere in the dashboard.")
                else:
                    st.info("No changes to apply.")

        if st.button("➡️ Check Data Quality"):
            set_page("validate")

