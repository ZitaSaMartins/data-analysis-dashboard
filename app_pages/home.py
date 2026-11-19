import streamlit as st

from .assets import LOGO_PATH


def render() -> None:
    logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
    with logo_col2:
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown('<p class="subtitle">Your easy guide to understanding data without coding</p>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.df is not None:
        st.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
    else:
        st.info("👉 Use the sidebar menu to get started — begin with **Upload Data**")

        st.markdown("### 🗺️ How it works")
        st.markdown("""
        Use the **menu in the sidebar** to navigate between steps:
        1. **Prepare Data** — Upload Data, Check Data Quality, then Rename Variables
        2. **Explore Data** — Choose variables to explore patterns and relationships
        3. **Analyse Data** — Check assumptions, review Suggested Analysis, then choose a test
        4. **Sensory Analysis** — Open the independent Sensory Analysis module when needed
        5. **Advanced Analysis** — Build Predictions, Get Recommendations
        6. **Help** — Getting Started, Analysis Types, and How to Choose Tests guides
        """)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #999; font-size: 0.9rem;'>
            <p>📚 <b>Tip:</b> Follow the steps in order: Upload → Check → Explore → Test → Advanced</p>
            <p>No coding knowledge needed • All analysis in your browser • Your data stays private</p>
        </div>
        """, unsafe_allow_html=True)
