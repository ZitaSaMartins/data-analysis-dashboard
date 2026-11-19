import streamlit as st

from app_pages.state import set_page
from modules.sensory_analysis import run_sensory_page

SENSORY_PAGES = {
    "sensory_triangle",
    "sensory_paired",
    "sensory_duo_trio",
    "sensory_tetrad",
    "sensory_ranking",
    "sensory_qda",
    "sensory_temporal",
    "sensory_panel_discrimination",
    "sensory_panel_repeatability",
    "sensory_panel_agreement",
    "sensory_panel_scale",
    "sensory_product_characterization",
    "sensory_cata",
    "sensory_jar",
    "sensory_hedonic",
    "sensory_segmentation",
    "sensory_internal_preference",
    "sensory_external_preference",
    "sensory_pca",
    "sensory_gpa",
    "sensory_statis",
    "sensory_hac",
    "sensory_mfa",
    "sensory_pls",
}

def render() -> None:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 👃 Sensory Analysis")
        st.markdown("Consumer preferences and sensory evaluation")

        run_sensory_page(st.session_state.df, st.session_state.page)
