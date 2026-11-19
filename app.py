import streamlit as st

# Import app_pages
from app_pages import advanced, eda_page, help_pages, home, rename, sensory, tests, upload, validate
from app_pages.assets import LOGO_SYMBOL_PATH
from app_pages.eda_page import EDA_PAGE_CATEGORIES
from app_pages.sensory import SENSORY_PAGES
from app_pages.state import init_state
from app_pages.styles import inject_styles
from app_pages.tests import TEST_PAGE_HANDLERS

# Import custom modules
from modules.sidebar_navigation import (
    build_main_navigation,
    build_top_navigation,
    show_sidebar_status,
)

# Page config
st.set_page_config(
    page_title="SciData Studio",
    page_icon=str(LOGO_SYMBOL_PATH),
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_styles()

# Initialize session state
init_state()

build_top_navigation()
build_main_navigation()
show_sidebar_status()


# HOME PAGE - Welcome & status
if st.session_state.page == "home":
    home.render()
elif st.session_state.page == "upload":
    upload.render()
elif st.session_state.page == "rename":
    rename.render()
elif st.session_state.page == "validate":
    validate.render()
elif st.session_state.page in EDA_PAGE_CATEGORIES:
    eda_page.render()
elif st.session_state.page in TEST_PAGE_HANDLERS:
    tests.render()
elif st.session_state.page in SENSORY_PAGES:
    sensory.render()
elif st.session_state.page in ("machine_learning", "recommendations"):
    advanced.render()
elif st.session_state.page.startswith("help_"):
    help_pages.render()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>SciData Studio • No coding required • Your data stays private</p>
    <p style='font-size: 0.75rem;'>Built with Streamlit • Open source • Free to use</p>
</div>
""", unsafe_allow_html=True)
