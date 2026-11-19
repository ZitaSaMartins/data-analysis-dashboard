# Custom styling - Clean, student-friendly interface

import streamlit as st


def inject_styles() -> None:
    st.markdown("""
    <style>
    /* Hide the default sidebar menu */
    [data-testid="stSidebarNav"] {display: none;}

    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Card styling for menu buttons */
    .menu-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        cursor: pointer;
        text-align: center;
        transition: transform 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem;
        font-size: 1.1rem;
        font-weight: bold;
    }

    .menu-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    .menu-card.upload { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .menu-card.validate { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .menu-card.eda { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .menu-card.hypothesis { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .menu-card.sensory { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .menu-card.ml { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); }
    .menu-card.recommend { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }

    .menu-card.recommend { color: #333; }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }

    .step-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-right: 1rem;
    }

    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #0288d1;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .back-button {
        background: #f0f0f0;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        margin-bottom: 1rem;
        display: inline-block;
    }

    /* Sidebar menu - flat list style with chevrons, like a help-center nav */
    section[data-testid="stSidebar"] p.nav-section-label {
        color: #8DB2D9;
        opacity: 0.75;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.25rem 0 0.25rem 0.25rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        position: relative;
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background-color: transparent;
        color: #C7DCE5;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0;
        padding: 0.55rem 1.6rem 0.55rem 1.15rem;
        font-weight: 500;
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        width: 100%;
        margin: 0;
        text-align: left;
    }

    section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button > div,
    section[data-testid="stSidebar"] .stButton > button > div > span {
        width: 100%;
        justify-content: flex-start;
    }

    section[data-testid="stSidebar"] .stButton > button::after {
        content: "›";
        position: absolute;
        right: 0.85rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.1rem;
        opacity: 0.55;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.08);
        color: #ffffff;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border-left: 3px solid #FFBF3E;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: rgba(255,255,255,0.14);
    }

    /* Top section bar - horizontal replacement for the long sidebar menu */
    .st-key-top_nav {
        background-color: #0E3C61;
        border-radius: 10px;
        padding: 0.5rem 0.5rem;
        margin-bottom: 1.5rem;
    }

    .st-key-top_nav .stButton > button {
        width: 100%;
        min-height: 3rem;
        background-color: transparent;
        color: #C7DCE5;
        border: none;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        padding: 0.6rem 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
        white-space: normal;
        line-height: 1.2;
    }

    .st-key-top_nav .stButton > button p {
        white-space: normal;
    }

    .st-key-top_nav .stButton > button:hover {
        background-color: rgba(255,255,255,0.08);
        color: #ffffff;
    }

    .st-key-top_nav .stButton > button[kind="primary"] {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border-bottom: 3px solid #FFBF3E;
    }
    </style>
""", unsafe_allow_html=True)
