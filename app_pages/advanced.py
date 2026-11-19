import streamlit as st

from app_pages.state import set_page
from modules.ml_models import run_ml_pipeline
from modules.model_recommender import recommend_models


def render() -> None:
    if st.session_state.page == "machine_learning":
        _machine_learning()
    elif st.session_state.page == "recommendations":
        _recommendations()

def _machine_learning() -> None:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 🤖 Build Predictions")
        st.markdown("Train machine learning models to predict outcomes")

        if not hasattr(st.session_state, 'dependent_vars') or not hasattr(st.session_state, 'independent_vars'):
            st.warning("⚠️ Please go to 'Check Data Quality' and specify your target and predictor variables first!")
            if st.button("Go to Check Data Quality"):
                set_page("validate")
        else:
            df = st.session_state.df
            run_ml_pipeline(
                df,
                st.session_state.dependent_vars,
                st.session_state.independent_vars
            )

def _recommendations() -> None:
    st.markdown("## 💡 Which Model Should I Use?")
    st.markdown("Get expert recommendations based on your data and question")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        df = st.session_state.df

        col1, col2 = st.columns(2)

        with col1:
            analysis_type = st.selectbox(
                "🎯 What are you trying to do?",
                ["EDA (Explore data)", "Statistical Inference (Test differences)",
                 "Classification (Predict categories)", "Regression (Predict numbers)",
                 "Sensory Analysis (Consumer preferences)"],
                format_func=lambda x: x.split("(")[0].strip() if "(" in x else x
            )

        with col2:
            st.markdown("**Your data looks like:**")
            has_small = st.checkbox("Small dataset (<100 rows)", value=False)
            has_imbalance = st.checkbox("Imbalanced classes", value=False)
            has_nonlinear = st.checkbox("Complex patterns", value=False)

        data_characteristics = []
        if has_small:
            data_characteristics.append("Small sample size (<100)")
        if has_imbalance:
            data_characteristics.append("Imbalanced classes")
        if has_nonlinear:
            data_characteristics.append("Non-linear relationships")

        if st.button("📌 Get Recommendations", type="primary"):
            # Extract analysis type from the formatted string
            analysis_type_mapping = {
                "EDA (Explore data)": "EDA",
                "Statistical Inference (Test differences)": "Hypothesis Testing",
                "Classification (Predict categories)": "Classification",
                "Regression (Predict numbers)": "Regression",
                "Sensory Analysis (Consumer preferences)": "Sensory Analysis"
            }
            analysis_type_key = analysis_type_mapping.get(analysis_type, analysis_type)

            recommendations = recommend_models(
                analysis_type_key,
                df,
                data_characteristics
            )

            st.markdown("---")
            st.markdown("### 📊 Top Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                color = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                with st.expander(f"{color} **{rec['model']}** (Suitability: {rec['suitability']}/10)", expanded=(i==1)):
                    st.markdown(f"**💡 Why use this?**\n{rec['reason']}")
                    st.markdown(f"**✅ Advantages:**\n{rec['pros']}")
                    st.markdown(f"**⚠️ Disadvantages:**\n{rec['cons']}")
                    st.markdown(f"**🎯 Best for:**\n{rec['when_to_use']}")
