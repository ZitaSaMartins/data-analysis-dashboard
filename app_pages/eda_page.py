import streamlit as st

from app_pages.state import (
    set_page,
    store_eda_dependent_selection,
    store_eda_independent_selection,
)
from modules.eda import CHART_CATALOG, CHART_INFO, build_preview, recommend_components, run_eda

EDA_PAGE_CATEGORIES = {
    "eda_distributions": "Distribution",
    "eda_correlations": "Correlation",
    "eda_comparisons": "Comparison",
    "eda_hierarchical": "Part-to-Whole & Hierarchical",
    "eda_temporal": "Temporal",
}


def render() -> None:
    st.session_state.eda_category = EDA_PAGE_CATEGORIES[st.session_state.page]
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 🔍 Explore Your Data")
        st.markdown("Visualize patterns and relationships in your data")

        df = st.session_state.df

        st.markdown("### Select Variables to Analyse")
        if "eda_independent_selection" not in st.session_state:
            st.session_state.eda_independent_selection = list(
                st.session_state.get(
                    "eda_independent_vars",
                    st.session_state.get("independent_vars", []),
                )
            )
        if "eda_dependent_selection" not in st.session_state:
            st.session_state.eda_dependent_selection = list(
                st.session_state.get(
                    "eda_dependent_vars",
                    st.session_state.get("dependent_vars", []),
                )
            )

        selected_independent = [
            variable
            for variable in st.session_state.eda_independent_selection
            if variable in df.columns
        ]
        selected_dependent = [
            variable
            for variable in st.session_state.eda_dependent_selection
            if variable in df.columns and variable not in selected_independent
        ]
        st.session_state.eda_independent_selection = selected_independent
        st.session_state.eda_dependent_selection = selected_dependent

        # Widget keys can be removed during a navigation rerun; restore them
        # from persistent, non-widget state before rendering each EDA page.
        st.session_state._eda_independent_vars = selected_independent
        st.session_state._eda_dependent_vars = selected_dependent

        col1, col2 = st.columns(2)
        with col1:
            eda_independent = st.multiselect(
                "Independent Variables (Features/Predictors):",
                options=df.columns,
                key="_eda_independent_vars",
                on_change=store_eda_independent_selection,
                help="Variables you'll use to predict or explain"
            )
        with col2:
            eda_dependent = st.multiselect(
                "Dependent Variables (Target/Response):",
                options=[c for c in df.columns if c not in eda_independent],
                key="_eda_dependent_vars",
                on_change=store_eda_dependent_selection,
                help="Variables you want to predict or analyse"
            )

        eda_vars = eda_dependent + [v for v in eda_independent if v not in eda_dependent]

        if not eda_vars:
            st.info("👆 Select at least one dependent or independent variable to analyse")
        else:
            with st.expander("🎨 Chart Settings", expanded=False):
                s1, s2, s3 = st.columns(3)
                with s1:
                    chart_template = st.selectbox(
                        "Theme",
                        ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
                        key="chart_template"
                    )
                    chart_palette = st.selectbox(
                        "Color palette (categories)",
                        ["Plotly", "D3", "T10", "G10", "Set1", "Set2", "Set3", "Pastel", "Bold", "Vivid", "Dark24"],
                        key="chart_palette"
                    )
                with s2:
                    chart_colorscale = st.selectbox(
                        "Color scale (heatmap/numeric)",
                        ["RdBu", "Viridis", "Plasma", "Blues", "Greens", "Reds", "Turbo", "Cividis", "Sunset", "Tealrose"],
                        key="chart_colorscale"
                    )
                    chart_height = st.slider("Plot height (px)", 300, 900, 450, step=50, key="chart_height")
                with s3:
                    chart_font_family = st.selectbox(
                        "Font",
                        ["Arial", "Helvetica", "Times New Roman", "Courier New", "Georgia", "Verdana", "Open Sans, sans-serif"],
                        key="chart_font_family"
                    )
                    chart_font_size = st.slider("Font size", 10, 24, 14, key="chart_font_size")

            chart_settings = {
                "template": chart_template,
                "palette": chart_palette,
                "colorscale": chart_colorscale,
                "height": chart_height,
                "font_family": chart_font_family,
                "font_size": chart_font_size,
            }

            st.markdown(f"### {st.session_state.eda_category}")
            st.caption("⭐ marks the chart types recommended for your current variable selection. Each card shows a live preview. Switch categories directly under \"Explore\" in the sidebar.")

            show_summary = st.checkbox("📋 Also include Summary Statistics table", value=False, key="include_summary_stats")

            recommended = recommend_components(df, eda_vars)
            preview_df = df[eda_vars]

            chart_names = list(CHART_CATALOG[st.session_state.eda_category].keys())
            eda_options = ["Summary Statistics"] if show_summary else []
            gallery_cols = st.columns(3)
            for idx, name in enumerate(chart_names):
                info = CHART_INFO[name]
                is_recommended = recommended.get(name, False)
                with gallery_cols[idx % 3]:
                    with st.container(border=True):
                        badge = " ⭐" if is_recommended else ""
                        st.markdown(f"**{info['icon']} {name}{badge}**")
                        st.caption(info['desc'])
                        preview_fig = build_preview(name, preview_df, chart_settings)
                        if preview_fig is not None:
                            st.plotly_chart(
                                preview_fig, use_container_width=True,
                                config={"staticPlot": True, "displayModeBar": False},
                                key=f"preview_{name}"
                            )
                        else:
                            st.caption("Not enough matching variables selected")
                        include = st.checkbox("Include", value=is_recommended, key=f"include_{name}")
                        if include:
                            eda_options.append(name)

            if st.button("🚀 Generate Analysis", type="primary"):
                st.session_state.eda_generated = True

            # Keep rendering on every rerun (e.g. when a customize color picker changes)
            # instead of only on the exact run the button was clicked.
            if st.session_state.get("eda_generated"):
                if eda_options:
                    run_eda(df, eda_options, eda_vars, eda_dependent, eda_independent, chart_settings)
                else:
                    st.warning("Select at least one option")
