import pandas as pd
import streamlit as st

from app_pages.state import set_page
from modules.data_validation import generate_validation_report, validate_dataset


def render() -> None:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## ✅ Check Data Quality")
        st.markdown("Making sure your data is ready for analysis")

        df = st.session_state.df

        # Run validation
        validation_result = validate_dataset(df)
        st.session_state.validation_report = validation_result

        # Display validation results
        if validation_result['is_valid']:
            st.success("✅ Data validation passed!")
        else:
            st.warning("⚠️ Data has issues that should be reviewed:")

        # Validation Report Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Column Info", "Missing Data", "Issues"])

        with tab1:
            st.markdown("**Dataset Summary:**")
            report = generate_validation_report(df)
            st.write(report)

        with tab2:
            st.markdown("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(df[col].dtype) for col in df.columns],
                'Non-Null Count': [df[col].notna().sum() for col in df.columns],
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

        with tab3:
            st.markdown("**Missing Values Analysis:**")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing[missing > 0].index,
                'Missing Count': missing[missing > 0].values,
                'Percentage': missing_pct[missing > 0].values
            })
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values!")

        with tab4:
            if validation_result['issues']:
                for issue in validation_result['issues']:
                    if issue.startswith('❌'):
                        st.error(issue)
                    else:
                        st.warning(issue)
            else:
                st.success("✅ No data format issues detected!")

        # Dependent/Independent variable specification
        st.markdown('<p class="section-text">Specify Variable Roles</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            independent_vars = st.multiselect(
                "Select Independent Variables (Features/Predictors):",
                options=df.columns,
                key="independent_vars",
                help="Variables you'll use to predict or explain"
            )

        with col2:
            dependent_vars = st.multiselect(
                "Select Dependent Variables (Target/Response):",
                options=[col for col in df.columns if col not in independent_vars],
                key="dependent_vars",
                help="Variables you want to predict or analyse"
            )

        if dependent_vars and independent_vars:
            st.success("✅ Variable roles configured!")
            if st.button("🤖 Go to Build Predictions"):
                set_page("machine_learning")



