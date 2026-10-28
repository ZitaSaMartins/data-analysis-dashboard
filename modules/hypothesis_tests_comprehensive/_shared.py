"""Shared helpers for the comprehensive hypothesis-testing pages."""

import numpy as np
import pandas as pd
import streamlit as st

ALPHA = 0.05


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _show_inference(
    p_value: float,
    significant_text: str,
    nonsignificant_text: str,
) -> None:
    if not np.isfinite(p_value):
        st.error("The test produced an invalid p-value. Check the selected data.")
    elif p_value <= ALPHA:
        st.success(f"✅ {significant_text} (p ≤ {ALPHA}).")
    else:
        st.info(f"❌ {nonsignificant_text} (p > {ALPHA}).")


def _show_statistic(
    statistic_label: str,
    statistic: float,
    p_value: float,
) -> None:
    statistic_col, p_value_col = st.columns(2)
    statistic_col.metric(statistic_label, f"{statistic:.4f}")
    p_value_col.metric("p-value", f"{p_value:.4f}")


def _require_numeric(df: pd.DataFrame, minimum: int = 1) -> list[str] | None:
    columns = _numeric_columns(df)
    if len(columns) < minimum:
        st.error(f"This test requires at least {minimum} numeric variable(s).")
        return None
    return columns


def _require_categorical(df: pd.DataFrame, minimum: int = 1) -> list[str] | None:
    columns = _categorical_columns(df)
    if len(columns) < minimum:
        st.error(f"This test requires at least {minimum} categorical variable(s).")
        return None
    return columns


def _two_group_inputs(
    df: pd.DataFrame,
    key_prefix: str,
) -> tuple[str, str, object, object] | None:
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return None

    numeric_variable = st.selectbox(
        "Numeric outcome variable:",
        numeric_columns,
        key=f"{key_prefix}_numeric",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key=f"{key_prefix}_grouping",
    )
    groups = df[grouping_variable].dropna().unique().tolist()
    if len(groups) < 2:
        st.error("The grouping variable must contain at least two groups.")
        return None

    group_col1, group_col2 = st.columns(2)
    group_1 = group_col1.selectbox(
        "Group 1:",
        groups,
        key=f"{key_prefix}_group_1",
    )
    remaining_groups = [group for group in groups if group != group_1]
    group_2 = group_col2.selectbox(
        "Group 2:",
        remaining_groups,
        key=f"{key_prefix}_group_2",
    )
    return numeric_variable, grouping_variable, group_1, group_2


def _paired_columns(
    df: pd.DataFrame,
    key_prefix: str,
) -> tuple[str, str] | None:
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return None
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First measurement:",
        numeric_columns,
        key=f"{key_prefix}_first",
    )
    second_options = [column for column in numeric_columns if column != first]
    second = col2.selectbox(
        "Second measurement:",
        second_options,
        key=f"{key_prefix}_second",
    )
    return first, second


def _store_assumption_variables(widget_key: str) -> None:
    st.session_state.assumption_variables = list(
        st.session_state.get(widget_key, [])
    )


def _select_assumption_variables(
    numeric_columns: list[str],
    widget_key: str,
    legacy_key: str,
) -> list[str]:
    if "assumption_variables" not in st.session_state:
        legacy_selection = st.session_state.get(legacy_key)
        st.session_state.assumption_variables = list(
            legacy_selection
            if legacy_selection is not None
            else numeric_columns[: min(3, len(numeric_columns))]
        )

    selected_variables = [
        variable
        for variable in st.session_state.assumption_variables
        if variable in numeric_columns
    ]
    st.session_state.assumption_variables = selected_variables
    st.session_state[widget_key] = selected_variables
    return st.multiselect(
        "Numeric variables:",
        numeric_columns,
        key=widget_key,
        on_change=_store_assumption_variables,
        args=(widget_key,),
    )

