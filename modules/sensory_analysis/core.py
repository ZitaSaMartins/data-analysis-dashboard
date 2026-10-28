"""Sensory navigation routing and backward-compatible entry points."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from ._shared import _normalize_columns
from .cata_jar import _cata, _jar
from .discrimination import _discrimination, _ranking
from .hedonic import _external_preference, _hedonic, _internal_preference, _segmentation
from .multivariate import _gpa, _hac, _mfa, _pca, _pls, _statis
from .panel_performance import (
    _panel_agreement,
    _panel_discrimination,
    _panel_repeatability,
    _panel_scale,
)
from .profiling import _product_characterization, _qda, _temporal


ROUTES: dict[str, Callable[[pd.DataFrame], None]] = {
    "sensory_triangle": lambda df: _discrimination(df, "Triangle Test", 1 / 3, "triangle"),
    "sensory_paired": lambda df: _discrimination(df, "Paired Comparison", 1 / 2, "paired"),
    "sensory_duo_trio": lambda df: _discrimination(df, "Duo-Trio Test", 1 / 2, "duotrio"),
    "sensory_tetrad": lambda df: _discrimination(df, "Tetrad Test", 1 / 3, "tetrad"),
    "sensory_ranking": _ranking,
    "sensory_qda": _qda,
    "sensory_temporal": _temporal,
    "sensory_panel_discrimination": _panel_discrimination,
    "sensory_panel_repeatability": _panel_repeatability,
    "sensory_panel_agreement": _panel_agreement,
    "sensory_panel_scale": _panel_scale,
    "sensory_product_characterization": _product_characterization,
    "sensory_cata": _cata,
    "sensory_jar": _jar,
    "sensory_hedonic": _hedonic,
    "sensory_segmentation": _segmentation,
    "sensory_internal_preference": _internal_preference,
    "sensory_external_preference": _external_preference,
    "sensory_pca": _pca,
    "sensory_gpa": _gpa,
    "sensory_statis": _statis,
    "sensory_hac": _hac,
    "sensory_mfa": _mfa,
    "sensory_pls": _pls,
}


def run_sensory_page(df: pd.DataFrame, page: str) -> None:
    """Dispatch an exact sensory navigation route to its interactive workflow."""
    workflow = ROUTES.get(page)
    if workflow is None:
        st.error(f"Unknown sensory route: {page}")
        return
    workflow(_normalize_columns(df))


def run_sensory_analysis(
    df: pd.DataFrame,
    default_analysis_type: str | None = None,
) -> None:
    """Backward-compatible entry point for the original sensory analysis menu."""
    options = {
        "Descriptive Analysis (Sensory Profile)": "sensory_qda",
        "Hedonic Analysis (Liking Scores)": "sensory_hedonic",
        "Attribute Ratings Summary": "sensory_pca",
        "Panelist Performance": "sensory_panel_repeatability",
        "Product Comparison": "sensory_product_characterization",
    }
    labels = list(options)
    index = labels.index(default_analysis_type) if default_analysis_type in options else 0
    selected = st.selectbox(
        "Select sensory analysis type:",
        labels,
        index=index,
        key=f"sensory_analysis_type_{default_analysis_type or 'default'}",
    )
    run_sensory_page(df, options[selected])


def descriptive_analysis(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original descriptive workflow."""
    _qda(_normalize_columns(df))


def hedonic_analysis(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original hedonic workflow."""
    _hedonic(_normalize_columns(df))


def attribute_summary(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original attribute-summary workflow."""
    _pca(_normalize_columns(df))


def panelist_performance(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original panelist workflow."""
    _panel_repeatability(_normalize_columns(df))


def product_comparison(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original product-comparison workflow."""
    _product_characterization(_normalize_columns(df))

