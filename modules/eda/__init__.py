"""Exploratory data analysis: chart catalog, previews, and chart rendering.

This package mirrors the previous single-file module's public API so existing
imports such as ``from modules.eda import run_eda`` continue to work unchanged.
Internals are split by concern:

- ``theme``: palettes, default settings, shared plot rendering.
- ``catalog``: chart catalog metadata and component recommendations.
- ``preview``: small read-only chart thumbnails for the chart picker UI.
- ``registry``: the renderer registry and the ``run_eda`` page entry point.
- ``distribution`` / ``correlation`` / ``comparison`` / ``part_to_whole`` /
  ``temporal``: individual chart renderers, grouped by chart family. Importing
  them (below) registers each renderer into ``registry._RENDERERS``.
"""

from . import comparison, correlation, distribution, part_to_whole, temporal  # noqa: F401
from .catalog import CHART_CATALOG, CHART_INFO, recommend_components
from .preview import build_preview
from .registry import run_eda

__all__ = [
    "run_eda",
    "recommend_components",
    "build_preview",
    "CHART_INFO",
    "CHART_CATALOG",
]
