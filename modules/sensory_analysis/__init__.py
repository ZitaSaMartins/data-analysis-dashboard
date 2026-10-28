"""Interactive sensory and consumer-science workflows.

This package mirrors the previous single-file module's public API so existing
imports such as ``from modules.sensory_analysis import run_sensory_page``
continue to work unchanged. Internals are split by workflow family:

- ``_shared``: data prep, significance labelling, and common plots.
- ``discrimination``: triangle/paired/duo-trio/tetrad and ranking tests.
- ``profiling``: QDA, temporal curves, product characterization.
- ``panel_performance``: panelist discrimination, repeatability, agreement, scale usage.
- ``cata_jar``: Check-All-That-Apply and Just-About-Right penalty analysis.
- ``hedonic``: liking comparisons, segmentation, and preference mapping.
- ``multivariate``: PCA, GPA, STATIS, HAC, MFA, PLS.
- ``core``: navigation routing and backward-compatible entry points.
"""

from .core import (
    attribute_summary,
    descriptive_analysis,
    hedonic_analysis,
    panelist_performance,
    product_comparison,
    run_sensory_analysis,
    run_sensory_page,
)

__all__ = [
    "run_sensory_page",
    "run_sensory_analysis",
    "descriptive_analysis",
    "hedonic_analysis",
    "attribute_summary",
    "panelist_performance",
    "product_comparison",
]
