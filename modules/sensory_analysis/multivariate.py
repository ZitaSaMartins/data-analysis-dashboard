"""Multivariate / multi-assessor methods: PCA, GPA, STATIS, HAC, MFA, PLS."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from scipy import linalg
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ._shared import _columns, _need, _scatter_labels


def _pca(df: pd.DataFrame) -> None:
    st.header("Principal Component Analysis")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need at least two numeric variables."):
        return
    label = st.selectbox("Point label", ["(row number)", *df.columns.astype(str).tolist()], key="pca_label")
    features = st.multiselect("Numeric variables", numeric, default=numeric[:8], key="pca_features")
    clean = df[features].dropna()
    if not _need(len(features) >= 2 and clean.shape[0] >= 3, "Select 2 variables with 3 complete rows."):
        return
    pca = PCA(n_components=2).fit(StandardScaler().fit_transform(clean))
    scores = pca.transform(StandardScaler().fit_transform(clean))
    labels = clean.index if label == "(row number)" else df.loc[clean.index, label]
    _scatter_labels(scores, labels, "PCA scores",
                    f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    st.dataframe(pd.DataFrame(pca.components_.T, index=features, columns=["PC1 loading", "PC2 loading"]),
                 width="stretch")
    st.caption("Nearby observations have similar standardized profiles; large loadings drive separation.")


def _multiassessor_inputs(df: pd.DataFrame, prefix: str) -> tuple[str, str, list[str]] | None:
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor/block, product/object, and numeric attributes."):
        return None
    assessor = st.selectbox("Assessor / table", columns, key=f"{prefix}_assessor")
    product = st.selectbox("Product / object", [c for c in columns if c != assessor], key=f"{prefix}_product")
    attributes = st.multiselect("Common attributes", [c for c in numeric if c not in {assessor, product}],
                                default=[c for c in numeric if c not in {assessor, product}][:6],
                                key=f"{prefix}_attributes")
    if not _need(len(attributes) >= 2, "Select at least two common attributes."):
        return None
    return assessor, product, attributes


def _gpa(df: pd.DataFrame) -> None:
    st.header("Generalized Procrustes Analysis (GPA)")
    st.write("Centers, normalizes, rotates, and iteratively averages assessor configurations.")
    selected = _multiassessor_inputs(df, "gpa")
    if selected is None:
        return
    assessor, product, attributes = selected
    tables: list[np.ndarray] = []
    labels: pd.Index | None = None
    for _, group in df.groupby(assessor, observed=True):
        table = group.groupby(product, observed=True)[attributes].mean().dropna()
        labels = table.index if labels is None else labels.intersection(table.index)
    if not _need(
        df[assessor].nunique() >= 2 and labels is not None and len(labels) >= 3,
        "Need at least 2 assessors and 3 complete products shared by all assessors.",
    ):
        return
    for _, group in df.groupby(assessor, observed=True):
        table = group.groupby(product, observed=True)[attributes].mean().loc[labels]
        values = table.to_numpy(dtype=float)
        values -= values.mean(axis=0)
        norm = np.linalg.norm(values)
        tables.append(values / norm if norm else values)
    consensus = tables[0]
    for _ in range(30):
        aligned = [table @ linalg.orthogonal_procrustes(table, consensus)[0] for table in tables]
        updated = np.mean(aligned, axis=0)
        if np.linalg.norm(updated - consensus) < 1e-8:
            break
        consensus = updated
    coordinates = PCA(n_components=2).fit_transform(consensus)
    _scatter_labels(coordinates, labels, "GPA consensus")
    residuals = [float(np.linalg.norm(table - consensus)) for table in aligned]
    st.dataframe(pd.DataFrame({"Assessor table": range(1, len(residuals) + 1), "Residual": residuals}),
                 width="stretch")
    st.caption("Close products have similar consensus profiles; smaller residuals indicate assessor conformity.")


def _statis(df: pd.DataFrame) -> None:
    st.header("STATIS Compromise")
    st.write("RV similarity weights multiple assessor tables into a shared compromise.")
    selected = _multiassessor_inputs(df, "statis")
    if selected is None:
        return
    assessor, product, attributes = selected
    grouped = [(name, group.groupby(product, observed=True)[attributes].mean().dropna())
               for name, group in df.groupby(assessor, observed=True)]
    if not _need(
        len(grouped) >= 2,
        "Need at least two assessor tables (check the assessor column for missing values).",
    ):
        return
    common = grouped[0][1].index
    for _, table in grouped[1:]:
        common = common.intersection(table.index)
    if not _need(len(common) >= 3, "Need two tables sharing at least three products."):
        return
    tables = [StandardScaler().fit_transform(table.loc[common]) for _, table in grouped]
    cross_products = [table @ table.T for table in tables]
    traces = [float(np.trace(cross_product)) for cross_product in cross_products]
    zero_norm_names = [str(name) for (name, _), trace in zip(grouped, traces) if trace <= 1e-12]
    if zero_norm_names:
        st.warning(
            "Excluded assessor table(s) with constant scores across the shared products "
            f"(zero variance): {', '.join(zero_norm_names)}."
        )
        keep = [trace > 1e-12 for trace in traces]
        grouped = [item for item, keep_item in zip(grouped, keep) if keep_item]
        tables = [table for table, keep_item in zip(tables, keep) if keep_item]
        cross_products = [cp for cp, keep_item in zip(cross_products, keep) if keep_item]
    if not _need(len(grouped) >= 2, "Need at least two non-constant assessor tables."):
        return
    rv = np.array([
        [np.trace(a @ b) / np.sqrt(np.trace(a @ a) * np.trace(b @ b)) for b in cross_products]
        for a in cross_products
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(rv)
    weights = np.abs(eigenvectors[:, -1])
    weights /= weights.sum()
    compromise = sum(weight * cross_product for weight, cross_product in zip(weights, cross_products))
    compromise_eigenvalues, compromise_eigenvectors = np.linalg.eigh(compromise)
    order = np.argsort(compromise_eigenvalues)[::-1][:2]
    scale = np.sqrt(np.clip(compromise_eigenvalues[order], 0, None))
    coordinates = compromise_eigenvectors[:, order] * scale
    _scatter_labels(coordinates, common, "STATIS compromise")
    st.dataframe(pd.DataFrame(rv, index=[str(x[0]) for x in grouped], columns=[str(x[0]) for x in grouped]),
                 width="stretch")
    st.dataframe(pd.DataFrame({"Table": [x[0] for x in grouped], "Weight": weights}), width="stretch")
    st.caption("High RV values mean similar tables; the compromise emphasizes the most mutually consistent tables.")


def _hac(df: pd.DataFrame) -> None:
    st.header("Hierarchical Agglomerative Clustering")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need at least two numeric variables."):
        return
    label = st.selectbox("Object label", ["(row number)", *df.columns.astype(str).tolist()], key="hac_label")
    features = st.multiselect("Variables", numeric, default=numeric[:6], key="hac_features")
    clean = df[features].dropna()
    if not _need(len(features) >= 2 and len(clean) >= 3, "Need 2 variables and 3 complete objects."):
        return
    labels = clean.index.astype(str) if label == "(row number)" else df.loc[clean.index, label].astype(str)
    method = st.selectbox("Linkage", ["ward", "average", "complete"], key="hac_method")
    tree = linkage(StandardScaler().fit_transform(clean), method=method)
    cluster_count = st.slider("Clusters", 2, min(8, len(clean)), min(3, len(clean)), key="hac_k")
    assignments = fcluster(tree, cluster_count, criterion="maxclust")
    st.dataframe(pd.DataFrame({"Object": labels.to_numpy(), "Cluster": assignments}), width="stretch")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(tree, labels=labels.to_numpy(), leaf_rotation=90, ax=ax)
    ax.set(title="Sensory-object dendrogram", ylabel="Linkage distance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Objects joined at short distances have more similar standardized profiles.")


def _mfa(df: pd.DataFrame) -> None:
    st.header("Multiple Factor Analysis (MFA)")
    st.write("Each variable block is divided by its first singular value before a global PCA.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 3, "Need at least three numeric variables in two named blocks."):
        return
    delimiter = st.text_input("Block-name delimiter in columns", value="_", key="mfa_delimiter")
    features = st.multiselect("Variables", numeric, default=numeric[:10], key="mfa_features")
    blocks: dict[str, list[str]] = {}
    for feature in features:
        block = feature.split(delimiter, 1)[0] if delimiter and delimiter in feature else "Other"
        blocks.setdefault(block, []).append(feature)
    if not _need(len(blocks) >= 2, "Column names must identify at least two blocks (for example taste_sweet, aroma_fruity)."):
        return
    clean = df[features].dropna()
    if not _need(clean.shape[0] >= 2, "Need at least two complete rows across the selected blocks."):
        return
    pieces: list[np.ndarray] = []
    for columns in blocks.values():
        values = StandardScaler().fit_transform(clean[columns])
        first_singular = np.linalg.svd(values, compute_uv=False)[0]
        pieces.append(values / first_singular if first_singular else values)
    global_data = np.column_stack(pieces)
    pca = PCA(n_components=2).fit(global_data)
    scores = pca.transform(global_data)
    _scatter_labels(scores, clean.index, "MFA global scores",
                    f"Dimension 1 ({pca.explained_variance_ratio_[0]:.1%})",
                    f"Dimension 2 ({pca.explained_variance_ratio_[1]:.1%})")
    st.dataframe(pd.DataFrame({"Block": list(blocks), "Variables": [", ".join(v) for v in blocks.values()]}),
                 width="stretch")
    st.caption("Block scaling prevents large or high-variance blocks from dominating the global map.")


def _pls_coefficient_frame(
    model: PLSRegression,
    predictors: Sequence[str],
    responses: Sequence[str],
) -> pd.DataFrame:
    """Return PLS coefficients as predictors × responses across sklearn versions."""
    values = np.asarray(model.coef_)
    # scikit-learn >= 1.1 returns coef_ shaped (n_targets, n_features); earlier
    # versions returned (n_features, n_targets). Decide orientation from the
    # installed version instead of guessing from the (possibly square) shape,
    # since a square coefficient matrix cannot be oriented by shape alone.
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    targets_by_features = version >= (1, 1)
    expected_shape = (
        (len(responses), len(predictors))
        if targets_by_features
        else (len(predictors), len(responses))
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected PLS coefficient shape {values.shape}; expected {expected_shape} "
            f"for scikit-learn {sklearn.__version__}."
        )
    oriented = values.T if targets_by_features else values
    return pd.DataFrame(oriented, index=predictors, columns=responses)


def _pls(df: pd.DataFrame) -> None:
    st.header("Partial Least Squares (PLS) Regression")
    st.write("Models one or more sensory/consumer outcomes from correlated predictors.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need numeric predictor and response columns."):
        return
    responses = st.multiselect("Response (Y)", numeric, default=numeric[-1:], key="pls_y")
    predictors = st.multiselect("Predictors (X)", [c for c in numeric if c not in responses],
                                default=[c for c in numeric if c not in responses][:6], key="pls_x")
    clean = df[[*predictors, *responses]].dropna()
    maximum = min(len(predictors), max(1, len(clean) - 1))
    if not _need(bool(responses) and bool(predictors) and len(clean) >= 3, "Select X and Y with 3 complete rows."):
        return
    if maximum <= 1:
        components = 1
        st.caption("Using 1 component (the maximum supported by the selected predictors/rows).")
    else:
        components = st.slider("Components", 1, maximum, min(2, maximum), key="pls_components")
    model = PLSRegression(n_components=components, scale=True).fit(clean[predictors], clean[responses])
    predicted = model.predict(clean[predictors])
    rows = []
    for index, response in enumerate(responses):
        observed = clean[response].to_numpy()
        residual_sum = float(np.sum((observed - predicted[:, index]) ** 2))
        total_sum = float(np.sum((observed - observed.mean()) ** 2))
        rows.append({"Response": response, "R² (training)": 1 - residual_sum / total_sum if total_sum else np.nan,
                     "RMSE": np.sqrt(residual_sum / len(observed))})
    st.dataframe(pd.DataFrame(rows), width="stretch")
    coefficients = _pls_coefficient_frame(model, predictors, responses)
    st.dataframe(coefficients.style.background_gradient(cmap="RdBu_r", axis=None), width="stretch")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(clean[responses[0]], predicted[:, 0], alpha=0.75)
    limits = [min(clean[responses[0]].min(), predicted[:, 0].min()),
              max(clean[responses[0]].max(), predicted[:, 0].max())]
    ax.plot(limits, limits, "--", color="grey")
    ax.set(xlabel="Observed", ylabel="Predicted", title=f"PLS fit: {responses[0]}")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Larger absolute coefficients have more influence; training R² should be confirmed on new data.")

