"""
Generate individual figure files for the retina batch correction comparison.

Reproduces all plots from retina_batch_correction_comparison.ipynb,
saving each image as a separate file.

Output directory: /data/omics_topic_models/figures/retina/
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("/data/retina_dataset")
FIG_DIR = Path("/data/omics_topic_models/figures/retina")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sc.set_figure_params(dpi=100, frameon=False, figsize=(8, 6))
sns.set_style("whitegrid")

SAVE_KW = dict(dpi=300, bbox_inches="tight")

# Pretty display names for the methods
DISPLAY_NAMES = {
    "X_omics_topic_encode_True": "omics-topic (encode=True)",
    "X_omics_topic_encode_False": "omics-topic (encode=False)",
    "X_scvi": "scVI",
    "X_linear_scvi": "LinearSCVI",
    "X_omics_topic_no_batch": "omics-topic (no batch)",
    "X_scvi_no_batch": "scVI (no batch)",
    "X_linear_scvi_no_batch": "LinearSCVI (no batch)",
}


def fname_safe(name: str) -> str:
    """Turn a representation key into a filesystem-safe string."""
    return name.lower().replace("x_", "").replace(" ", "_")


# ---------------------------------------------------------------------------
# Helpers (same as notebook)
# ---------------------------------------------------------------------------
def is_topic_representation(name, rep, tol=1e-3):
    x = np.asarray(rep)
    name_flag = any(k in name.lower() for k in ["topic", "lda", "etm"])
    simplex_flag = (
        x.ndim == 2
        and np.nanmin(x) >= -1e-8
        and np.allclose(x.sum(axis=1), 1.0, atol=tol)
    )
    return name_flag or simplex_flag


def prepare_representation(name, rep):
    x = np.asarray(rep)
    if is_topic_representation(name, x):
        return np.sqrt(np.clip(x, a_min=0.0, a_max=None))
    return x


# ---------------------------------------------------------------------------
# 1. Load data and representations
# ---------------------------------------------------------------------------
print("Loading data …")
adata = sc.read_h5ad(DATA_DIR / "retina_preprocessed.h5ad")
print(f"Data shape: {adata.shape}")

rep_sources = {
    "X_omics_topic_encode_True": "retina_with_omics_topic.h5ad",
    "X_omics_topic_encode_False": "retina_with_omics_topic.h5ad",
    "X_scvi": "retina_with_scvi.h5ad",
    "X_linear_scvi": "retina_with_linear_scvi.h5ad",
    "X_omics_topic_no_batch": "retina_with_omics_topic_no_batch.h5ad",
    "X_scvi_no_batch": "retina_with_scvi_no_batch.h5ad",
    "X_linear_scvi_no_batch": "retina_with_linear_scvi_no_batch.h5ad",
}

representations = {}
for key, fname in rep_sources.items():
    fpath = DATA_DIR / fname
    if fpath.exists():
        tmp_adata = sc.read_h5ad(fpath)
        if key in tmp_adata.obsm:
            representations[key] = tmp_adata.obsm[key]
            print(f"  Loaded {key}: shape {representations[key].shape}")
        else:
            print(f"  Warning: {key} not found in {fname}")
    else:
        print(f"  Warning: {fname} not found, skipping {key}")

print(f"\nLoaded {len(representations)} representations")

# ---------------------------------------------------------------------------
# 2. Compute neighbours + UMAP for every representation
# ---------------------------------------------------------------------------
print("\nComputing UMAPs …")
for name, rep in representations.items():
    print(f"  {name} …")
    tmp = adata.copy()
    tmp.obsm["X_latent"] = prepare_representation(name, rep)
    sc.pp.neighbors(tmp, use_rep="X_latent", n_neighbors=15)
    sc.tl.umap(tmp)
    adata.obsm[f"X_umap_{name}"] = tmp.obsm["X_umap"]

# ---------------------------------------------------------------------------
# 3. Individual UMAP figures (batch + cell type per method)
# ---------------------------------------------------------------------------
print("\nSaving UMAP figures …")
for name in representations:
    tmp = adata.copy()
    tmp.obsm["X_umap"] = adata.obsm[f"X_umap_{name}"]
    safe = fname_safe(name)
    display = DISPLAY_NAMES.get(name, name)

    # -- batch --
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(tmp, color="batch", ax=ax, show=False, title=f"{display} — Batch")
    fig.savefig(FIG_DIR / f"umap_{safe}_batch.png", **SAVE_KW)
    plt.close(fig)

    # -- cell type --
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        tmp, color="labels", ax=ax, show=False,
        title=f"{display} — Cell Type", legend_loc="right margin",
    )
    fig.savefig(FIG_DIR / f"umap_{safe}_celltype.png", **SAVE_KW)
    plt.close(fig)

print(f"  Saved {2 * len(representations)} UMAP images")

# ---------------------------------------------------------------------------
# 4. Batch mixing metrics
# ---------------------------------------------------------------------------
print("\nComputing batch mixing metrics …")
batch_metrics = []
for name, rep in representations.items():
    metrics = {"method": name}
    rep_eval = prepare_representation(name, rep)

    tmp = adata.copy()
    tmp.obsm["X_latent"] = rep_eval
    sc.pp.neighbors(tmp, use_rep="X_latent", n_neighbors=15)

    try:
        metrics["batch_silhouette"] = silhouette_score(rep_eval, adata.obs["batch"])
        metrics["celltype_silhouette"] = silhouette_score(rep_eval, adata.obs["labels"])
    except Exception as e:
        print(f"  Silhouette error for {name}: {e}")

    try:
        from scib.metrics import ilisi_graph, clisi_graph
        metrics["iLISI"] = ilisi_graph(tmp, batch_key="batch", type_="embed", use_rep="X_latent")
        metrics["cLISI"] = clisi_graph(tmp, label_key="labels", type_="embed", use_rep="X_latent")
    except Exception:
        pass  # scib LISI not available on this system

    batch_metrics.append(metrics)

batch_metrics_df = pd.DataFrame(batch_metrics)

# ---------------------------------------------------------------------------
# 5. Classification metrics
# ---------------------------------------------------------------------------
print("Computing classification metrics …")
classification_metrics = []
for name, rep in representations.items():
    X = prepare_representation(name, rep)
    y_cell = adata.obs["labels"].values
    y_batch = adata.obs["batch"].values

    # Cell-type classification
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cell, test_size=0.3, random_state=42, stratify=y_cell,
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cv = cross_val_score(clf, X, y_cell, cv=5, n_jobs=-1)

    # Batch classification
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X, y_batch, test_size=0.3, random_state=42, stratify=y_batch,
    )
    clf_b = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_b.fit(X_tr_b, y_tr_b)
    y_pred_b = clf_b.predict(X_te_b)
    cv_b = cross_val_score(clf_b, X, y_batch, cv=5, n_jobs=-1)

    classification_metrics.append({
        "method": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
        "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
        "cv_accuracy_mean": cv.mean(),
        "cv_accuracy_std": cv.std(),
        "batch_accuracy": accuracy_score(y_te_b, y_pred_b),
        "batch_balanced_accuracy": balanced_accuracy_score(y_te_b, y_pred_b),
        "batch_cv_accuracy_mean": cv_b.mean(),
        "batch_cv_accuracy_std": cv_b.std(),
    })

classification_df = pd.DataFrame(classification_metrics)
summary_df = batch_metrics_df.merge(classification_df, on="method")

# Save CSV
summary_df.to_csv(FIG_DIR / "retina_comparison_metrics.csv", index=False)
print(f"  Metrics CSV saved to {FIG_DIR / 'retina_comparison_metrics.csv'}")

# ---------------------------------------------------------------------------
# 6. Individual metric bar-chart figures
# ---------------------------------------------------------------------------
print("\nSaving metric bar charts …")
metrics_to_plot = [
    ("batch_silhouette", "Batch Silhouette (lower is better)", False),
    ("celltype_silhouette", "Cell Type Silhouette (higher is better)", True),
    ("iLISI", "iLISI (higher is better)", True),
    ("cLISI", "cLISI (higher is better)", True),
    ("accuracy", "Cell Type Accuracy (higher is better)", True),
    ("f1_macro", "Cell Type F1 macro (higher is better)", True),
    ("batch_accuracy", "Batch Classifier Accuracy (lower is better)", False),
    ("batch_cv_accuracy_mean", "Batch Classifier CV Accuracy (lower is better)", False),
]

n_saved = 0
for metric, title, higher_better in metrics_to_plot:
    if metric not in summary_df.columns:
        print(f"  Skipping {metric} (not available)")
        continue

    data = summary_df.sort_values(metric, ascending=not higher_better)
    colors = sns.color_palette(
        "RdYlGn" if higher_better else "RdYlGn_r", n_colors=len(data),
    )
    labels = [DISPLAY_NAMES.get(m, m) for m in data["method"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(data)), data[metric], color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"metric_{metric}.png", **SAVE_KW)
    plt.close(fig)
    n_saved += 1

print(f"  Saved {n_saved} metric bar charts")

# ---------------------------------------------------------------------------
# 7. Individual batch-mixing UMAP figures
# ---------------------------------------------------------------------------
print("\nSaving batch mixing figures …")
for name in representations:
    umap_coords = adata.obsm[f"X_umap_{name}"]
    safe = fname_safe(name)
    display = DISPLAY_NAMES.get(name, name)

    batch_col = adata.obs["batch"]
    batch_vals = sorted(batch_col.unique())
    palette = {"0.0": "tab:blue", "1.0": "tab:orange"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for bval in batch_vals:
        mask = batch_col == bval
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=palette.get(str(bval), None), s=1, alpha=0.4,
            rasterized=True, label=f"Batch {bval}",
        )
    ax.set_title(f"{display} — Batch Mixing")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"batch_mixing_{safe}.png", **SAVE_KW)
    plt.close(fig)

print(f"  Saved {len(representations)} batch mixing images")

# ---------------------------------------------------------------------------
print(f"\nDone! All figures saved to {FIG_DIR}")
