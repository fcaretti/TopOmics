#!/usr/bin/env python
"""
Annotate TEA-seq RNA modality with CellTypist.

Example:
    python annotate_teaseq_celltypist.py --model Immune_All_Low --download-model
"""

import argparse
import os
from pathlib import Path

import celltypist
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from celltypist import models


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate TEA-seq with CellTypist")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/GSE158013/GSM5123951.h5mu",
        help="Path to input MuData (.h5mu)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write annotated MuData (default: <data_path>_celltypist.h5mu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Immune_All_Low",
        help="CellTypist model name (or path if --model-path not used)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional local path to a CellTypist model .pkl",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the specified model before annotation",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/.celltypist",
        help="CellTypist home or models directory",
    )
    parser.add_argument(
        "--use-counts-layer",
        action="store_true",
        help="Use RNA counts layer as .X before annotation",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply CellTypist normalization (or scanpy fallback) before annotation",
    )
    parser.add_argument(
        "--majority-voting",
        action="store_true",
        help="Enable majority voting in CellTypist",
    )
    parser.add_argument(
        "--no-majority-voting",
        dest="majority_voting",
        action="store_false",
        help="Disable majority voting in CellTypist",
    )
    parser.set_defaults(majority_voting=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="best match",
        help="CellTypist annotation mode (best match or prob match)",
    )
    parser.add_argument(
        "--save-probabilities",
        type=str,
        default=None,
        help="Optional path to save probability matrix as CSV",
    )
    return parser.parse_args()


def resolve_output_path(data_path, output_path):
    if output_path:
        return Path(output_path)
    data_path = Path(data_path)
    return data_path.with_name(f"{data_path.stem}_celltypist.h5mu")


def maybe_download_model(model_name):
    if hasattr(models, "download_model"):
        models.download_model(model_name)
    else:
        models.download_models()


def load_model(args, model_dir):
    if args.model_path:
        return models.Model.load(args.model_path)

    model_name = args.model
    model_path = Path(model_name)
    if model_path.exists():
        return models.Model.load(str(model_path))

    if model_dir:
        candidate = Path(model_dir) / model_name
        if candidate.suffix != ".pkl":
            candidate = candidate.with_suffix(".pkl")
        if candidate.exists():
            return models.Model.load(str(candidate))

    fallback = Path.home() / ".celltypist" / "data" / "models" / f"{model_name}.pkl"
    if fallback.exists():
        return models.Model.load(str(fallback))

    return models.Model.load(model_name)


def resolve_model_paths(model_dir):
    base = Path(model_dir)
    if (base / "models.json").exists():
        model_path = base
        if base.name == "models" and base.parent.name == "data":
            home = base.parent.parent
        else:
            home = base
        return home, model_path
    candidate = base / "data" / "models"
    return base, candidate


def apply_normalization(adata):
    if hasattr(celltypist, "normalize"):
        celltypist.normalize(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)


def extract_labels(result):
    labels = result.predicted_labels
    if isinstance(labels, pd.DataFrame):
        if "majority_voting" in labels.columns:
            return labels["majority_voting"].values
        if "predicted_labels" in labels.columns:
            return labels["predicted_labels"].values
        return labels.iloc[:, 0].values
    return labels


def normalize_mode(mode):
    mode_norm = mode.strip().lower().replace("_", " ").replace("-", " ")
    if mode_norm in {"best", "best match", "bestmatch"}:
        return "best match"
    if mode_norm in {"prob", "probability", "prob match", "probmatch"}:
        return "prob match"
    raise ValueError("Invalid mode. Use 'best match' or 'prob match'.")


def main():
    args = parse_args()

    home_dir, model_path = resolve_model_paths(args.model_dir)
    os.environ["CELLTYPIST_HOME"] = str(home_dir)
    if hasattr(models, "model_path"):
        models.model_path = str(model_path)

    print("Loading MuData...")
    mdata = mu.read_h5mu(args.data_path)
    adata = mdata.mod["rna"].copy()

    if args.use_counts_layer and "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    adata.var_names_make_unique()

    if args.normalize:
        print("Normalizing RNA for CellTypist...")
        apply_normalization(adata)

    if args.download_model:
        print(f"Downloading CellTypist model: {args.model}")
        maybe_download_model(args.model)

    print("Loading CellTypist model...")
    model = load_model(args, model_path)

    print("Annotating cells...")
    mode = normalize_mode(args.mode)

    result = celltypist.annotate(
        adata,
        model=model,
        majority_voting=args.majority_voting,
        mode=mode,
    )

    labels = extract_labels(result)
    adata.obs["celltypist_label"] = labels

    if getattr(result, "probability_matrix", None) is not None:
        prob = result.probability_matrix.max(axis=1)
        adata.obs["celltypist_confidence"] = np.asarray(prob)
        if args.save_probabilities:
            result.probability_matrix.to_csv(args.save_probabilities)

    mdata.mod["rna"].obs["celltypist_label"] = adata.obs["celltypist_label"]
    if "celltypist_confidence" in adata.obs:
        mdata.mod["rna"].obs["celltypist_confidence"] = adata.obs["celltypist_confidence"]

    mdata.obs["celltypist_label"] = adata.obs["celltypist_label"]
    if "celltypist_confidence" in adata.obs:
        mdata.obs["celltypist_confidence"] = adata.obs["celltypist_confidence"]

    output_path = resolve_output_path(args.data_path, args.output_path)
    print(f"Writing annotated MuData to: {output_path}")
    mdata.write(output_path)


if __name__ == "__main__":
    main()
