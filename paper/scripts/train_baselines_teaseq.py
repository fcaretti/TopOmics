#!/usr/bin/env python
"""
Training script for baseline models (MultiVI, MOFA+, GLUE) on TEA-seq dataset.

This script trains multimodal integration methods for comparison:
- MultiVI: Deep generative model for RNA + ATAC
- MultiVI (linear decoder): MultiVI with linear decoders for all modalities
- scvi-tools AmortizedLDA: RNA-only, ATAC-only, and protein-only topic models
- MOFA+: Multi-Omics Factor Analysis (all modalities)
- GLUE: Graph-Linked Unified Embedding (RNA + ATAC with genomic guidance)

All models are saved for later evaluation.

Usage:
    python train_baselines_teaseq.py --n_latent 10 --max_epochs 300
    python train_baselines_teaseq.py --skip_glue  # Skip GLUE if issues arise
"""

import argparse
import inspect
import os
import warnings
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore')

# Data path
DATA_PATH = "/data/GSE158013/GSM5123951.h5mu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models on TEA-seq dataset")
    parser.add_argument(
        "--n_latent",
        type=int,
        default=10,
        help="Number of latent dimensions (default: 10)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Maximum training epochs (default: 300)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/omics_topic_models/teaseq/baselines",
        help="Output directory for models"
    )
    parser.add_argument(
        "--skip_multivi",
        action="store_true",
        help="Skip MultiVI training"
    )
    parser.add_argument(
        "--skip_multivi_linear",
        action="store_true",
        help="Skip MultiVI linear decoder training"
    )
    parser.add_argument(
        "--skip_scvi_lda",
        action="store_true",
        help="Skip scvi-tools AmortizedLDA training (RNA/ATAC/Protein)"
    )
    parser.add_argument(
        "--skip_mofa",
        action="store_true",
        help="Skip MOFA+ training"
    )
    parser.add_argument(
        "--skip_glue",
        action="store_true",
        help="Skip GLUE training"
    )
    return parser.parse_args()


def load_data():
    """Load and preprocess TEA-seq data."""
    print("Loading TEA-seq data...")
    mdata = mu.read_h5mu(DATA_PATH)

    # Binarize ATAC data
    mdata.mod['atac'].layers['counts'] = (mdata.mod['atac'].layers['counts'] > 0).astype(int)

    # Filter to highly variable genes for RNA
    sc.pp.highly_variable_genes(mdata.mod['rna'], n_top_genes=2000, flavor='seurat_v3', layer='counts')
    mdata.mod['rna'] = mdata.mod['rna'][:, mdata.mod['rna'].var['highly_variable']].copy()

    # Filter to highly variable peaks for ATAC
    sc.pp.highly_variable_genes(mdata.mod['atac'], n_top_genes=10000, flavor='seurat_v3', layer='counts')
    mdata.mod['atac'] = mdata.mod['atac'][:, mdata.mod['atac'].var['highly_variable']].copy()

    # Sync MuData axes after feature filtering
    mdata.update()

    print(f"  RNA: {mdata.mod['rna'].shape}")
    print(f"  ATAC: {mdata.mod['atac'].shape}")
    print(f"  Protein: {mdata.mod['prot'].shape}")

    return mdata


def _history_to_df(history):
    if isinstance(history, pd.DataFrame):
        return history
    if history is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(history)
    except Exception:
        if not isinstance(history, dict):
            return pd.DataFrame([history])
        fixed = {}
        max_len = 0
        for key, value in history.items():
            if np.isscalar(value) or (hasattr(value, "shape") and np.ndim(value) == 0):
                values = [value]
            else:
                try:
                    values = list(value)
                except TypeError:
                    values = [value]
            fixed[key] = values
            max_len = max(max_len, len(values))
        for key, values in fixed.items():
            if len(values) < max_len:
                fixed[key] = values + [np.nan] * (max_len - len(values))
        return pd.DataFrame(fixed)


def _save_history(history, path):
    history_df = _history_to_df(history)
    history_df.to_csv(path, index=False)


def _infer_decoder_input_dim(decoder, n_latent):
    for attr in ("n_input", "n_latent", "n_in"):
        if hasattr(decoder, attr):
            return getattr(decoder, attr)
    for attr in ("px_decoder", "py_decoder"):
        module = getattr(decoder, attr, None)
        if module is None:
            continue
        for name in ("n_in", "in_features"):
            if hasattr(module, name):
                return getattr(module, name)
    return n_latent


def _infer_decoder_output_dim(decoder):
    for attr in ("n_output", "n_genes", "n_regions", "n_proteins"):
        if hasattr(decoder, attr):
            return getattr(decoder, attr)
    for attr in ("px_scale_decoder", "px_rate_decoder", "px_decoder"):
        module = getattr(decoder, attr, None)
        if module is None:
            continue
        for name in ("out_features", "n_out"):
            if hasattr(module, name):
                return getattr(module, name)
    return None


def _build_linear_decoder(decoder_cls, base_decoder, n_latent, n_output_override=None):
    import torch.nn as nn

    n_input = _infer_decoder_input_dim(base_decoder, n_latent)
    n_output = n_output_override
    if n_output is None:
        n_output = _infer_decoder_output_dim(base_decoder)
    if n_output is None:
        raise ValueError(
            f"Could not infer n_output for {decoder_cls.__name__}."
        )
    n_cat_list = getattr(base_decoder, "n_cat_list", []) or []
    n_hidden = getattr(base_decoder, "n_hidden", None) or n_input

    overrides = {
        "n_layers": 1,
        "n_hidden": n_hidden,
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "use_activation": False,
        "activation_fn": nn.Identity(),
    }

    sig = inspect.signature(decoder_cls.__init__)
    init_kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in overrides:
            if overrides[name] is not None:
                init_kwargs[name] = overrides[name]
            continue
        if name == "n_input":
            init_kwargs[name] = n_input
            continue
        if name == "n_output":
            init_kwargs[name] = n_output
            continue
        if name == "n_cat_list":
            init_kwargs[name] = n_cat_list
            continue
        if hasattr(base_decoder, name):
            init_kwargs[name] = getattr(base_decoder, name)
            continue
        if param.default is not inspect._empty:
            init_kwargs[name] = param.default

    missing = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            continue
        if name not in init_kwargs or init_kwargs[name] is None:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Missing required args for {decoder_cls.__name__}: {missing}"
        )

    return decoder_cls(**init_kwargs)


def _linearize_multivi_decoders(model, n_latent, output_dims=None):
    import scvi.nn as scvi_nn

    module = model.module
    decoder_attrs = [
        "z_decoder_expression",
        "z_decoder_accessibility",
        "z_decoder_protein",
        "z_decoder_pro",
    ]

    decoders = []
    for attr in decoder_attrs:
        if hasattr(module, attr):
            decoders.append((attr, getattr(module, attr)))

    if not decoders:
        for name, submod in module.named_children():
            if submod.__class__.__name__.startswith("Decoder"):
                decoders.append((name, submod))

    if not decoders:
        raise ValueError("Could not locate MultiVI decoders to linearize.")

    def _pick_output_dim(decoder_name):
        if not output_dims:
            return None
        name = decoder_name.lower()
        if "expression" in name or "rna" in name:
            return output_dims.get("expression") or output_dims.get("rna")
        if "accessibility" in name or "atac" in name or "peak" in name:
            return output_dims.get("accessibility") or output_dims.get("atac")
        if "protein" in name or name.endswith("_pro") or "adt" in name:
            return output_dims.get("protein") or output_dims.get("prot")
        return None

    replaced = []
    for name, base_decoder in decoders:
        decoder_cls = base_decoder.__class__
        if (
            base_decoder.__class__.__name__ == "DecoderSCVI"
            and hasattr(scvi_nn, "LinearDecoderSCVI")
        ):
            decoder_cls = scvi_nn.LinearDecoderSCVI
        n_output_override = _pick_output_dim(name)
        new_decoder = _build_linear_decoder(
            decoder_cls,
            base_decoder,
            n_latent,
            n_output_override=n_output_override,
        )
        setattr(module, name, new_decoder)
        replaced.append(name)

    return replaced


# =============================================================================
# MultiVI (RNA + ATAC)
# =============================================================================
def train_multivi(mdata, n_latent, max_epochs, output_dir):
    """Train MultiVI model on RNA + ATAC."""
    import scvi

    print("\n" + "=" * 70)
    print("Training MultiVI (RNA + ATAC)")
    print("=" * 70)

    # Create a copy for MultiVI
    mdata_multivi = mdata.copy()

    # MultiVI needs counts in .X
    mdata_multivi.mod['rna'].X = mdata_multivi.mod['rna'].layers['counts'].copy()
    mdata_multivi.mod['atac'].X = mdata_multivi.mod['atac'].layers['counts'].copy()

    # Setup MultiVI for RNA + ATAC
    scvi.model.MULTIVI.setup_mudata(
        mdata_multivi,
        rna_layer=None,
        atac_layer=None,
        batch_key=None,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        }
    )

    # Create model
    model = scvi.model.MULTIVI(
        mdata_multivi,
        n_latent=n_latent,
        n_hidden=128,
    )

    # Train
    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
        early_stopping=True,
    )

    # Get latent representation
    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")

    # Save model
    model_path = os.path.join(output_dir, "multivi")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    # Save latent representation
    np.save(os.path.join(output_dir, "latent_multivi.npy"), latent)

    # Save training history
    _save_history(model.history, os.path.join(output_dir, "multivi_history.csv"))

    return latent, mdata_multivi


# =============================================================================
# MultiVI (linear decoder)
# =============================================================================
def train_multivi_linear(mdata, n_latent, max_epochs, output_dir):
    """Train MultiVI model with linear decoders for all modalities."""
    import scvi

    print("\n" + "=" * 70)
    print("Training MultiVI Linear (RNA + ATAC)")
    print("=" * 70)

    mdata_multivi = mdata.copy()
    mdata_multivi.mod["rna"].X = mdata_multivi.mod["rna"].layers["counts"].copy()
    mdata_multivi.mod["atac"].X = mdata_multivi.mod["atac"].layers["counts"].copy()

    scvi.model.MULTIVI.setup_mudata(
        mdata_multivi,
        rna_layer=None,
        atac_layer=None,
        batch_key=None,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )

    model = scvi.model.MULTIVI(
        mdata_multivi,
        n_latent=n_latent,
        n_hidden=128,
    )
    output_dims = {
        "expression": mdata_multivi.mod["rna"].n_vars,
        "accessibility": mdata_multivi.mod["atac"].n_vars,
    }
    if "prot" in mdata_multivi.mod:
        output_dims["protein"] = mdata_multivi.mod["prot"].n_vars
    replaced = _linearize_multivi_decoders(model, n_latent, output_dims=output_dims)
    try:
        model.module.to(model.device)
    except Exception:
        pass
    if replaced:
        print(f"Linearized decoders: {', '.join(replaced)}")

    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
        early_stopping=True,
    )

    latent = model.get_latent_representation()
    print(f"Latent shape: {latent.shape}")

    model_path = os.path.join(output_dir, "multivi_linear")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    np.save(os.path.join(output_dir, "latent_multivi_linear.npy"), latent)
    _save_history(model.history, os.path.join(output_dir, "multivi_linear_history.csv"))

    return latent, mdata_multivi


# =============================================================================
# scvi-tools AmortizedLDA (unimodal)
# =============================================================================
def train_scvi_amortized_lda(adata, n_topics, max_epochs, output_dir, modality_name):
    import scvi

    print("\n" + "=" * 70)
    print(f"Training AmortizedLDA ({modality_name})")
    print("=" * 70)

    adata_lda = adata.copy()
    if "counts" in adata_lda.layers:
        adata_lda.X = adata_lda.layers["counts"].copy()

    scvi.model.AmortizedLDA.setup_anndata(adata_lda, layer=None)
    model = scvi.model.AmortizedLDA(
        adata_lda,
        n_topics=n_topics,
        n_hidden=128,
        cell_topic_prior=1 / n_topics,
    )

    print("Training...")
    model.train(
        max_epochs=max_epochs,
        train_size=0.8,
        validation_size=0.2,
        batch_size=128,
    )

    theta = model.get_latent_representation()
    latent = np.asarray(theta)
    print(f"Latent shape: {latent.shape}")

    model_path = os.path.join(output_dir, f"amortized_lda_{modality_name}")
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path, overwrite=True)
    print(f"Model saved to: {model_path}")

    np.save(os.path.join(output_dir, f"latent_amortized_lda_{modality_name}.npy"), latent)
    _save_history(model.history, os.path.join(output_dir, f"amortized_lda_{modality_name}_history.csv"))

    return latent, adata_lda
# =============================================================================
# MOFA+
# =============================================================================
def train_mofa(mdata, n_latent, output_dir):
    """Train MOFA+ model."""
    print("\n" + "=" * 70)
    print("Training MOFA+ (RNA + ATAC + Protein)")
    print("=" * 70)

    # Create a copy for MOFA
    mdata_mofa = mdata.copy()

    # MOFA needs normalized data
    # RNA: log-normalize and scale
    mdata_mofa.mod['rna'].X = mdata_mofa.mod['rna'].layers['counts'].copy()
    sc.pp.normalize_total(mdata_mofa.mod['rna'], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod['rna'])
    sc.pp.scale(mdata_mofa.mod['rna'])

    # ATAC: normalize and scale (already binarized)
    mdata_mofa.mod['atac'].X = mdata_mofa.mod['atac'].layers['counts'].copy()
    sc.pp.normalize_total(mdata_mofa.mod['atac'], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod['atac'])
    sc.pp.scale(mdata_mofa.mod['atac'])

    # Protein: log-normalize and scale
    mdata_mofa.mod['prot'].X = mdata_mofa.mod['prot'].layers['counts'].copy()
    sc.pp.normalize_total(mdata_mofa.mod['prot'], target_sum=1e4)
    sc.pp.log1p(mdata_mofa.mod['prot'])
    sc.pp.scale(mdata_mofa.mod['prot'])

    # Train MOFA+
    print("Training...")
    mu.tl.mofa(
        mdata_mofa,
        n_factors=n_latent,
        convergence_mode='medium',
        use_obs='intersection',
    )

    # Get latent representation
    latent = mdata_mofa.obsm['X_mofa']
    print(f"Latent shape: {latent.shape}")

    # Save MuData with MOFA results
    mdata_path = os.path.join(output_dir, "mdata_mofa.h5mu")
    mdata_mofa.write(mdata_path)
    print(f"MuData saved to: {mdata_path}")

    # Save latent representation
    np.save(os.path.join(output_dir, "latent_mofa.npy"), latent)

    return latent, mdata_mofa


# =============================================================================
# GLUE (RNA + ATAC with genomic guidance)
# =============================================================================
def train_glue(mdata, n_latent, max_epochs, output_dir):
    """Train GLUE model on RNA + ATAC."""
    import scglue

    print("\n" + "=" * 70)
    print("Training GLUE (RNA + ATAC)")
    print("=" * 70)

    # GLUE requires separate AnnData objects and genomic coordinates
    rna = mdata.mod['rna'].copy()
    atac = mdata.mod['atac'].copy()

    # Use counts
    rna.X = rna.layers['counts'].copy()
    atac.X = atac.layers['counts'].copy()

    # Check if genomic coordinates are available
    has_coords = 'chromStart' in atac.var.columns or 'start' in atac.var.columns

    if not has_coords:
        print("ATAC peaks lack genomic coordinates.")
        print("Attempting to parse from var_names (format: chr:start-end or chr_start_end)...")

        # Try to parse coordinates from var_names
        try:
            # Try chr:start-end format first
            coords = atac.var_names.str.extract(r'(chr[^:_]+)[:\-_](\d+)[:\-_](\d+)')
            if coords.isna().any().any():
                raise ValueError("Could not parse coordinates")
            atac.var['chrom'] = coords[0].values
            atac.var['chromStart'] = coords[1].astype(int).values
            atac.var['chromEnd'] = coords[2].astype(int).values
            has_coords = True
            print("Successfully parsed coordinates from var_names")
        except Exception as e:
            print(f"Failed to parse coordinates: {e}")
            print("GLUE training skipped - genomic coordinates required")
            return None, None

    # Preprocess for GLUE
    print("Preprocessing for GLUE...")

    # RNA preprocessing
    rna.layers['counts'] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, use_highly_variable=False)

    # ATAC preprocessing with LSI
    atac.layers['counts'] = atac.X.copy()
    scglue.data.lsi(atac, n_components=100)

    # Build guidance graph
    print("Building guidance graph...")

    # Check if RNA has gene coordinates
    if 'chromStart' not in rna.var.columns:
        print("RNA genes lack genomic coordinates.")
        print("Attempting to get gene coordinates from Ensembl/UCSC...")

        # Try to add gene coordinates using scglue utility
        try:
            scglue.data.get_gene_annotation(
                rna,
                gtf="http://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
                gtf_by="gene_name"
            )
            print("Successfully added gene coordinates")
        except Exception as e:
            print(f"Could not get gene coordinates: {e}")
            print("Using correlation-based guidance graph instead...")

    # Build guidance graph
    try:
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        print(f"Guidance graph: {guidance.number_of_nodes()} nodes, {guidance.number_of_edges()} edges")
    except Exception as e:
        print(f"Could not build genomic guidance graph: {e}")
        print("GLUE training skipped")
        return None, None

    # Configure datasets
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=False,
        use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=False,
        use_layer="counts", use_rep="X_lsi"
    )

    # Train GLUE
    print("Training GLUE model...")
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance,
        fit_kws={"directory": os.path.join(output_dir, "glue_checkpoints")},
    )

    # Get latent representations
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    # Use RNA latent as the combined representation
    latent = rna.obsm["X_glue"]
    print(f"Latent shape: {latent.shape}")

    # Save model
    model_path = os.path.join(output_dir, "glue_model")
    os.makedirs(model_path, exist_ok=True)
    glue.save(os.path.join(model_path, "glue.dill"))
    print(f"Model saved to: {model_path}")

    # Save latent representation
    np.save(os.path.join(output_dir, "latent_glue.npy"), latent)

    # Save processed AnnData objects
    rna.write(os.path.join(output_dir, "rna_glue.h5ad"))
    atac.write(os.path.join(output_dir, "atac_glue.h5ad"))

    return latent, rna


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("TEA-seq Baseline Models Training")
    print("=" * 70)
    print(f"N latent dimensions: {args.n_latent}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load data
    mdata = load_data()

    results = {}

    # Train MultiVI (RNA + ATAC)
    if not args.skip_multivi:
        try:
            latent, _ = train_multivi(
                mdata, args.n_latent, args.max_epochs, args.output_dir
            )
            results['multivi'] = latent
        except Exception as e:
            print(f"MultiVI training failed: {e}")
            import traceback
            traceback.print_exc()

    # Train MultiVI Linear (RNA + ATAC)
    if not args.skip_multivi_linear:
        try:
            latent, _ = train_multivi_linear(
                mdata, args.n_latent, args.max_epochs, args.output_dir
            )
            results['multivi_linear'] = latent
        except Exception as e:
            print(f"MultiVI linear training failed: {e}")
            import traceback
            traceback.print_exc()

    # Train scvi-tools AmortizedLDA (RNA/ATAC/Protein)
    if not args.skip_scvi_lda:
        try:
            latent, _ = train_scvi_amortized_lda(
                mdata.mod["rna"], args.n_latent, args.max_epochs, args.output_dir, "rna"
            )
            results["amortized_lda_rna"] = latent
        except Exception as e:
            print(f"AmortizedLDA (RNA) failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            latent, _ = train_scvi_amortized_lda(
                mdata.mod["atac"], args.n_latent, args.max_epochs, args.output_dir, "atac"
            )
            results["amortized_lda_atac"] = latent
        except Exception as e:
            print(f"AmortizedLDA (ATAC) failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            latent, _ = train_scvi_amortized_lda(
                mdata.mod["prot"], args.n_latent, args.max_epochs, args.output_dir, "prot"
            )
            results["amortized_lda_prot"] = latent
        except Exception as e:
            print(f"AmortizedLDA (Protein) failed: {e}")
            import traceback
            traceback.print_exc()

    # Train MOFA+ (all modalities)
    if not args.skip_mofa:
        try:
            latent, _ = train_mofa(mdata, args.n_latent, args.output_dir)
            results['mofa'] = latent
        except Exception as e:
            print(f"MOFA+ training failed: {e}")
            import traceback
            traceback.print_exc()

    # Train GLUE (RNA + ATAC)
    if not args.skip_glue:
        try:
            latent, _ = train_glue(
                mdata, args.n_latent, args.max_epochs, args.output_dir
            )
            if latent is not None:
                results['glue'] = latent
        except Exception as e:
            print(f"GLUE training failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for name, latent in results.items():
        print(f"  {name}: latent shape = {latent.shape}")
    print(f"\nAll results saved to: {args.output_dir}")

    # Save summary
    summary = {
        'model': list(results.keys()),
        'n_latent': [r.shape[1] for r in results.values()],
        'n_cells': [r.shape[0] for r in results.values()],
    }
    pd.DataFrame(summary).to_csv(
        os.path.join(args.output_dir, "training_summary.csv"), index=False
    )


if __name__ == "__main__":
    main()
