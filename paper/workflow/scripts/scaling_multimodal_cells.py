"""
Scaling benchmark: multimodal cell-count sweep on synthetic RNA+ATAC.

Generates synthetic data from a SHARE-Topic model fitted on lymphoma B-cells,
then benchmarks MultiVI, TopOmics, and MOFA+ for increasing cell counts.
"""
import json
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

N_HVG = 2000
N_PEAKS = 20000
MOFA_MAX_CELLS = 10000
SHARE_TOPIC_INITIAL_BURNIN = 500
SHARE_TOPIC_N_SAMPLES = 200


def fit_share_topic(rna_path, atac_path, model_path, n_topics):
    import anndata as ad
    from topomics.models.share_topic import ShareTopic_LDA_Multi

    print("Fitting SHARE-Topic on lymphoma data...")
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)

    model = ShareTopic_LDA_Multi({"rna": rna, "chromatin": atac}, n_topics=n_topics)
    model.fit(
        n_samples=SHARE_TOPIC_N_SAMPLES,
        initial_burnin=SHARE_TOPIC_INITIAL_BURNIN,
        auto_burnin=True,
    )
    model.save(model_path)
    print(f"  Model saved to {model_path}")
    return model


def generate_synthetic(model_path, rna_orig_path, atac_orig_path, out_dir, n_cells, seed):
    import anndata as ad
    import torch
    from topomics.models.share_topic import ShareTopic_LDA_Multi

    rna_path = os.path.join(out_dir, f"synth_rna_{n_cells}.h5ad")
    atac_path = os.path.join(out_dir, f"synth_atac_{n_cells}.h5ad")

    if os.path.exists(rna_path) and os.path.exists(atac_path):
        return rna_path, atac_path

    rna_orig = ad.read_h5ad(rna_orig_path)
    atac_orig = ad.read_h5ad(atac_orig_path)
    model = ShareTopic_LDA_Multi.load(
        model_path, {"rna": rna_orig, "chromatin": atac_orig}, device="cpu",
    )

    torch.manual_seed(seed)
    generated = model.generate(n_cells)

    generated["rna"].write_h5ad(rna_path)
    generated["chromatin"].write_h5ad(atac_path)
    print(f"  Generated {n_cells} cells -> {rna_path}")
    return rna_path, atac_path


def _select_top_variable(adata, n_top):
    X = adata.X
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
    else:
        var = np.var(X, axis=0)
    n_top = min(n_top, len(var))
    return adata[:, np.argsort(-var)[:n_top]].copy()


def load_synthetic_mudata(rna_path, atac_path):
    import anndata as ad
    import mudata as mu
    import scanpy as sc

    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)

    sc.pp.filter_genes(rna, min_cells=1)
    sc.pp.filter_genes(atac, min_cells=1)

    sc.pp.highly_variable_genes(rna, n_top_genes=min(N_HVG, rna.n_vars),
                                flavor="seurat_v3", subset=True)
    atac = _select_top_variable(atac, N_PEAKS)

    return mu.MuData({"rna": rna, "atac": atac})


def train_multivi(mdata, n_topics, max_epochs, batch_size):
    import scvi
    scvi.model.MULTIVI.setup_mudata(
        mdata, rna_layer=None, atac_layer=None,
        modalities={"rna_layer": "rna", "atac_layer": "atac"},
    )
    model = scvi.model.MULTIVI(mdata, n_latent=n_topics)
    t0 = time.perf_counter()
    model.train(max_epochs=max_epochs, batch_size=batch_size, early_stopping=True)
    elapsed = time.perf_counter() - t0
    n_epochs = model.history["train_loss_epoch"].shape[0]
    elbo = float(model.get_elbo())
    return {"time": elapsed, "epochs": n_epochs, "elbo": elbo}


def train_omics_topic(mdata, n_topics, max_epochs, batch_size):
    from topomics.models import MultimodalAmortizedLDA
    model = MultimodalAmortizedLDA.from_mudata(
        mdata, layer_dict={"rna": None, "atac": None},
        n_topics=n_topics, likelihoods=["gamma_poisson", "bernoulli"],
    )
    t0 = time.perf_counter()
    model.train(
        max_epochs=max_epochs, batch_size=batch_size,
        early_stopping=True, early_stopping_monitor="elbo_val",
        early_stopping_patience=50, early_stopping_min_delta=0.0,
    )
    elapsed = time.perf_counter() - t0
    n_epochs = len(model.history["elbo_train"])
    elbo = float(model.get_elbo())
    return {"time": elapsed, "epochs": n_epochs, "elbo": elbo}


def train_mofa(mdata, n_topics, seed):
    import muon as mu
    import scanpy as sc

    mdata_mofa = mdata.copy()
    for mod_name in ["rna", "atac"]:
        sc.pp.normalize_total(mdata_mofa.mod[mod_name], target_sum=1e4)
        sc.pp.log1p(mdata_mofa.mod[mod_name])
        sc.pp.scale(mdata_mofa.mod[mod_name])

    t0 = time.perf_counter()
    mu.tl.mofa(mdata_mofa, n_factors=n_topics, convergence_mode="fast",
               seed=seed, use_obs="intersection")
    elapsed = time.perf_counter() - t0
    n_factors = mdata_mofa.obsm["X_mofa"].shape[1]
    return {"time": elapsed, "epochs": n_factors, "elbo": None}


def _empty():
    return {"time": None, "epochs": None, "elbo": None}


def main(snakemake):
    cfg = snakemake.config
    scfg = cfg["scaling"]["multimodal_cells"]
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir

    rna_orig = os.path.join(data_dir, scfg["lymphoma_rna"])
    atac_orig = os.path.join(data_dir, scfg["lymphoma_atac"])
    n_topics = scfg["n_topics"]
    max_epochs = scfg["max_epochs"]
    batch_size = scfg["batch_size"]
    seed = scfg["seed"]
    n_runs = scfg["n_runs"]
    n_cells_list = sorted(scfg["n_cells"])

    os.makedirs(out_dir, exist_ok=True)
    synth_dir = os.path.join(out_dir, "synthetic_data")
    os.makedirs(synth_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaling_results.json")

    # Step 1: Fit SHARE-Topic if needed
    model_path = os.path.join(out_dir, "share_topic_fitted.pt")
    if not os.path.exists(model_path):
        fit_share_topic(rna_orig, atac_orig, model_path, n_topics)

    # Step 2: Generate synthetic data
    print("\nGenerating synthetic data...")
    for n_cells in n_cells_list:
        generate_synthetic(model_path, rna_orig, atac_orig, synth_dir, n_cells, seed)

    # Step 3: Benchmark
    results = {
        "n_cells": [], "run": [],
        "multivi": [], "topomics": [], "mofa": [],
        "n_topics": n_topics, "max_epochs": max_epochs, "n_runs": n_runs,
    }

    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        for n_cells in n_cells_list:
            rna_path = os.path.join(synth_dir, f"synth_rna_{n_cells}.h5ad")
            atac_path = os.path.join(synth_dir, f"synth_atac_{n_cells}.h5ad")

            if not os.path.exists(rna_path):
                continue

            print(f"\n{'='*60}")
            print(f"  N={n_cells:,}  Run {run_idx+1}/{n_runs}  seed={run_seed}")
            print(f"{'='*60}")

            results["n_cells"].append(n_cells)
            results["run"].append(run_idx)

            try:
                mdata = load_synthetic_mudata(rna_path, atac_path)
                print(f"  RNA: {mdata.mod['rna'].shape}, ATAC: {mdata.mod['atac'].shape}")
            except Exception as e:
                print(f"  Load failed: {e}")
                results["multivi"].append(_empty())
                results["topomics"].append(_empty())
                results["mofa"].append(_empty())
                continue

            # MultiVI
            print("  MultiVI...", end=" ", flush=True)
            try:
                r = train_multivi(mdata, n_topics, max_epochs, batch_size)
                print(f"{r['time']:.1f}s ({r['epochs']} ep)")
                results["multivi"].append(r)
            except Exception as e:
                print(f"FAILED: {e}")
                results["multivi"].append(_empty())

            # TopOmics
            print("  TopOmics...", end=" ", flush=True)
            try:
                r = train_topomics(mdata, n_topics, max_epochs, batch_size)
                print(f"{r['time']:.1f}s ({r['epochs']} ep)")
                results["topomics"].append(r)
            except Exception as e:
                print(f"FAILED: {e}")
                results["topomics"].append(_empty())

            # MOFA+ (only up to MOFA_MAX_CELLS)
            if n_cells <= MOFA_MAX_CELLS:
                print("  MOFA+...", end=" ", flush=True)
                try:
                    r = train_mofa(mdata, n_topics, run_seed)
                    print(f"{r['time']:.1f}s")
                    results["mofa"].append(r)
                except Exception as e:
                    print(f"FAILED: {e}")
                    results["mofa"].append(_empty())
            else:
                results["mofa"].append(_empty())

            # Save after each run
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"\nResults saved to {out_path}")


main(snakemake)
