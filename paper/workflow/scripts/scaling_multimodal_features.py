"""
Scaling benchmark: multimodal feature-fraction sweep on lymphoma RNA+ATAC.

Trains MultiVI, OmicsTopic, and MOFA+ for increasing feature fractions
of a fixed-size dataset and records wall-clock training time.
"""
import json
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")


def load_data(data_path, n_hvg_full, n_peaks_full):
    import mudata as mu
    import scanpy as sc

    mdata = mu.read_h5mu(data_path)
    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]

    sc.pp.filter_genes(rna, min_cells=1)

    if sp.issparse(atac.X):
        atac.X = atac.X.copy()
        atac.X.data = np.ones_like(atac.X.data)
    else:
        atac.X = (atac.X > 0).astype(np.float32)
    sc.pp.filter_genes(atac, min_cells=1)

    sc.pp.highly_variable_genes(rna, n_top_genes=min(n_hvg_full, rna.n_vars),
                                flavor="seurat_v3", subset=False)
    sc.pp.highly_variable_genes(atac, n_top_genes=min(n_peaks_full, atac.n_vars),
                                flavor="seurat_v3", subset=False)
    return mdata


def subsample_features(mdata_full, rna_frac, atac_frac, n_hvg_full, n_peaks_full, seed):
    import mudata as mu

    rna_full = mdata_full.mod["rna"]
    atac_full = mdata_full.mod["atac"]

    n_rna = max(1, min(int(n_hvg_full * rna_frac), rna_full.n_vars))
    n_atac = max(1, min(int(n_peaks_full * atac_frac), atac_full.n_vars))

    for mod, n in [(rna_full, n_rna), (atac_full, n_atac)]:
        if "highly_variable_rank" in mod.var.columns:
            idx = np.argsort(mod.var["highly_variable_rank"].values)[:n]
        else:
            idx = np.argsort(-mod.var["dispersions_norm"].values)[:n]
        if mod is rna_full:
            rna_idx = idx
        else:
            atac_idx = idx

    return mu.MuData({"rna": rna_full[:, rna_idx].copy(), "atac": atac_full[:, atac_idx].copy()})


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
    return elapsed, n_epochs


def train_omics_topic(mdata, n_topics, max_epochs, batch_size):
    from omics_topic.models import MultimodalAmortizedLDA
    model = MultimodalAmortizedLDA.from_mudata(
        mdata, layer_dict={"rna": None, "atac": None},
        n_topics=n_topics, likelihoods=["gamma_poisson", "bernoulli"],
    )
    t0 = time.perf_counter()
    model.train(
        max_epochs=max_epochs, batch_size=batch_size,
        early_stopping=True, early_stopping_monitor="elbo_val",
        early_stopping_patience=10, early_stopping_min_delta=0.0,
    )
    elapsed = time.perf_counter() - t0
    n_epochs = len(model.history["elbo_train"])
    return elapsed, n_epochs


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
    return elapsed, n_factors


def main(snakemake):
    cfg = snakemake.config
    scfg = cfg["scaling"]["multimodal_features"]
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir

    data_path = os.path.join(data_dir, scfg["data_path"])
    n_hvg_full = scfg["n_hvg_full"]
    n_peaks_full = scfg["n_peaks_full"]
    n_topics = scfg["n_topics"]
    max_epochs = scfg["max_epochs"]
    batch_size = scfg["batch_size"]
    seed = scfg["seed"]
    fractions = sorted(scfg["feature_fractions"])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaling_results.json")

    print("Loading data...")
    mdata_full = load_data(data_path, n_hvg_full, n_peaks_full)
    print(f"  RNA: {mdata_full.mod['rna'].shape}, ATAC: {mdata_full.mod['atac'].shape}")

    results = {
        "feature_fractions": [], "n_rna_features": [], "n_atac_features": [],
        "multivi_time": [], "multivi_epochs": [],
        "omics_topic_time": [], "omics_topic_epochs": [],
        "mofa_time": [], "mofa_factors": [],
        "n_topics": n_topics, "max_epochs": max_epochs,
    }

    for frac in fractions:
        print(f"\n{'='*60}\n  Feature fraction = {frac:.0%}\n{'='*60}")

        mdata = subsample_features(mdata_full, frac, frac, n_hvg_full, n_peaks_full, seed)
        n_rna = mdata.mod["rna"].n_vars
        n_atac = mdata.mod["atac"].n_vars
        print(f"  RNA: {n_rna}, ATAC: {n_atac}")

        results["feature_fractions"].append(frac)
        results["n_rna_features"].append(n_rna)
        results["n_atac_features"].append(n_atac)

        # MultiVI
        print("  Training MultiVI...", end=" ", flush=True)
        try:
            t, ep = train_multivi(mdata, n_topics, max_epochs, batch_size)
            print(f"{t:.1f}s ({ep} epochs)")
            results["multivi_time"].append(t)
            results["multivi_epochs"].append(ep)
        except Exception as e:
            print(f"FAILED: {e}")
            results["multivi_time"].append(None)
            results["multivi_epochs"].append(None)

        # OmicsTopic
        print("  Training OmicsTopic...", end=" ", flush=True)
        try:
            t, ep = train_omics_topic(mdata, n_topics, max_epochs, batch_size)
            print(f"{t:.1f}s ({ep} epochs)")
            results["omics_topic_time"].append(t)
            results["omics_topic_epochs"].append(ep)
        except Exception as e:
            print(f"FAILED: {e}")
            results["omics_topic_time"].append(None)
            results["omics_topic_epochs"].append(None)

        # MOFA+
        print("  Training MOFA+...", end=" ", flush=True)
        try:
            t, nf = train_mofa(mdata, n_topics, seed)
            print(f"{t:.1f}s ({nf} factors)")
            results["mofa_time"].append(t)
            results["mofa_factors"].append(nf)
        except Exception as e:
            print(f"FAILED: {e}")
            results["mofa_time"].append(None)
            results["mofa_factors"].append(None)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"\nResults saved to {out_path}")


main(snakemake)
