"""Train MOFA+ baseline."""
import json, os, sys, warnings
import numpy as np
import scanpy as sc
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    import muon as mu

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]
    ds_cfg = cfg["datasets"][dataset]

    print(f"=== MOFA+: {dataset} ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])
    mdata = data.copy()

    # MOFA needs log-normalized + scaled data
    for mod_name in ds_cfg["modalities"]:
        mod = mdata.mod[mod_name]
        if "counts" in mod.layers:
            mod.X = mod.layers["counts"].copy()
        sc.pp.normalize_total(mod, target_sum=1e4)
        sc.pp.log1p(mod)
        sc.pp.scale(mod)

    mu.tl.mofa(mdata, n_factors=bl["n_latent"], convergence_mode="medium", use_obs="intersection")

    latent = mdata.obsm["X_mofa"]

    os.makedirs(out_dir, exist_ok=True)
    mdata.write(os.path.join(out_dir, "mdata_mofa.h5mu"))
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_factors": bl["n_latent"], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
