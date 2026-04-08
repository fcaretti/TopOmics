"""Train totalVI baseline (RNA + Protein)."""
import json, os, sys, warnings
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    import scvi

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]
    ds_cfg = cfg["datasets"][dataset]

    print(f"=== totalVI: {dataset} ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])
    mdata = data.copy()

    modalities = ds_cfg["modalities"]
    rna_mod = modalities[0]
    prot_mod = modalities[1]

    if "counts" in mdata.mod[rna_mod].layers:
        mdata.mod[rna_mod].X = mdata.mod[rna_mod].layers["counts"].copy()
    if "counts" in mdata.mod[prot_mod].layers:
        mdata.mod[prot_mod].X = mdata.mod[prot_mod].layers["counts"].copy()

    # totalVI protein prior init requires dense protein matrix
    if sp.issparse(mdata.mod[prot_mod].X):
        mdata.mod[prot_mod].X = np.asarray(mdata.mod[prot_mod].X.todense())

    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer=None,
        protein_layer=None,
        batch_key=None,
        modalities={"rna_layer": rna_mod, "protein_layer": prot_mod},
    )

    model = scvi.model.TOTALVI(mdata, n_latent=bl["n_latent"], n_hidden=bl["n_hidden"])
    model.train(max_epochs=bl["max_epochs"], train_size=0.8, early_stopping=True)

    latent = model.get_latent_representation()

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": bl["n_latent"], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
