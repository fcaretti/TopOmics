"""Train GLUE baseline (RNA + ATAC with genomic guidance)."""
import json, os, sys, warnings
import numpy as np
import scanpy as sc
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset


def main(snakemake):
    import scglue

    dataset = snakemake.params.dataset
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config
    bl = cfg["baseline_defaults"]

    print(f"=== GLUE: {dataset} ===")

    data = load_dataset(dataset, data_dir, cfg["datasets"])
    ds_cfg = cfg["datasets"][dataset]
    modalities = ds_cfg["modalities"]

    rna = data.mod[modalities[0]].copy()
    atac = data.mod[modalities[1]].copy()

    if "counts" in rna.layers:
        rna.X = rna.layers["counts"].copy()
    if "counts" in atac.layers:
        atac.X = atac.layers["counts"].copy()

    # RNA preprocessing
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, use_highly_variable=False)

    # ATAC preprocessing with LSI
    atac.layers["counts"] = atac.X.copy()
    scglue.data.lsi(atac, n_components=100)

    # Parse ATAC peak coordinates
    if "chromStart" not in atac.var.columns:
        coords = atac.var_names.str.extract(r"(chr[^:_]+)[:\-_](\d+)[:\-_](\d+)")
        if not coords.isna().any().any():
            atac.var["chrom"] = coords[0].values
            atac.var["chromStart"] = coords[1].astype(int).values
            atac.var["chromEnd"] = coords[2].astype(int).values

    # Gene coordinates
    if "chromStart" not in rna.var.columns:
        try:
            scglue.data.get_gene_annotation(
                rna,
                gtf="http://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
                gtf_by="gene_name",
            )
        except Exception:
            pass

    # Build guidance graph
    try:
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    except Exception as e:
        print(f"  GLUE guidance graph failed: {e}")
        os.makedirs(out_dir, exist_ok=True)
        with open(snakemake.output.sentinel, "w") as f:
            f.write("skipped\n")
        return

    scglue.models.configure_dataset(rna, "NB", use_highly_variable=False, use_layer="counts", use_rep="X_pca")
    scglue.models.configure_dataset(atac, "NB", use_highly_variable=False, use_layer="counts", use_rep="X_lsi")

    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac},
        guidance,
        fit_kws={"directory": os.path.join(out_dir, "glue_checkpoints")},
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    latent = rna.obsm["X_glue"]

    os.makedirs(out_dir, exist_ok=True)
    glue.save(os.path.join(out_dir, "model", "glue.dill"))
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"n_latent": latent.shape[1], "n_cells": latent.shape[0]}, f, indent=2)

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Saved to: {out_dir}")


main(snakemake)
