# --- TopOmics training rules ---
# Multimodal datasets (4 configs) and unimodal datasets (2 configs)
# are matched by the same wildcard rule; the script resolves the
# config from config.yaml at runtime.


def _topomics_dataset_constraint():
    """Wildcard constraint: all dataset names except retina."""
    all_ds = [ds for ds in config["datasets"] if ds != "retina"]
    return "|".join(all_ds)


rule train_topomics:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/topomics_{{cfg}}/done.sentinel",
    params:
        dataset="{dataset}",
        cfg="{cfg}",
        out_dir=f"{OUTPUT}/{{dataset}}/topomics_{{cfg}}",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_topomics_dataset_constraint(),
    resources:
        gpu=1,
    script:
        "../scripts/train_topomics.py"
