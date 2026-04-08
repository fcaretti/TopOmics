# --- Retina-specific rules ---
# Retina uses a batch-correction axis instead of aggregation.


rule train_topomics_retina:
    output:
        sentinel=f"{OUTPUT}/retina/topomics_{{cfg}}/done.sentinel",
    params:
        cfg="{cfg}",
        out_dir=f"{OUTPUT}/retina/topomics_{{cfg}}",
        data_dir=DATA,
    wildcard_constraints:
        cfg="|".join(RETINA_CONFIGS),
    resources:
        gpu=1,
    script:
        "../scripts/train_topomics_retina.py"


rule train_retina_scvi:
    output:
        sentinel=f"{OUTPUT}/retina/{{variant}}/done.sentinel",
    params:
        variant="{variant}",
        out_dir=f"{OUTPUT}/retina/{{variant}}",
        data_dir=DATA,
    wildcard_constraints:
        variant="scvi_batch|scvi_no_batch",
    resources:
        gpu=1,
    script:
        "../scripts/train_scvi.py"


rule train_retina_linear_scvi:
    output:
        sentinel=f"{OUTPUT}/retina/{{variant}}/done.sentinel",
    params:
        variant="{variant}",
        out_dir=f"{OUTPUT}/retina/{{variant}}",
        data_dir=DATA,
    wildcard_constraints:
        variant="linear_scvi_batch|linear_scvi_no_batch",
    resources:
        gpu=1,
    script:
        "../scripts/train_linear_scvi.py"
