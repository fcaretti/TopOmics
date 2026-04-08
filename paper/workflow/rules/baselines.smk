# --- Baseline training rules ---
# One rule per baseline method, parameterized by dataset wildcard.
# Each script reads dataset info from config.yaml via snakemake.config.

# Helper: all non-retina datasets
_NON_RETINA = "|".join(ds for ds in config["datasets"] if ds != "retina")


rule train_multivi:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/multivi/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/multivi",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_multivi.py"


rule train_multivi_linear:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/multivi_linear/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/multivi_linear",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_multivi.py"


rule train_totalvi:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/totalvi/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/totalvi",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_totalvi.py"


rule train_scvi_baseline:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/scvi/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/scvi",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_scvi.py"


rule train_amortized_lda:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/amortized_lda_{{modality}}/done.sentinel",
    params:
        dataset="{dataset}",
        modality="{modality}",
        out_dir=f"{OUTPUT}/{{dataset}}/amortized_lda_{{modality}}",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_amortized_lda.py"


rule train_mofa:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/mofa/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/mofa",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    script:
        "../scripts/train_mofa.py"


rule train_glue:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/glue/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/glue",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_glue.py"


rule train_spatialglue:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/spatialglue/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/spatialglue",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_spatialglue.py"


rule train_stamp:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/stamp/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/stamp",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_stamp.py"


rule train_cosmos:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/cosmos/done.sentinel",
    params:
        dataset="{dataset}",
        out_dir=f"{OUTPUT}/{{dataset}}/cosmos",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_cosmos.py"


# Catch-all for the unimodal amortized_lda (no modality suffix, e.g. visium)
rule train_amortized_lda_unimodal:
    output:
        sentinel=f"{OUTPUT}/{{dataset}}/amortized_lda/done.sentinel",
    params:
        dataset="{dataset}",
        modality="rna",
        out_dir=f"{OUTPUT}/{{dataset}}/amortized_lda",
        data_dir=DATA,
    wildcard_constraints:
        dataset=_NON_RETINA,
    resources:
        gpu=1,
    script:
        "../scripts/train_amortized_lda.py"
