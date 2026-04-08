# --- Scaling benchmark rules ---
# Two benchmarks: feature-fraction sweep and cell-count sweep (both multimodal RNA+ATAC).


rule scaling_multimodal_features:
    output:
        sentinel=f"{OUTPUT}/scaling/multimodal_features/done.sentinel",
    params:
        out_dir=f"{OUTPUT}/scaling/multimodal_features",
        data_dir=DATA,
    resources:
        gpu=1,
    script:
        "../scripts/scaling_multimodal_features.py"


rule scaling_multimodal_cells:
    output:
        sentinel=f"{OUTPUT}/scaling/multimodal_cells/done.sentinel",
    params:
        out_dir=f"{OUTPUT}/scaling/multimodal_cells",
        data_dir=DATA,
    resources:
        gpu=1,
    script:
        "../scripts/scaling_multimodal_cells.py"
