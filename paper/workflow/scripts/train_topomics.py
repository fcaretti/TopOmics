"""
Unified TopOmics training script for all datasets (except retina).
Called by Snakemake with params: dataset, cfg, out_dir, data_dir.
"""

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*was not registered in the param store.*")
warnings.filterwarnings("ignore", message=".*Found plate statements in guide but not model.*")

# Allow imports from the scripts directory
sys.path.insert(0, os.path.dirname(__file__))
from preprocess_data import load_dataset

from topomics import MultimodalAmortizedLDA


def main(snakemake):
    dataset_name = snakemake.params.dataset
    cfg_name = snakemake.params.cfg
    out_dir = snakemake.params.out_dir
    data_dir = snakemake.params.data_dir
    cfg = snakemake.config

    ds_cfg = cfg["datasets"][dataset_name]
    defaults = cfg["topomics_defaults"]
    ds_topomics = ds_cfg.get("topomics", {})

    # Resolve TopOmics config (multimodal or unimodal)
    if cfg_name in cfg["topomics_configs"]:
        run_cfg = cfg["topomics_configs"][cfg_name]
    elif cfg_name in cfg["topomics_configs_unimodal"]:
        run_cfg = cfg["topomics_configs_unimodal"][cfg_name]
    else:
        raise ValueError(f"Unknown config: {cfg_name}")

    # Merge parameters: defaults < dataset-specific < run config
    n_topics = ds_topomics.get("n_topics", defaults["n_topics"])
    n_hidden = ds_topomics.get("n_hidden", defaults["n_hidden"])
    max_epochs = ds_topomics.get("max_epochs", 500)
    batch_size = ds_topomics.get("batch_size", 128)
    lr = ds_topomics.get("lr", None)
    train_size = defaults["train_size"]

    feature_prior_type = run_cfg["feature_prior_type"]
    aggregation_type = run_cfg.get("aggregation_type", "moe")
    weight_mode = run_cfg.get("weight_mode", "cell")

    # Dataset-specific weight_mode override (e.g. mouse_brain uses "universal")
    if "weight_mode_override" in ds_topomics and aggregation_type == "moe":
        weight_mode = ds_topomics["weight_mode_override"]

    modalities = ds_cfg["modalities"]
    likelihoods = ds_cfg["likelihoods"]
    is_spatial = ds_cfg.get("spatial", False)

    print(f"=== TopOmics: {dataset_name} / {cfg_name} ===")
    print(f"  prior={feature_prior_type}, agg={aggregation_type}, weight={weight_mode}")
    print(f"  n_topics={n_topics}, n_hidden={n_hidden}, epochs={max_epochs}")

    # --- Load data ---
    data = load_dataset(dataset_name, data_dir, cfg["datasets"])

    # --- Build model ---
    is_multimodal = ds_cfg["type"] == "multimodal"
    is_mudata = hasattr(data, "mod")

    model_kwargs = dict(
        n_topics=n_topics,
        likelihoods=likelihoods,
        n_hidden=n_hidden,
        cell_topic_prior=1 / n_topics,
        topic_feature_prior_type=feature_prior_type,
        learnable_dispersion=defaults["learnable_dispersion"],
        global_dispersion=defaults["global_dispersion"],
        normalize_encoder_inputs=True,
    )

    if is_multimodal:
        model_kwargs["weight_mode"] = weight_mode
        model_kwargs["aggregation_type"] = aggregation_type
        model_kwargs["att_dim"] = 16

    if is_spatial and is_mudata:
        # Use from_mudata with spatial graph
        layer_dict = {}
        for mod_name in modalities:
            mod = data.mod[mod_name]
            if "binary" in mod.layers:
                layer_dict[mod_name] = "binary"
            elif "counts" in mod.layers:
                layer_dict[mod_name] = "counts"
            else:
                layer_dict[mod_name] = None

        model = MultimodalAmortizedLDA.from_mudata(
            data,
            layer_dict=layer_dict,
            spatial_key="spatial_connectivities",
            gcn_n_layers=defaults["gcn_n_layers"],
            gcn_conv_type=defaults["gcn_conv_type"],
            gcn_alpha_init=defaults["gcn_alpha"],
            gcn_use_learned_alpha=defaults["gcn_use_learned_alpha"],
            kl_weight=1,
            **model_kwargs,
        )
    elif is_spatial and not is_mudata:
        # Unimodal spatial (e.g. visium)
        model = MultimodalAmortizedLDA.from_data(
            data,
            n_topics=n_topics,
            likelihoods=likelihoods,
            cell_topic_prior=1 / n_topics,
            spatial_keys="spatial_connectivities",
            gcn_n_layers=defaults["gcn_n_layers"],
            gcn_conv_type=defaults["gcn_conv_type"],
            gcn_alpha_init=defaults["gcn_alpha"],
            gcn_use_learned_alpha=defaults["gcn_use_learned_alpha"],
            kl_weight=1,
            use_feature_background=False,
            topic_feature_prior_type=feature_prior_type,
            learnable_dispersion=defaults["learnable_dispersion"],
            global_dispersion=defaults["global_dispersion"],
        )
    elif is_multimodal:
        # Non-spatial multimodal
        layers_arg = "counts" if any("counts" in data.mod[m].layers for m in modalities) else None
        model = MultimodalAmortizedLDA.from_data(
            data,
            modalities=modalities,
            layers=layers_arg,
            **model_kwargs,
        )
    else:
        # Non-spatial unimodal (shouldn't happen with current datasets, but safe)
        model = MultimodalAmortizedLDA.from_data(
            data,
            n_topics=n_topics,
            likelihoods=likelihoods,
            cell_topic_prior=1 / n_topics,
            topic_feature_prior_type=feature_prior_type,
            n_hidden=n_hidden,
        )

    # --- Train ---
    plan_kwargs = {}
    if lr is not None:
        plan_kwargs["optim_kwargs"] = {"lr": lr}

    val_size = 1.0 - train_size if train_size < 1.0 else 0.0
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_size=train_size,
        validation_size=val_size,
        plan_kwargs=plan_kwargs if plan_kwargs else None,
    )

    # --- Save ---
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model"), overwrite=True)

    # Latent representation
    n_obs = data.n_obs if hasattr(data, "n_obs") else data.shape[0]
    if is_mudata:
        theta = model.get_latent_representation(batch_size=n_obs)
    else:
        theta = model.get_latent_representation(data, batch_size=n_obs)

    latent_vals = theta.values if hasattr(theta, "values") else np.asarray(theta)
    np.save(os.path.join(out_dir, "latent_representation.npy"), latent_vals)

    # Metrics
    metrics = {
        "perplexity": model.get_perplexity(),
        "entropy": model.get_entropy(normalised=True),
        "diversity": model.get_topic_diversity(),
    }
    if is_multimodal:
        for mod_name, ppl in model.get_perplexity_per_modality().items():
            metrics[f"perplexity_{mod_name}"] = ppl
        for mod_name in modalities:
            metrics[f"diversity_{mod_name}"] = model.get_topic_diversity(modality=mod_name)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Sentinel
    with open(snakemake.output.sentinel, "w") as f:
        f.write("done\n")

    print(f"  Perplexity: {metrics['perplexity']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    print(f"  Diversity: {metrics['diversity']:.4f}")
    print(f"  Saved to: {out_dir}")


main(snakemake)
