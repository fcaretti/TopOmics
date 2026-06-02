"""Test data import and from_data() functionality with synthetic data."""

import numpy as np
import pytest
from anndata import AnnData
from mudata import MuData

from topomics.data import (
    detect_data_type,
    extract_from_adata_dict,
    extract_from_anndata,
    extract_from_mudata,
    validate_data_type,
)
from topomics.models import MultimodalAmortizedLDA


@pytest.fixture
def synthetic_adata_rna():
    """Create synthetic RNA data."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    # Create count matrix
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(float)

    adata = AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Add a layer
    adata.layers["counts"] = X.copy()
    adata.layers["normalized"] = X / X.sum(axis=1, keepdims=True)

    return adata


@pytest.fixture
def synthetic_adata_protein():
    """Create synthetic protein data."""
    np.random.seed(43)
    n_cells = 100
    n_proteins = 20

    # Create count matrix
    X = np.random.poisson(3, size=(n_cells, n_proteins)).astype(float)

    adata = AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add layers
    adata.layers["raw"] = X.copy()
    adata.layers["counts"] = X.copy()

    return adata


@pytest.fixture
def synthetic_mudata(synthetic_adata_rna, synthetic_adata_protein):
    """Create synthetic MuData."""
    return MuData({"rna": synthetic_adata_rna, "protein": synthetic_adata_protein})


# ============================================================================
# Type Detection Tests
# ============================================================================


def test_detect_anndata(synthetic_adata_rna):
    """Test AnnData detection."""
    assert detect_data_type(synthetic_adata_rna) == "anndata"


def test_detect_mudata(synthetic_mudata):
    """Test MuData detection."""
    assert detect_data_type(synthetic_mudata) == "mudata"


def test_detect_dict(synthetic_adata_rna, synthetic_adata_protein):
    """Test dict detection."""
    data_dict = {"rna": synthetic_adata_rna, "protein": synthetic_adata_protein}
    assert detect_data_type(data_dict) == "dict"


def test_detect_unknown():
    """Test unknown type detection."""
    assert detect_data_type("invalid") == "unknown"


def test_validate_data_type_success(synthetic_adata_rna):
    """Test validation succeeds for valid types."""
    validate_data_type(synthetic_adata_rna)  # Should not raise


def test_validate_data_type_failure():
    """Test validation fails for invalid types."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        validate_data_type("invalid")


# ============================================================================
# Extraction Tests
# ============================================================================


def test_extract_from_mudata_basic(synthetic_mudata):
    """Test basic MuData extraction."""
    adata_concat, metadata = extract_from_mudata(synthetic_mudata)

    assert adata_concat.n_obs == 100
    assert adata_concat.n_vars == 70  # 50 genes + 20 proteins
    assert metadata["modality_names"] == ["rna", "protein"]
    assert metadata["feature_counts"] == [50, 20]
    assert metadata["spatial_info"] is None


def test_extract_from_mudata_with_layers(synthetic_mudata):
    """Test MuData extraction with layer specification."""
    # Test with dict layers
    adata_concat, metadata = extract_from_mudata(
        synthetic_mudata, layers={"rna": "counts", "protein": "raw"}
    )

    assert adata_concat.n_obs == 100
    assert adata_concat.n_vars == 70
    assert metadata["layer_dict"] == {"rna": "counts", "protein": "raw"}

    # Test with string layer (same for all)
    adata_concat2, metadata2 = extract_from_mudata(synthetic_mudata, layers="counts")

    assert metadata2["layer_dict"] == {"rna": "counts", "protein": "counts"}


def test_extract_from_mudata_modality_subset(synthetic_mudata):
    """Test extracting subset of modalities."""
    adata_concat, metadata = extract_from_mudata(synthetic_mudata, modalities=["rna"])

    assert adata_concat.n_obs == 100
    assert adata_concat.n_vars == 50  # Only RNA
    assert metadata["modality_names"] == ["rna"]
    assert metadata["feature_counts"] == [50]


def test_extract_from_adata_dict(synthetic_adata_rna, synthetic_adata_protein):
    """Test extraction from dict of AnnData."""
    data_dict = {"rna": synthetic_adata_rna, "protein": synthetic_adata_protein}

    adata_concat, metadata = extract_from_adata_dict(data_dict)

    assert adata_concat.n_obs == 100
    assert adata_concat.n_vars == 70
    assert metadata["modality_names"] == ["rna", "protein"]


def test_extract_from_adata_dict_with_layers(synthetic_adata_rna, synthetic_adata_protein):
    """Test extraction from dict with layer specification."""
    data_dict = {"rna": synthetic_adata_rna, "protein": synthetic_adata_protein}

    adata_concat, metadata = extract_from_adata_dict(
        data_dict, layers={"rna": "counts", "protein": "raw"}
    )

    assert metadata["layer_dict"] == {"rna": "counts", "protein": "raw"}


def test_extract_from_anndata_basic(synthetic_adata_rna):
    """Test extraction from single AnnData."""
    adata_processed, metadata = extract_from_anndata(synthetic_adata_rna)

    assert adata_processed.n_obs == 100
    assert adata_processed.n_vars == 50
    assert metadata["modality_names"] == ["rna"]
    assert metadata["feature_counts"] == [50]


def test_extract_from_anndata_with_layer(synthetic_adata_rna):
    """Test extraction from AnnData with layer."""
    adata_processed, metadata = extract_from_anndata(
        synthetic_adata_rna, modality_name="rna", layer="counts"
    )

    assert adata_processed.n_obs == 100
    assert metadata["modality_names"] == ["rna"]
    assert metadata["layer_dict"] == {"rna": "counts"}

    # Verify layer was extracted (should be in .X now)
    assert np.allclose(adata_processed.X, synthetic_adata_rna.layers["counts"])


def test_extract_from_anndata_missing_layer(synthetic_adata_rna):
    """Test error handling for missing layer."""
    with pytest.raises(KeyError, match="Layer 'nonexistent' not found"):
        extract_from_anndata(synthetic_adata_rna, layer="nonexistent")


# ============================================================================
# from_data() Integration Tests
# ============================================================================


def test_from_data_with_mudata(synthetic_mudata):
    """Test from_data() with MuData input."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata,
        modalities=["rna", "protein"],
        layers={"rna": "counts", "protein": "raw"},
        n_topics=5,
        n_hidden=32,
    )

    assert model is not None
    assert model.n_modalities == 2
    assert model.n_inputs_modalities == [50, 20]
    assert model.module.n_topics == 5


def test_from_data_with_mudata_string_layer(synthetic_mudata):
    """Test from_data() with string layer specification."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata, layers="counts", n_topics=5, n_hidden=32
    )

    assert model is not None
    assert model.n_modalities == 2


def test_from_data_with_dict(synthetic_adata_rna, synthetic_adata_protein):
    """Test from_data() with dict input."""
    data_dict = {"rna": synthetic_adata_rna, "protein": synthetic_adata_protein}

    model = MultimodalAmortizedLDA.from_data(
        data_dict, layers={"rna": "counts"}, n_topics=5, n_hidden=32
    )

    assert model is not None
    assert model.n_modalities == 2


def test_from_data_with_anndata(synthetic_adata_rna):
    """Test from_data() with single AnnData input."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_adata_rna,
        modalities=["rna"],
        layers="counts",
        n_topics=5,
        n_hidden=32,
    )

    assert model is not None
    assert model.n_modalities == 1
    assert model.n_inputs_modalities == [50]


def test_from_data_auto_infer_likelihoods(synthetic_mudata):
    """Test that likelihoods are auto-inferred."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata, n_topics=5, n_hidden=32
    )

    # Should auto-infer gamma_poisson for rna, multinomial for protein
    assert model.likelihoods == ["gamma_poisson", "multinomial"]


def test_from_data_custom_likelihoods(synthetic_mudata):
    """Test custom likelihoods override auto-inference."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata,
        likelihoods=["multinomial", "multinomial"],
        n_topics=5,
        n_hidden=32,
    )

    assert model.likelihoods == ["multinomial", "multinomial"]


def test_from_data_invalid_type():
    """Test from_data() with invalid input type."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        MultimodalAmortizedLDA.from_data("invalid", n_topics=5)


# ============================================================================
# Training Test (Quick Sanity Check)
# ============================================================================


def test_model_trains(synthetic_mudata):
    """Test that model can train for a few iterations."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata, n_topics=5, n_hidden=32
    )

    # Quick training run (just to verify it doesn't crash)
    model.train(max_epochs=2, check_val_every_n_epoch=None)

    # Verify we can get outputs
    theta = model.get_cell_topic_dist()
    assert theta.shape == (100, 5)  # 100 cells, 5 topics

    # Verify topics sum to 1
    assert np.allclose(theta.sum(axis=1), 1.0)


def test_model_trains_with_layer_extraction(synthetic_mudata):
    """Test that model trains correctly with layer extraction."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata, layers={"rna": "counts"}, n_topics=5, n_hidden=32
    )

    # Quick training
    model.train(max_epochs=2, check_val_every_n_epoch=None)

    # Get feature-topic distributions
    phi_dict = model.get_feature_topic_dist(n_samples=100, as_dict=True)

    assert len(phi_dict) == 2  # Two modalities
    assert 0 in phi_dict and 1 in phi_dict  # Keys are modality indices


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_extract_missing_modality(synthetic_mudata):
    """Test error when requesting non-existent modality."""
    with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
        extract_from_mudata(synthetic_mudata, modalities=["nonexistent"])


def test_extract_missing_layer(synthetic_mudata):
    """Test error when requesting non-existent layer."""
    with pytest.raises(KeyError, match="Layer 'nonexistent' not found"):
        extract_from_mudata(synthetic_mudata, layers={"rna": "nonexistent"})


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def test_from_mudata_still_works(synthetic_mudata):
    """Test that the original from_mudata() method still works."""
    model = MultimodalAmortizedLDA.from_mudata(
        synthetic_mudata,
        modality_order=["rna", "protein"],
        layer_dict=None,
        n_topics=5,
        n_hidden=32,
    )

    assert model is not None
    assert model.n_modalities == 2


# ============================================================================
# New scvi-style API Tests (setup_data + instantiation)
# ============================================================================


def test_setup_data_with_mudata(synthetic_mudata):
    """Test new scvi-style API: setup_data() + instantiation with MuData."""
    # Step 1: Setup
    MultimodalAmortizedLDA.setup_data(
        synthetic_mudata,
        modalities=["rna", "protein"],
        layers="counts"
    )

    # Step 2: Instantiate
    model = MultimodalAmortizedLDA(
        synthetic_mudata,
        n_inputs_modalities=[50, 20],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None
    assert model.n_modalities == 2
    assert model.n_topics == 10


def test_setup_data_with_anndata(synthetic_adata_rna):
    """Test new scvi-style API: setup_data() + instantiation with AnnData."""
    # Step 1: Setup
    MultimodalAmortizedLDA.setup_data(
        synthetic_adata_rna,
        modalities=["rna"],
        layers="counts"
    )

    # Step 2: Instantiate
    model = MultimodalAmortizedLDA(
        synthetic_adata_rna,
        n_inputs_modalities=[50],
        likelihoods=["gamma_poisson"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None
    assert model.n_modalities == 1
    assert model.n_topics == 10


def test_setup_data_with_dict(synthetic_adata_rna, synthetic_adata_protein):
    """Test new scvi-style API: setup_data() + instantiation with dict."""
    adata_dict = {"rna": synthetic_adata_rna, "protein": synthetic_adata_protein}

    # Step 1: Setup (returns processed AnnData for dict case)
    adata_concat = MultimodalAmortizedLDA.setup_data(
        adata_dict,
        layers={"rna": "counts", "protein": "counts"}
    )

    # Step 2: Instantiate with the processed AnnData
    model = MultimodalAmortizedLDA(
        adata_concat,
        n_inputs_modalities=[50, 20],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None
    assert model.n_modalities == 2
    assert model.n_topics == 10


def test_setup_mudata_with_new_parameters(synthetic_mudata):
    """Test setup_mudata() with new parameter names (modalities, layers)."""
    # Test with new parameter names
    mdata_setup, modality_names, feat_counts = MultimodalAmortizedLDA.setup_mudata(
        synthetic_mudata,
        modalities=["rna", "protein"],  # New name
        layers="counts",                 # New name
    )

    assert modality_names == ["rna", "protein"]
    assert len(feat_counts) == 2

    # Can instantiate model
    model = MultimodalAmortizedLDA(
        mdata_setup,
        n_inputs_modalities=feat_counts,
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None
    assert model.n_modalities == 2


def test_setup_anndata_backward_compatibility(synthetic_adata_rna):
    """Test that old setup_anndata() API still works."""
    # Old API with layer parameter
    MultimodalAmortizedLDA.setup_anndata(synthetic_adata_rna, layer="counts")

    model = MultimodalAmortizedLDA(
        synthetic_adata_rna,
        n_inputs_modalities=[50],
        likelihoods=["gamma_poisson"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None


def test_setup_anndata_with_new_api(synthetic_adata_rna):
    """Test setup_anndata() with new parameters (layers, modalities)."""
    # New API with layers parameter
    MultimodalAmortizedLDA.setup_anndata(
        synthetic_adata_rna,
        modalities=["rna"],
        layers="counts"
    )

    model = MultimodalAmortizedLDA(
        synthetic_adata_rna,
        n_inputs_modalities=[50],
        likelihoods=["gamma_poisson"],
        n_topics=10,
        n_hidden=16
    )

    assert model is not None


def test_from_data_still_works_after_refactoring(synthetic_mudata):
    """Test that from_data() convenience wrapper still works after refactoring."""
    model = MultimodalAmortizedLDA.from_data(
        synthetic_mudata,
        modalities=["rna", "protein"],
        layers="counts",
        n_topics=10,
        n_hidden=16
    )

    assert model is not None
    assert model.n_modalities == 2
    assert model.n_topics == 10


def test_setup_data_with_mudata_trains(synthetic_mudata):
    """Test that models created with setup_data() can train."""
    MultimodalAmortizedLDA.setup_data(
        synthetic_mudata,
        modalities=["rna", "protein"],
        layers="counts"
    )

    model = MultimodalAmortizedLDA(
        synthetic_mudata,
        n_inputs_modalities=[50, 20],
        likelihoods=["gamma_poisson", "multinomial"],
        n_topics=10,
        n_hidden=16
    )

    # Train for 1 epoch
    model.train(max_epochs=1, batch_size=32)

    # Model should have been trained
    assert model.is_trained
