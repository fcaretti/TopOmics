import muon as mu
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

MuDataType = mu.MuData


class BaseTopicModel:
    """
    Base class for all models in the topomics package.

    This class provides a common interface and shared functionality for all models.
    It can be extended by specific model implementations.
    """

    spatial: bool = False  # set True when spatial connectivities are provided

    def __init__(
        self,
        mdata: MuDataType | dict[str, AnnData] | list[AnnData] | AnnData,
        modalities: list[str] | str | None = None,
    ):
        """
        Initialize the BaseModel. Checks the input data.

        Args:
            mdata: Multi-modal data container:
                - MuData object (mu.MuData),
                - dict mapping modality names to AnnData,
                - list of AnnData objects (requires `modalities`)
                - single AnnData object (requires `modalities`).
            modalities: Names corresponding to each AnnData in a list input.
        Initializes:
                - `self.data_dict`: Dictionary mapping modality names to tensors.
                - `self.modalities`: List of modality names.
                - `self.n_cells`: Number of cells (assumed to be the same across modalities).
        """
        self.spatial = False
        self.check_input(mdata, modalities)
        self.check_modalities_names()
        self.n_modalities = len(self.modalities)

        n_cells_set = {v.shape[0] for v in self.data_dict.values()}
        if len(n_cells_set) != 1:
            raise ValueError("All modalities must share the same cells / order")

        self.n_cells = self.data_dict[self.modalities[0]].shape[0]

        # Initialize metric cache
        self._cached_metrics = {}

        print("Initializing model with the following modalities:", self.modalities)

    def fit(self, data):
        """
        Fit the model to the provided data.

        Parameters
        ----------
            data: The input data to fit the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, data):
        """
        Predict using the fitted model on the provided data.

        Parameters
        ----------
            data: The input data for prediction.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def check_input(self, mdata, modalities):
        """
        Validate and process the input data.

        Checks that data are adata or mudata objects, and that the modalities are correctly specified.
        """
        if isinstance(mdata, dict):
            self.data_dict = mdata
        elif MuDataType and isinstance(mdata, MuDataType):
            self.data_dict = {mod: mdata[mod].X for mod in mdata.mod}
        elif isinstance(mdata, list):
            if modalities is None or len(modalities) != len(mdata):
                raise ValueError("When passing a list of AnnData, `modalities` must be a list of the same length.")
            if not all(isinstance(data, AnnData) for data in mdata):
                raise ValueError("All elements in the list must be AnnData objects.")
            self.data_dict = dict(zip(modalities, mdata.X, strict=False))
        elif isinstance(mdata, AnnData):
            if modalities is not None and len(modalities) != 1:
                raise ValueError("You passed a single AnnData but provided multiple modality names. ")
            if modalities is None:
                raise Warning("No modality names provided for a single AnnData. Defaulting to 'rna'.")
                self.modalities = ["rna"]
            self.data_dict = {modalities[0]: mdata.X}
            self.modalities = modalities
        else:
            raise TypeError(
                "`mdata` must be a MuData object, a dict of AnnData, a list of AnnData or a single AnnData."
            )

    def check_modalities_names(self):
        """
        Standardize and validate modality keys in data_dict.

        Maps various synonyms to 'rna', 'protein', or 'chromatin',
        and rebuilds data_dict with standardized keys.
        """
        if len(self.data_dict) == 0:
            raise ValueError("data_dict is empty. Please provide valid data.")

        # Ensure all keys are strings
        for k in self.data_dict.keys():
            if not isinstance(k, str):
                raise ValueError(f"Invalid modality key {k!r}. Must be a string.")

        # Define valid groups
        rna_syn = {"rna", "RNA", "genes", "transcripts"}
        prot_syn = {"adt", "protein", "prot", "proteins", "proteomics"}
        chrom_syn = {"chromatin", "atac"}

        seen = set()
        remap: dict[str, str] = {}
        for orig in list(self.data_dict.keys()):
            lname = orig.lower()
            if lname in rna_syn:
                std = "rna"
            elif lname in prot_syn:
                std = "protein"
            elif lname in chrom_syn:
                std = "chromatin"
            else:
                raise ValueError(
                    f"Invalid modality name '{orig}'. Must be one of rna, protein, or chromatin (or synonyms)."
                )
            if std in seen:
                raise ValueError(f"Duplicate modality '{std}' detected from key '{orig}'.")
            seen.add(std)
            remap[orig] = std

        # Rebuild dict with standardized keys
        new_dict: dict[str, torch.Tensor] = {}
        for orig, std in remap.items():
            new_dict[std] = self.data_dict[orig]
        self.data_dict = new_dict
        self.modalities = list(self.data_dict.keys())

    def get_cell_topic_dist(self) -> np.ndarray:
        """
        Get the cell-topic matrix Θ (C × K).

        Returns
        -------
        Θ : np.ndarray
            Cell-topic matrix, where C is the number of cells and K is the number of topics.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_feature_topic_dist(self, modality: str) -> np.ndarray | pd.DataFrame:
        """
        Get the feature-topic matrix Φ (K × G).

        Parameters
        ----------
        modality : str
            The name of the modality for which to retrieve the feature-topic matrix.

        Returns
        -------
        Φ : np.ndarray or pd.DataFrame
            Feature-topic matrix, where K is the number of topics and G is the number of features.
            If the modality has feature names, returns a DataFrame with those names.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def cross_modality_score(
        self,
        mod_a: str,
        mod_b: str,
        *,
        normalise: bool = True,
        return_df: bool = True,
    ) -> "np.ndarray | pd.DataFrame":
        """
        Compute SHARE-Topic–style cross-modal interaction matrix  P_{a,b}

        Parameters
        ----------
        model      : fitted topic model with the two accessors above
        mod_a      : modality name of *source* features  (e.g. 'rna')
        mod_b      : modality name of *target* features  (e.g. 'chromatin')
        normalise  : divide by global max so that scores ∈ [0,1]
        return_df  : return a DataFrame (keeps feature names) instead of ndarray

        Returns
        -------
        P  : shape (n_feat_a, n_feat_b) – interaction score between every
            feature of `mod_a` and every feature of `mod_b`
        """

        if self.n_modalities == 1:
            raise ValueError("This function is available only with more than one modality")
        # ------------------------------------------------------------------
        # 1.  Pull matrices from the model
        # ------------------------------------------------------------------
        Θ = np.asarray(self.get_cell_topic_dist())  # (C × K)
        Φa_raw = self.get_feature_topic_dist(mod_a)  # may be DataFrame, orientation varies
        Φb_raw = self.get_feature_topic_dist(mod_b)

        # normalise orientation to (K × G)
        def _normalize_phi(phi, n_topics):
            names = None
            if isinstance(phi, pd.DataFrame):
                if phi.shape[1] == n_topics and phi.shape[0] != n_topics:
                    # (features × topics) -> transpose to (topics × features)
                    names = phi.index.tolist()
                    phi = phi.values.T
                elif phi.shape[0] == n_topics:
                    names = phi.columns.tolist()
                    phi = phi.values
                else:
                    raise ValueError(
                        f"Unexpected feature-topic shape {phi.shape}; expected topics on one axis."
                    )
            else:
                phi = np.asarray(phi, dtype=float)
            return phi, names

        Φa, names_a = _normalize_phi(Φa_raw, Θ.shape[1])
        Φb, names_b = _normalize_phi(Φb_raw, Θ.shape[1])

        # ------------------------------------------------------------------
        # 2.  Normalise across *topics* for every feature   (λ*, φ*)
        # ------------------------------------------------------------------
        Φa /= Φa.sum(axis=0, keepdims=True) + 1e-12
        Φb /= Φb.sum(axis=0, keepdims=True) + 1e-12

        # ------------------------------------------------------------------
        # 3.  Average topic proportions across cells        ( s_t = 1/C Σ_c θ_ct )
        # ------------------------------------------------------------------
        Θ /= Θ.sum(axis=1, keepdims=True) + 1e-12  # θ*  (guarantees rows sum-to-1)
        s_t = Θ.mean(axis=0)  # shape (K,)

        # ------------------------------------------------------------------
        # 4.  Interaction matrix         P_{a,b} = Σ_t λ*_ta  φ*_tb  s_t
        #     → compute in two BLAS calls:   diag(s_t) · Φa  then   (Φa)^T · Φb
        # ------------------------------------------------------------------
        Φa_weighted = Φa * s_t[:, None]  # (K × G_a)
        P = Φa_weighted.T @ Φb  # (G_a × G_b)

        # ------------------------------------------------------------------
        # 5.  Optional global-max normalisation
        # ------------------------------------------------------------------
        if normalise and P.max() > 0:
            P /= P.max()

        if return_df and (names_a is not None) and (names_b is not None):
            P = pd.DataFrame(P, index=names_a, columns=names_b)

        return P

    def clear_metric_cache(self):
        """
        Clear the cached metrics.

        Call this method after retraining the model to ensure metrics are recomputed
        with the updated parameters.
        """
        self._cached_metrics = {}

    # ------------------------------------------------------------------
    # Model-specific metrics (abstract - must be implemented by subclasses)
    # ------------------------------------------------------------------
    def get_perplexity(self, **kwargs) -> float:
        """
        Compute perplexity (reconstruction quality).

        Lower is better. Perplexity = exp(-log_likelihood / N_tokens)

        Returns
        -------
        float
            Perplexity score
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_likelihood_per_modality(self, **kwargs) -> dict[str, float]:
        """
        Compute log-likelihood for each modality separately.

        Higher is better.

        Returns
        -------
        dict[str, float]
            Dictionary mapping modality names to log-likelihood values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_perplexity_per_modality(self, **kwargs) -> dict[str, float]:
        """
        Compute perplexity for each modality separately.

        Lower is better. Perplexity = exp(-log_likelihood / N_tokens)

        Returns
        -------
        dict[str, float]
            Dictionary mapping modality names to perplexity values
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_modality_weights(self, **kwargs) -> "pd.DataFrame | dict[str, np.ndarray]":
        """
        Get normalized mixing weights showing how much each modality contributes to topic assignments.

        Only applicable for multimodal models with mixture-of-experts or similar architectures.
        Returns weights in range [0, 1] that sum to 1 per cell.
        Higher weight = model relies more on that modality for inferring topics.

        Returns
        -------
        pd.DataFrame or dict[str, np.ndarray]
            Normalized mixing weights for each cell and modality.
            DataFrame: cells × modalities
            Dict: modality name → weights array
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    # ------------------------------------------------------------------
    # Concrete metrics (implemented using abstract get_cell_topic_dist and get_feature_topic_dist)
    # ------------------------------------------------------------------
    def get_entropy(self, normalised: bool = True) -> float:
        """
        Compute mean entropy of cell-topic distributions.

        Higher entropy means topics are more evenly distributed across cells.
        This measures the uncertainty in topic assignments per cell.

        Parameters
        ----------
        normalised : bool
            Whether to normalize cell-topic distributions before computing entropy.
            If True, ensures distributions sum to 1 (default: True).

        Returns
        -------
        float
            Mean entropy across all cells
        """
        cache_key = f"entropy_normalised={normalised}"
        if cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        # Get cell-topic matrix Θ (C × K)
        theta = self.get_cell_topic_dist()

        if normalised:
            # Normalize to ensure rows sum to 1
            theta = theta / (theta.sum(axis=1, keepdims=True) + 1e-12)

        # Compute entropy per cell: -Σ_k θ_ck * log(θ_ck)
        # Use np.log with clipping to avoid log(0)
        entropy_per_cell = -(theta * np.log(np.clip(theta, 1e-8, None))).sum(axis=1)

        # Return mean entropy across cells
        result = float(entropy_per_cell.mean())

        # Cache result
        self._cached_metrics[cache_key] = result
        return result

    def get_topic_diversity(self, modality: str | None = None) -> float:
        """
        Compute topic diversity as average pairwise cosine distance.

        Higher values indicate more distinct topics. This metric measures how
        different the topic-feature distributions are from each other.

        Parameters
        ----------
        modality : str, optional
            If provided, compute diversity for this specific modality's
            feature-topic distribution. If None, compute diversity averaged
            across all modalities (default: None).

        Returns
        -------
        float
            Average pairwise cosine distance between topic distributions (0-1).
            Higher = more diverse/distinct topics.
        """
        if modality is not None:
            cache_key = f"topic_diversity_modality={modality}"
            if cache_key in self._cached_metrics:
                return self._cached_metrics[cache_key]

            # Get feature-topic dist for specific modality (K × F)
            phi = self.get_feature_topic_dist(modality)
            phi = np.asarray(phi, dtype=float)

            # Normalize topics to unit vectors for cosine similarity
            phi_norm = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-12)

            # Compute pairwise cosine similarity
            cosine_sim = phi_norm @ phi_norm.T  # (K × K)

            # Extract upper triangle (excluding diagonal)
            K = phi.shape[0]
            upper_tri_indices = np.triu_indices(K, k=1)
            similarities = cosine_sim[upper_tri_indices]

            # Diversity = 1 - average similarity
            result = float(1 - similarities.mean())

            # Cache result
            self._cached_metrics[cache_key] = result
            return result
        else:
            # Average across all modalities
            cache_key = "topic_diversity_all_modalities"
            if cache_key in self._cached_metrics:
                return self._cached_metrics[cache_key]

            diversities = []
            for mod in self.modalities:
                diversities.append(self.get_topic_diversity(modality=mod))

            result = float(np.mean(diversities))

            # Cache result
            self._cached_metrics[cache_key] = result
            return result

    def get_top_features_per_topic(
        self,
        modality: str,
        n_features: int = 10,
        return_scores: bool = False,
    ) -> dict[str, list[str]] | dict[str, list[tuple[str, float]]]:
        """
        Get top N features for each topic in a specific modality.

        Parameters
        ----------
        modality : str
            Modality name (e.g., 'rna', 'protein', 'chromatin')
        n_features : int
            Number of top features to return per topic (default: 10)
        return_scores : bool
            If True, return (feature_name, score) tuples.
            If False, return feature names only (default: False).

        Returns
        -------
        dict[str, list[str]] or dict[str, list[tuple[str, float]]]
            Dictionary mapping topic names (e.g., 'topic_0') to lists of
            top feature names or (feature_name, score) tuples.
        """
        cache_key = f"top_features_{modality}_n={n_features}_scores={return_scores}"
        if cache_key in self._cached_metrics:
            return self._cached_metrics[cache_key]

        # Get feature-topic distribution Φ (K × F)
        phi = self.get_feature_topic_dist(modality)

        # Convert to array if DataFrame
        if isinstance(phi, pd.DataFrame):
            # DataFrame returned as (features × topics); transpose to (topics × features)
            feature_names = phi.index.tolist()
            phi_array = phi.values.T  # (K × F)
        else:
            phi_array = np.asarray(phi)
            feature_names = [f"feature_{i}" for i in range(phi_array.shape[1])]

        K = phi_array.shape[0]
        result = {}

        for k in range(K):
            # Get top n_features indices for topic k
            top_indices = np.argsort(phi_array[k, :])[-n_features:][::-1]

            if return_scores:
                # Return list of (feature_name, score) tuples
                result[f"topic_{k}"] = [
                    (feature_names[i], float(phi_array[k, i])) for i in top_indices
                ]
            else:
                # Return list of feature names only
                result[f"topic_{k}"] = [feature_names[i] for i in top_indices]

        # Cache result
        self._cached_metrics[cache_key] = result
        return result
