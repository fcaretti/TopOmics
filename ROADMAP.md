# TopOmics

The basic idea of the package is to be able to run topic modeling on any kind of (optionally spatial) multiomics data.
In an analogy with standard topic models, cells/spots are documents, genes/proteins/chromatin accessibility areas are words, different modalities correspond to different "chapters" and the spatial information can be converted to a neighborhood graph, equivalent (although with very different statistics) to a citation network.

The main content of the models is going to go in src/topomics/models, with some stuff that can go in src/topomics/module, especially NN architectures for the amortized models.

So basically every model is going to inherit the class BaseTopicModel, that contains functions for the standard evaluation of topic models.

# Next To-Do:
- [X] Copy the AmortizedLDA implementation in SCVI
- [ ] Make AmortizedLDA run with multiple modalities
  - [X] Insert a Mixture-of-Experts to begin with (or PoE?)
  - [X] Make it setup with multiple modalities
  - [ ] Test on the same datasets of SHARE-Topic
- [ ] Implement evaluation metrics in BaseTopicModel
    - [X] Cross-Modality correlations (already implemented)
    - [X] **Perplexity**
      - Implementation: `exp(-ELBO / total_counts)` for AmortizedLDA
      - Abstract method in BaseTopicModel, implemented in each model class
      - Cached after first computation
    - [X] **Perplexity (per-modality)**
      - Implementation: `exp(-log_lik_m / N_tokens_m)` for each modality
      - Extracts per-modality log-probs using `poutine.trace` in Pyro models
      - Returns dict mapping modality names to perplexity values
      - Cached after first computation
    - [X] **Likelihood (per-modality)**
      - Implementation: Extract observation site log-probs per modality from model trace
      - Uses `poutine.trace` to get `feature_counts_{m}` log-probabilities
      - Returns dict mapping modality names to log-likelihood values
      - Cached after first computation
    - [X] **Entropy**
      - Implementation: Mean entropy of cell-topic distributions: `mean(-Σ_k θ_ck * log(θ_ck))`
      - Measures how evenly topics are distributed within each cell
      - Implemented in BaseTopicModel using `get_cell_topic_dist()`
      - Cached after first computation
    - [X] **Topic Diversity**
      - Implementation: Average pairwise cosine distance between topic-feature distributions
      - Higher values indicate more distinct topics
      - Metric: `1 - mean(cosine_similarity)` between all topic pairs
      - Future: Add KL divergence alternative
      - Implemented in BaseTopicModel using `get_feature_topic_dist()`
      - Cached after first computation
    - [X] **Top Features Per Topic**
      - Implementation: Extract top N features per topic from feature-topic distributions
      - Default: n=10 features, returns feature names only
      - Optional: `return_scores=True` to get (feature, score) tuples
      - Implemented in BaseTopicModel using `get_feature_topic_dist()`
    - [X] **Metric Caching System**
      - All metrics are automatically cached in `model._cached_metrics` dict after first computation
      - Cache keys include method parameters (e.g., `"entropy_normalised=True"`)
      - Provides `model.clear_metric_cache()` to invalidate after retraining
      - Avoids expensive recomputations when calling same metric multiple times
    - [ ] Coherence with paths (future work)
- [X] Add support for spatial data in AmortizedLDA
  - [X] Update `MultimodalAmortizedLDA.setup_anndata/setup_mudata/from_mudata` in `src/topomics/models/amortizedLDA.py` to accept `spatial_key`/`spatial_modality_keys`, call the helper, and store resolved graph + metadata in `adata.uns` / `mdata.uns`
  - [X] Let `MultimodalAmortizedLDA.__init__` (same file) pick up the stored graph handle and propagate adjacency to the Pyro module; error if the requested `spatial_key` is missing
  - [X] Add GCN encoder branch in `MultimodalLDAPyroGuide` (in `src/topomics/module/_amortizedLDA.py`) that is automatically used when spatial graphs are present (no user flag), otherwise fall back to MLP encoders
  - [X] Convert stored CSR adjacency from `adata.uns["_spatial_graph"]` / `["_spatial_graphs"]` into device-ready `edge_index` (+ weights) tensors for the guide; support modality-specific graphs
  - [X] Ensure the data loader / guide path passes adjacency to the GCN (full-batch or documented constraint) and raises if spatial graph is missing when required
  - [X] Add dependency handling for GCN backend (`torch_geometric` or minimal torch-sparse stack) under the spatial extra in `pyproject.toml`
  - [X] Add tests for GCN path (toy graph)
  - [ ] Implement `get_topic_by_location` (in `MultimodalAmortizedLDA`, same file) to aggregate θ over spatial neighborhoods or coordinates using the chosen `spatial_key`
  - [X] **Implement train/test set in the presence of graph (semi-supervised learning)**
    - [X] Track validation ELBO during training
  - [X] Use GAT instead of GCN
- [ ] Implement different distribution choices
  - [X] Standard Gamma-Poisson
  - [X] Standard Dirichlet
  - [X] Horseshoe prior Gamma-Poisson
  - [X] Horseshoe prior Dirichlet

  - [X] **Add feature background term (bg) - STAMP style**


- [X] **Add Bernoulli likelihood for binary data (ATAC-seq, methylation)**
  - [X] Rescaling parameter due to library size

- [X] **Add method to quantify modality importance/contribution**
  - [X] Implemented `get_modality_weights()` in BaseTopicModel (abstract) and MultimodalAmortizedLDA
  - [X] Extracts normalized mixing weights from Mixture-of-Experts architecture
  - [X] Shows which modality the model trusts/relies on more for topic inference
  - [X] Handles three weight modes:
    - "equal": uniform weights (1/M for each modality)
    - "universal": single learned weight per modality (same across all cells)
    - "cell": per-cell learned weights
  - [X] Returns DataFrame (cells × modalities) or dict format
  - [X] Automatically cached after first computation
  - Implementation: Extracts `mod_w` from guide, applies `masked_softmax` to account for missing modalities

- [X] **Add per-cell topic entropy regularization** (PARTIALLY IMPLEMENTED - needs renaming)

  **Current Status:** Implemented but targets wrong problem (encourages all cells to be diverse, not different from each other)

  **What's implemented:**
  - Parameter: `entropy_weight` (will be renamed to `cell_topic_entropy_weight`)
  - Objective: `ELBO + λ * Σ_n H(θ_n)` where `H(θ_n) = -Σ_k θ_n,k log θ_n,k`
  - Effect: Each cell uses multiple topics uniformly
  - Problem: Makes cell collapse WORSE (all cells become equally diverse)

  **Implemented in:**
  - [X] `src/topomics/models/amortizedLDA.py` - parameter passing
  - [X] `src/topomics/module/_amortizedLDA.py` - computation and pyro.factor
  - [X] `tests/test_amortized.py` - comprehensive tests

  **TODO - Rename existing implementation:**
  - [ ] Rename `entropy_weight` → `cell_topic_entropy_weight` everywhere
  - [ ] Update docstrings to clarify this encourages per-cell diversity (not cell-to-cell differences)
  - [ ] Keep tests but update names

---

- [ ] **Add topic variance regularization to prevent cell collapse**

  **Goal:** Prevent all cells from having identical topic distributions by encouraging topics to be used differently across cells.

  **Problem being solved:**
  - Current issue: All cells collapse to same distribution (e.g., all cells = [40% topic 4, 20% topic 2, ...])
  - We want: Different cells specialize in different topics

  **Mathematical formulation:**
  - For each topic k, compute variance of usage across cells in batch: `Var(θ_:,k) = (1/B) Σ_n (θ_n,k - mean_k)²`
  - Objective: `maximize ELBO + λ * Σ_k Var(θ_:,k)`
  - λ = `topic_variance_weight` (hyperparameter)
  - High variance = topic k heavily used by some cells, rarely by others = diverse cell population

  **Implementation Steps:**

  1. **Add `topic_variance_weight` hyperparameter** [Files: `src/topomics/models/amortizedLDA.py`]
     - [X] Add `topic_variance_weight: float = 0.0` parameter to `MultimodalAmortizedLDA.__init__()` (line ~158, after entropy_weight)
     - [X] Pass to module: store in `self.module.topic_variance_weight`
     - [X] Add docstring:
       ```
       topic_variance_weight
           Weight for topic variance regularization (default: 0.0).
           When > 0, encourages different cells to use different topics, preventing
           cell collapse where all cells have identical topic distributions.
           Objective: ELBO + λ * Σ_k Var(θ_:,k) where Var is variance across cells.
           Typical values: 1.0-10.0 (higher than entropy_weight because variance is smaller).
       ```

  2. **Store topic_variance_weight in Pyro module and guide** [Files: `src/topomics/module/_amortizedLDA.py`]
     - [X] Add parameter to `MultimodalAmortizedLDAPyroModule.__init__()` (line ~967)
     - [X] Store as `self.topic_variance_weight = topic_variance_weight`
     - [X] Pass to guide: Add parameter to `MultimodalLDAPyroGuide.__init__()` (line ~561)
     - [X] Store in guide as `self.topic_variance_weight = topic_variance_weight`
     - [X] Initialize tracking: `self._last_topic_variance = None`

  3. **Compute topic variance in guide.forward()** [Files: `src/topomics/module/_amortizedLDA.py`]
     - [X] Location: In `MultimodalLDAPyroGuide.forward()` OUTSIDE the `pyro.plate("cells", ...)` context
     - [X] After computing `theta = F.softmax(log_theta, dim=-1)` (if entropy regularization is active)
     - [X] **If entropy_weight == 0, still need to compute theta:**
       ```python
       # Inside pyro.plate("cells", ...) after sampling log_theta
       if self.topic_variance_weight > 0:
           # Compute theta if not already computed
           if self.entropy_weight == 0:
               theta = F.softmax(log_theta, dim=-1)  # (B, K)

           # Compute variance of each topic across cells
           topic_variance = theta.var(dim=0)  # (K,) - variance for each topic
           total_variance = topic_variance.sum()  # scalar

           # Store mean for logging (convert to mean across topics for interpretability)
           self._last_topic_variance = topic_variance.mean().detach()

           # Add variance bonus to ELBO (NOT inside kl_weight scale, NOT per-cell)
           pyro.factor("topic_variance_bonus", self.topic_variance_weight * total_variance, has_rsample=True)
       ```

  4. **IMPORTANT: Extensive formulation consideration**
     - [X] **Problem:** Variance is computed per-batch, not extensive over full dataset
     - [X] **Solution:** Do NOT place factor inside `pyro.plate` - the variance is already a batch-level statistic
     - [X] Place `pyro.factor()` OUTSIDE the plate, after the plate context closes
     - [X] Pyro will NOT automatically scale this (good - variance is already a batch statistic)
     - [X] **Code structure:**
       ```python
       with pyro.plate("cells", size=n_obs or self.n_obs, subsample_size=B):
           # ... sample log_theta ...
           pass  # exit plate context

       # OUTSIDE plate: compute variance regularization
       if self.topic_variance_weight > 0:
           theta = F.softmax(log_theta, dim=-1)  # (B, K)
           topic_variance = theta.var(dim=0)  # (K,)
           self._last_topic_variance = topic_variance.mean().detach()
           pyro.factor("topic_variance_bonus", self.topic_variance_weight * topic_variance.sum(), has_rsample=True)
       ```

  5. **Add API methods** [Files: `src/topomics/module/_amortizedLDA.py`, `src/topomics/models/amortizedLDA.py`]
     - [X] **Module level** (in `MultimodalAmortizedLDAPyroModule`):
       - Add `get_last_topic_variance()` → returns `self.guide._last_topic_variance`
       - Add `get_topic_variance(x, libs)` → computes variance from data
     - [X] **Model level** (in `MultimodalAmortizedLDA`):
       - Add `get_topic_variance_weight()` → returns weight
       - Add `get_last_topic_variance()` → returns last computed variance
       - Add `get_topic_variance()` → computes per-topic variance across all cells

  6. **Update validation logging** [Files: `src/topomics/utils/training_plan.py`]
     - [X] In `validation_step()`, after entropy logging, add:
       ```python
       if hasattr(self.module, 'guide') and hasattr(self.module.guide, '_last_topic_variance'):
           variance = self.module.guide._last_topic_variance
           if variance is not None and self.module.topic_variance_weight > 0:
               self.log("topic_variance_mean_val", variance, ...)
               output["topic_variance_mean_val"] = float(variance)
       ```
     - [X] Update `on_validation_epoch_end()` to aggregate variance values

  7. **Testing** [Files: `tests/test_amortized.py`]
     - [X] Test parameter storage propagates correctly
     - [X] Test that `topic_variance_weight > 0` leads to different cell distributions
     - [X] Compute pairwise cosine similarity between cells:
       ```python
       # Should see LOWER similarity with topic_variance_weight > 0
       theta = model.get_latent_representation()
       similarity = cosine_similarity(theta)
       mean_similarity = similarity.mean()  # Lower = more diverse cells
       ```
     - [X] Test with MuData and spatial data
     - [X] Test that both regularizations can be used together

  8. **Documentation updates**
     - [X] Update `ROADMAP.md` to mark this as completed
     - [X] Add comparison in docstring explaining difference:
       - `cell_topic_entropy_weight`: Each cell uses many topics uniformly
       - `topic_variance_weight`: Different cells use different topics (prevents cell collapse)

  **Key implementation notes:**
  - Variance must be computed OUTSIDE `pyro.plate` (it's a batch-level statistic, not per-cell)
  - Typical values: 1.0-10.0 (higher than entropy because variance values are smaller)
  - Can be combined with `cell_topic_entropy_weight` for both effects
  - Variance computation: O(B*K) - very efficient

  **Expected impact:**
  - Without: All cells have identical distributions [0.4, 0.2, 0.15, 0.25]
  - With (λ=5.0): Cells specialize - Cell 1 = [0.7, 0.1, 0.1, 0.1], Cell 2 = [0.1, 0.6, 0.2, 0.1], etc.
  - Metrics: Higher topic variance, lower mean pairwise cell similarity
