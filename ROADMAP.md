# Omics-Topic

The basic idea of the package is to be able to run topic modeling on any kind of (optionally spatial) multiomics data.
In an analogy with standard topic models, cells/spots are documents, genes/proteins/chromatin accessibility areas are words, different modalities correspond to different "chapters" and the spatial information can be converted to a neighborhood graph, equivalent (although with very different statistics) to a citation network.

The main content of the models is going to go in src/omics_topic/models, with some stuff that can go in src/omics_topic/module, especially NN architectures for the amortized models.

So basically every model is going to inherit the class BaseTopicModel, that contains functions for the standard evaluation of topic models. 

# Next To-Do:
- [X] Copy the AmortizedLDA implementation in SCVI
- [ ] Make AmortizedLDA run with multiple modalities
  - [X] Insert a Mixture-of-Experts to begin with (or PoE?)
  - [X] Make it setup with multiple modalities
  - [ ] Test on the same datasets of SHARE-Topic
- [ ] Implement evaluation metrics
    - [ ] Perplexity
    - [ ] Entropy
    - [ ] Coherence with paths
    - [ ] Cross-Modality correlations
- [ ] Add support for spatial data in AmortizedLDA
  - [X] Update `MultimodalAmortizedLDA.setup_anndata/setup_mudata/from_mudata` in `src/omics_topic/models/amortizedLDA.py` to accept `spatial_key`/`spatial_modality_keys`, call the helper, and store resolved graph + metadata in `adata.uns` / `mdata.uns`
  - [X] Let `MultimodalAmortizedLDA.__init__` (same file) pick up the stored graph handle and propagate adjacency to the Pyro module; error if the requested `spatial_key` is missing
  - [X] Add GCN encoder branch in `MultimodalLDAPyroGuide` (in `src/omics_topic/module/_amortizedLDA.py`) that is automatically used when spatial graphs are present (no user flag), otherwise fall back to MLP encoders
  - [X] Convert stored CSR adjacency from `adata.uns["_spatial_graph"]` / `["_spatial_graphs"]` into device-ready `edge_index` (+ weights) tensors for the guide; support modality-specific graphs
  - [X] Ensure the data loader / guide path passes adjacency to the GCN (full-batch or documented constraint) and raises if spatial graph is missing when required
  - [X] Add dependency handling for GCN backend (`torch_geometric` or minimal torch-sparse stack) under the spatial extra in `pyproject.toml`
  - [X] Add tests for GCN path (toy graph)
  - [ ] Implement `get_topic_by_location` (in `MultimodalAmortizedLDA`, same file) to aggregate θ over spatial neighborhoods or coordinates using the chosen `spatial_key`
  - [X] **Implement train/test set in the presence of graph (semi-supervised learning)**
    - [X] Track validation ELBO during training
  - [ ] Use GAT instead of GCN
- [ ] Implement different distribution choices
  - [X] Standard Gamma-Poisson
  - [X] Standard Dirichlet
  - [ ] Horseshoe prior Gamma-Poisson
  - [ ] Horseshoe prior Dirichlet
- [ ] **Add entropy term to avoid topic collapse**
- [ ] **Add library size param in Gamma-Poisson**
