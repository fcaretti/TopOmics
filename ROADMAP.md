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
- [ ] Add support for spatial data in AmortizedLDA
  - [ ] Make it check if a connectivity graph is present
  - [ ] Implement GCN encoder
  - [ ] Implement function to get topic x location/coordinate
- [ ] Implement different distribution choices
  - [ ] Standard Gamma-Poisson
  - [ ] Standard Dirichlet
  - [ ] Horseshoe prior Gamma-Poisson
  - [ ] Horseshoe prior Dirichlet