# Development notes

- Nahual is a light layer of abstraction that defines the interface to integrate with others.
- Nahual is supposed to have as few dependencies as possible. The only dependencies are requests, numpy, pooch (to download and cache data) and developer dependencies.

The models that are supported are:
- Images to features (2D.float -> 1D.float)
  - DINO (Self-supervised learning)
  - cp_measure (engineered features)
- Images to masks (2D.float -> 2D.int)
  - Cellpose
  - Baby
- Images + masks -> dictionary of labels (2D.float,2D.int -> dict of 1D.int)
  - trackastra
- masks -> dictionary of labels (2D.int->dict of 1D.int)
  - Baby

