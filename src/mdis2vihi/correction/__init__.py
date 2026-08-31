"""Separate hollow-contrast correction layer (`unionbb0.8`).

Applied to the delivered mosaic, it raises the brightness contrast of catalogued
hollows, which the model under-predicts because the quality filter keeps almost no
hollow-floor footprint in training:

    output = deliverable(x) + g(lon, lat, reflectance) * [ c(x) @ B.T ]
    g      = 0.797 * [ (Thomas 2016 union HORNET 0.8) inter bright+blue ]

The deliverable IS the output of the fixed base model: it is not recomputed, the correction
is added to it. Outside the selection the output is therefore identical to the deliverable
*by construction*.

Where the form comes from. `c(x) @ B.T` is the basis-function formulation of spectral
super-resolution, `M ~ v0 + sum_j v_j p_j` in the review of He et al. (2023, Inf. Fusion,
eq. 4): k fixed spectral basis functions with per-pixel coefficients, a lineage running
back to the PCA and Karhunen-Loeve reconstructions of reflectance spectra (Agahian et al.
2008; Eslahi et al. 2009). Here `v0` is the frozen deliverable rather than a constant, the
coefficients come from a small network rather than a linear solve, and `B` is fitted on
the residual instead of on the spectra. Constraining a multivariate response to a fixed
low-rank subspace is reduced-rank regression (Anderson 1951; Izenman 1975); adding a
correction to a frozen predictor is residual learning, one of the ten strategies the same
review enumerates. The assembly is this project's, the ingredients are not.

`selection`: catalogues, rasterised spatial stage, bright+blue spectral stage.
`layer`   : the trained correction network, the sparse layer (5 numbers per corrected
             pixel), and how it is applied.
`residual`: the training side: shape basis, gated modules, Lightning wrapper. Used by
             `scripts/08_train_residual.py` to produce the checkpoint `layer` then freezes.
"""