"""mdis2vihi: pixel-wise spectral reconstruction, MDIS 8 bands -> VIHI-like 231 bands.

Sub-packages
------------
data       readers for the MDIS mosaic and the MASCS/VIRS sidecar files.
models     the plain ``nn.Module`` (``SpectralMLP``), Lightning-agnostic.
lit        the ``LightningModule`` wrapping it with loss, metrics and steps.
inference  the tiled, streamed mosaic writer (its own problem, kept separate).
eval       the single source of truth for metrics and spectral parameters.
correction   the optional separate hollow-contrast correction layer.

The package is used from a src/ layout without installation: the scripts insert
``<repo>/src`` on ``sys.path`` themselves, so ``python scripts/03_train_final.py``
works from a bare checkout.
"""

__version__ = "1.0.0"