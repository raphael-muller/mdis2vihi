"""Shared evaluation: spectral parameters and metrics.

Single source of truth, so nothing copy-pastes `refl`, `slope` or `r_key`.
"""

from mdis2vihi.eval.params import (
    GRID, PARAMS, WL, FAMILIES, R_KEY_BALANCED,
    refl, slope, poly2_curv, uv_downturn, uv_depth, spectral_params,
)
from mdis2vihi.eval.metrics import (
    pearson, ccc, ols_slope, rmse, mae, mrae, sam_deg, sga_deg, sid,
    per_band, param_fidelity, r_key, knn_floor, reliability_ceiling, metric_set,
)

__all__ = [
    "GRID", "PARAMS", "WL", "FAMILIES", "R_KEY_BALANCED",
    "refl", "slope", "poly2_curv", "uv_downturn", "uv_depth", "spectral_params",
    "pearson", "ccc", "ols_slope", "rmse", "mae", "mrae", "sam_deg", "sga_deg", "sid",
    "per_band", "param_fidelity", "r_key", "knn_floor", "reliability_ceiling", "metric_set",
]