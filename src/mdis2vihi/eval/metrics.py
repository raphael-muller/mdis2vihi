"""Reconstruction-quality metrics, several perspectives rather than one scalar.

  magnitude : RMSE, MAE, MRAE       MRAE ranks the NTIRE spectral reconstruction
                                    challenges (Arad et al., CVPRW 2018/2020/2022)
  shape     : SAM (Kruse 1993), SID (Chang 2000), SGA = SAM on the band-to-band
              derivative, sensitive to slope rather than level
  per band  : Pearson r, OLS slope, Lin's CCC (Lin 1989), bias
  domain    : param_fidelity over the parameters of `mdis2vihi.eval.params`
  bounds    : knn_floor, reliability_ceiling

All functions are NaN-aware: MASCS targets carry NaN where a spectrum does not cover
[300, 1450] nm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mdis2vihi.eval.params import (
    PARAMS, R_KEY_BALANCED, spectral_params,
)

EPS = 1e-8


# ----------------------------------------------------------------- correlation
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-aware Pearson r between two 1-D arrays."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < EPS or b.std() < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _slope_ccc(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """NaN-aware (OLS slope of b on a, Lin's CCC), shared by per_band and
    param_fidelity so the two never drift apart."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, np.nan
    am, bm = a[m], b[m]
    var_a, var_b = am.var(), bm.var()
    cov = ((am - am.mean()) * (bm - bm.mean())).mean()
    slope = float(cov / var_a) if var_a > 0 else np.nan
    denom = var_a + var_b + (am.mean() - bm.mean()) ** 2
    return slope, (float(2.0 * cov / denom) if denom > 0 else np.nan)


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-aware Lin's concordance correlation coefficient between two 1-D arrays.

    Unlike Pearson r, CCC is NOT affine-invariant: it measures agreement with the
    1:1 line, so a prediction that is perfectly correlated but compressed in
    dynamic range (slope != 1) or offset (bias != 0) is penalised."""
    return _slope_ccc(a, b)[1]


def ols_slope(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-aware OLS slope of b on a. < 1 = the prediction compresses the dynamic."""
    return _slope_ccc(a, b)[0]


# ----------------------------------------------------------------- magnitude
def _finite_mask(Yp, Yt):
    return np.isfinite(Yp) & np.isfinite(Yt)


def rmse(Yp, Yt):
    m = _finite_mask(Yp, Yt)
    return float(np.sqrt(((Yp[m] - Yt[m]) ** 2).mean()))


def mae(Yp, Yt):
    m = _finite_mask(Yp, Yt)
    return float(np.abs(Yp[m] - Yt[m]).mean())


def mrae(Yp, Yt, eps=1e-6):
    """Mean Relative Absolute Error (NTIRE primary): |p-t|/|t|, robust to amplitude."""
    m = _finite_mask(Yp, Yt) & (np.abs(Yt) > eps)
    return float((np.abs(Yp[m] - Yt[m]) / np.abs(Yt[m])).mean())


# ----------------------------------------------------------------- shape
def _angle_deg(A, B):
    """Per-row angle (deg) between A and B, NaN-aware. Shapes (N, K)."""
    m = np.isfinite(A) & np.isfinite(B)
    A0 = np.where(m, A, 0.0); B0 = np.where(m, B, 0.0)
    dot = (A0 * B0).sum(-1)
    norm = np.linalg.norm(A0, axis=-1) * np.linalg.norm(B0, axis=-1)
    cos = np.clip(dot / (norm + EPS), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def sam_deg(Yp, Yt):
    """Spectral Angle Mapper per sample (deg)."""
    return _angle_deg(Yp, Yt)


def sga_deg(Yp, Yt):
    """Spectral Gradient Angle: SAM on the band-to-band derivative (shape of slope)."""
    return _angle_deg(np.diff(Yp, axis=-1), np.diff(Yt, axis=-1))


def sid(Yp, Yt, eps=1e-8):
    """Spectral Information Divergence per sample (symmetric KL of band distributions)."""
    m = _finite_mask(Yp, Yt)
    P = np.where(m, np.clip(Yp, eps, None), 0.0)
    Q = np.where(m, np.clip(Yt, eps, None), 0.0)
    Pn = P / (P.sum(-1, keepdims=True) + EPS)
    Qn = Q / (Q.sum(-1, keepdims=True) + EPS)
    pos = (Pn > 0) & (Qn > 0)
    term = np.where(pos, Pn * np.log((Pn + EPS) / (Qn + EPS)) + Qn * np.log((Qn + EPS) / (Pn + EPS)), 0.0)
    return term.sum(-1)


# ----------------------------------------------------------------- per band
def per_band(Yp, Yt):
    """Per-band Pearson r, OLS slope, Lin's CCC and bias (pred-true), as a DataFrame
    indexed by band. `slope`/`ccc` expose the amplitude calibration r is blind to."""
    from mdis2vihi.eval.params import GRID
    n = Yt.shape[1]
    r = np.array([pearson(Yt[:, b], Yp[:, b]) for b in range(n)])
    sc = np.array([_slope_ccc(Yt[:, b], Yp[:, b]) for b in range(n)])
    bias = np.array([np.nanmean(Yp[:, b] - Yt[:, b]) for b in range(n)])
    return pd.DataFrame({"wavelength_nm": GRID, "pearson_r": r,
                         "slope": sc[:, 0], "ccc": sc[:, 1], "bias": bias})


# ----------------------------------------------------------------- domain
def param_fidelity(Yp, Yt) -> pd.DataFrame:
    """Per-parameter r / slope / ccc / bias / rel_bias (truth vs pred): the domain-side
    audit, written by the pipeline as `runs/final/eval/final_param_fidelity.csv`.
    `slope` (OLS pred~target) and `ccc` (Lin's concordance) capture the absolute
    calibration that Pearson r is blind to (r is affine-invariant: a model predicting
    half the true dynamic everywhere still scores r = 1)."""
    pt, pp = spectral_params(Yt), spectral_params(Yp)
    rows = []
    for name in PARAMS:
        t, p = pt[name], pp[name]
        m = np.isfinite(t) & np.isfinite(p)
        bias = float(np.median(p[m] - t[m])) if m.any() else np.nan
        med_t = float(np.median(t[m])) if m.any() else np.nan
        slope_, ccc_ = _slope_ccc(t, p)
        rows.append({"param": name, "pearson_r": pearson(t, p), "slope": slope_,
                     "ccc": ccc_, "bias": bias,
                     "rel_bias": bias / med_t if med_t not in (0.0, np.nan) else np.nan,
                     "median_target": med_t})
    return pd.DataFrame(rows)


def r_key(Yp, Yt, pt=None, pp=None, metric: str = "r"):
    """Mean correlation over `R_KEY_BALANCED` (one representative per physical family,
    spanning UV->NIR). Returns (mean, {param: value}).

    `metric` ∈ {r, ccc}. "r" = Pearson (shape agreement, blind to compression);
             "ccc" = Lin's concordance (agreement with the 1:1 line). The two can
             disagree in SIGN on the same comparison when a change trades
             correlation for restored dynamic range, so report both.

    Pass precomputed `pt`/`pp` (= spectral_params of target/pred) to avoid
    recomputing them."""
    if metric not in ("r", "ccc"):
        raise ValueError(f"metric must be 'r' or 'ccc', got {metric!r}")
    names = R_KEY_BALANCED
    if pt is None:
        pt = spectral_params(Yt)
    if pp is None:
        pp = spectral_params(Yp)
    fn = pearson if metric == "r" else ccc
    rs = {n: fn(pt[n], pp[n]) for n in names}
    vals = [0.0 if not np.isfinite(v) else v for v in rs.values()]
    return float(np.mean(vals)), rs


# ----------------------------------------------------------------- floor
def knn_floor(Xin, Y, k=5, device=None):
    """Nearest-neighbour estimate of the irreducible error `E_x[Var(y | x)]`.

    A 665 m MDIS pixel maps to a footprint of a few km, so footprints with nearly the
    same 8-band input carry genuinely different spectra: the achievable MSE is bounded
    below by the target variance in a small input neighbourhood. Averaging that variance
    over the `k` nearest neighbours estimates it. Pure torch, no scikit-learn.

    **The neighbourhood is in input space, and that is what makes this a bound.** The
    model is a deterministic function of the input vector, so two footprints with the
    same input necessarily receive the same prediction and the spread of their targets is
    irreducible for it. Neighbours picked by ground distance instead do not bound this
    model, since two spots a few km apart usually have different MDIS inputs which the
    model is free to map to different spectra; that variant measures a different thing,
    namely what a spatially aware model could hope to explain, and it is reported next to
    this one in `runs/final/eval/final_floor_variants.csv` (see `scripts/05_eval_final.py`
    for the six neighbourhoods and their median separation). On the delivered test split
    the geographic neighbours sit 15.7 km away and belong to the same observation three
    times out of four, so that floor mostly measures along-track repeatability: forbidding
    same-observation neighbours raises it from 2.92e-5 to 3.34e-5, above this one.

    The value is convention-dependent. Two biases pull opposite ways: `np.nanvar` uses
    ddof = 0, biasing each neighbourhood low by (k-1)/k (20 % at k = 5), while the finite
    neighbourhood width biases it up. On the delivered test split the raw estimate runs
    2.46e-5 (k=3) to 4.05e-5 (k=50), against 3.68e-5 to 4.14e-5 debiased; the ratio to the
    model MSE therefore runs 1.89 to 1.15 raw, 1.26 to 1.12 debiased. The quoted
    **1.557x is k = 5, ddof = 0, Euclidean distance on the 8 raw I/F bands** (1.552
    standardised, 1.440 on the 9 model inputs). Read it as "close to the bound, not at
    zero", not as a constant.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(np.ascontiguousarray(Xin)).double().to(device)
    idx = torch.cdist(Xt, Xt).topk(k + 1, largest=False).indices[:, 1:].cpu().numpy()
    return float(np.nanmean(np.nanvar(Y[idx], axis=1)))


def reliability_ceiling(Xin, Yt, Yp, params=None, device=None):
    """How much of each parameter is knowable from the input, and how much of it the
    model reaches.

    Two MASCS spectra that the 8 MDIS bands cannot tell apart are two measurements of the
    same point. The correlation of a parameter between a spectrum and its nearest
    neighbour in input space is that parameter's reliability; correction for attenuation
    (Spearman 1904) then caps any predictor at `sqrt(reliability)`. Same idea as
    `knn_floor`, per parameter and in correlation units.

    Returns: param, reliability, ceiling, model_r, frac_of_ceiling.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(np.ascontiguousarray(Xin)).double().to(device)
    nb = torch.cdist(Xt, Xt).topk(2, largest=False).indices[:, 1].cpu().numpy()
    pt, pp = spectral_params(Yt), spectral_params(Yp)
    rows = []
    for name in (params or PARAMS):
        rel = pearson(pt[name], pt[name][nb])
        cap = np.sqrt(rel) if np.isfinite(rel) and rel > 0 else np.nan
        rm = pearson(pt[name], pp[name])
        rows.append({"param": name, "reliability": rel, "ceiling": cap, "model_r": rm,
                     "frac_of_ceiling": rm / cap if cap and np.isfinite(cap) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- metric set
def metric_set(Yp, Yt) -> dict:
    """Every scalar metric for one model. param_fidelity / per_band are separate calls.

    Reports `r_key_balanced` (Pearson), `ccc_key_balanced` (Lin) and
    `slope_key_balanced` side by side: a change that raises ccc_key while lowering
    r_key is restoring dynamic range, and vice versa."""
    pt, pp = spectral_params(Yt), spectral_params(Yp)  # computed once, shared
    rkb, _ = r_key(Yp, Yt, pt=pt, pp=pp)
    ckb, _ = r_key(Yp, Yt, pt=pt, pp=pp, metric="ccc")
    slope_key = float(np.mean([ols_slope(pt[k], pp[k]) for k in R_KEY_BALANCED]))
    return {
        "rmse": rmse(Yp, Yt), "mae": mae(Yp, Yt), "mrae": mrae(Yp, Yt),
        "sam_median": float(np.median(sam_deg(Yp, Yt))),
        "sam_mean": float(np.mean(sam_deg(Yp, Yt))),
        "sga_median": float(np.median(sga_deg(Yp, Yt))),
        "sid_mean": float(np.mean(sid(Yp, Yt))),
        "r_key_balanced": rkb, "ccc_key_balanced": ckb,
        "slope_key_balanced": slope_key,
    }