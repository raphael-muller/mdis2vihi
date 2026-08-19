"""Canonical Mercury spectral parameters on the 5 nm x [300, 1450] nm grid.

Single source of truth: every script imports `refl` / `slope` / `spectral_params` from
here rather than redefining them.

Cross-checked spectrum by spectrum against Barraud's own `param.dat` on the 153 214
pairs. Reproduced at r >= 0.997: R415, R433, R480, R629, R749, R750, R828, R1050,
`ci_415_750`, `ci_750_415`, `vis_slope` (vs her `v_sl`), `vn_slope` (`vn_sl`),
`curvature` (`curv`), `nir_slope` (`n_sl`, since its window was narrowed to
1050-1400 nm). Noisier at the VIS/NIR junction: R899 (0.937), R950 (0.976), R996 (0.989),
and `ci_750_950` (0.663) which inherits it although its definition is identical to hers.
Not reproduced: `uv_downturn`, see its definition below.
"""

from __future__ import annotations

import numpy as np

GRID = np.arange(300.0, 1450.0 + 1e-6, 5.0)  # 231 bins, as scripts/01_build_pairs.py


def refl(Y, wl):
    """Linear interpolation of reflectance at wavelength `wl` (nm). Y: (..., 231) on GRID."""
    j = np.clip(np.searchsorted(GRID, wl), 1, GRID.size - 1)
    w = (wl - GRID[j - 1]) / (GRID[j] - GRID[j - 1])
    return Y[..., j - 1] * (1 - w) + Y[..., j] * w


def slope(Y, lo, hi):
    """OLS slope of reflectance vs wavelength (per nm) over [lo, hi]."""
    sel = (GRID >= lo) & (GRID <= hi)
    x = GRID[sel]; xm = x.mean()
    ys = Y[..., sel]; ym = ys.mean(-1, keepdims=True)
    return ((ys - ym) * (x - xm)).sum(-1) / ((x - xm) ** 2).sum()


def poly2_curv(Y, lo=300.0, hi=600.0):
    """Mercury's DIAGNOSTIC near-UV curvature (Barraud et al. 2020): the quadratic
    coefficient c of a degree-2 fit R = c·λ² + a·λ + b over [lo, hi] nm, the formal
    hollow diagnostic (curvature 5–17× the mean on hollows). NaN-aware per spectrum."""
    Y = np.atleast_2d(Y)
    sel = (GRID >= lo) & (GRID <= hi); x = GRID[sel]
    out = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        y = Y[i, sel]; f = np.isfinite(y)
        if f.sum() >= 5:
            out[i] = np.polyfit(x[f], y[f], 2)[0]
    return out


def uv_downturn(Y):
    """Absolute depth of the near-UV downturn: sum over 300/325/350 nm of the deviation
    below the linear extrapolation of the visible continuum fitted over 445-750 nm.
    NaN-aware per spectrum.

    The feature is the one described by Goudge et al. (2014); `uv_depth` below is their
    own relative form. The balanced set uses this one because it is better conditioned:
    at fixed MDIS input two MASCS spectra agree on it at r = 0.644, against r = 0.379 for
    `uv_depth`, whose denominator R(303) sits on the noisiest part of the VIRS range.
    """
    Y = np.atleast_2d(Y)
    sel = (GRID >= 445) & (GRID <= 750); x = GRID[sel]
    out = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        y = Y[i, sel]; f = np.isfinite(y)
        if f.sum() < 5:
            continue
        a, b = np.polyfit(x[f], y[f], 1); d = 0.0
        for wl in (300.0, 325.0, 350.0):
            j = int(np.where(GRID == wl)[0][0])
            if np.isfinite(Y[i, j]):
                d += (a * wl + b) - Y[i, j]
        out[i] = d
    return out


def uv_depth(Y):
    """`UVdepth` of Goudge et al. (2014), eq. (1)-(5):

        VISslope = [R(550) - R(750)] / (550 - 750)
        Depth_l  = [R(401) - (401 - l) * VISslope] / R(l)      for l = 303, 324, 350
        UVdepth  = Depth_303 + Depth_324 + Depth_350

    Each term is the continuum-predicted reflectance over the measured one, so the whole
    parameter is invariant to an overall brightness factor; Goudge reports ~3.11 +/- 0.08
    on Mercury's pyroclastic deposits.

    Two departures from the paper, both forced by the data here: it is evaluated on the
    unratioed spectrum already resampled to 5 nm, where Goudge uses deposit-over-
    background ratios at native resolution. That reaches r = 0.74 against Barraud's
    `uv_down` column, which is not a reproduction; the remaining disagreement is
    unexplained.
    """
    vis = (refl(Y, 550.0) - refl(Y, 750.0)) / (550.0 - 750.0)
    return sum((refl(Y, 401.0) - (401.0 - wl) * vis) / refl(Y, wl)
               for wl in (303.0, 324.0, 350.0))


# Reflectances at named wavelengths, from Barraud's `param.dat`, so a parameter
# computed here compares with hers spectrum by spectrum. `R559` is at **556.9 nm** on
# purpose: it reproduces her `r556_9` column to 0.10 % against 0.99 % at 558.9 nm. That
# is her anchor, ~2 nm off the WAC filter-D centre used for the bandpass model in
# `scripts/02_build_lsf_target.py`.
WL = {'R310': 310.0, 'R390': 390.0, 'R415': 415.0, 'R433': 433.2, 'R480': 479.9,
      'R559': 556.9, 'R629': 628.8, 'R749': 748.7, 'R750': 750.0, 'R828': 828.4,
      'R899': 898.8, 'R950': 950.0, 'R996': 996.2, 'R1050': 1050.0, 'R1400': 1400.0}


def spectral_params(Y):
    """Comprehensive Mercury spectral parameters on GRID. Returns dict of (N,) arrays."""
    p = {name: refl(Y, wl) for name, wl in WL.items()}
    p['ci_310_390'] = p['R310'] / p['R390']      # Barraud ci_310_390 (noisy 2-band)
    p['ci_415_750'] = p['R415'] / p['R750']      # Barraud ci_415_750, blue index
    p['ci_750_415'] = p['R750'] / p['R415']      # red index (= 1 / ci_415_750)
    p['ci_750_950'] = p['R750'] / p['R950']      # Barraud ci_750_950, VIS/NIR ratio
    p['ratio_430_1000'] = p['R433'] / p['R996']  # MDIS colour discriminant
    p['vis_slope'] = slope(Y, 415, 750)          # red slope (steep on Mercury)
    p['nir_slope'] = slope(Y, 1050, 1400)        # Barraud n_sl, clear of the junction
    p['vn_slope'] = slope(Y, 415, 1400)          # full VIS->NIR slope
    p['uv_slope'] = slope(Y, 300, 415)           # near-UV slope (noisy blue edge)
    p['curvature'] = poly2_curv(Y, 300, 600)     # Barraud 2020 (hollow diagnostic)
    p['uv_downturn'] = uv_downturn(Y)            # depth below the visible continuum
    p['uv_depth'] = uv_depth(Y)                  # Goudge et al. (2014), eq. 1-5
    return p


PARAMS = list(spectral_params(np.ones((1, GRID.size))).keys())

FAMILIES = {
    "reflectance": "R559",       # VIS amplitude / albedo level
    "vis_slope": "vis_slope",    # VIS red slope (dominant Mercury feature)
    "vis_nir": "ratio_430_1000", # VIS↔NIR discriminant
    "nir_slope": "nir_slope",    # NIR slope (extrapolated, target-noisy)
    "uv_curvature": "curvature", # near-UV curvature (hollow diagnostic, extrapolated)
    "uv_downturn": "uv_downturn",# UV downturn (extrapolated)
}
R_KEY_BALANCED = list(FAMILIES.values())