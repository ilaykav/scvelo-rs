"""Rust-backed `compute_dynamics`. Mirrors `scvelo.utils.compute_dynamics`."""

from __future__ import annotations

import numpy as np

from ._scvelo_rs import compute_dynamics_kernel as _kernel


def compute_dynamics(
    adata,
    basis,
    key: str = "true",
    extrapolate=None,
    sort: bool = True,
    t_=None,
    t=None,
):
    idx = adata.var_names.get_loc(basis) if isinstance(basis, str) else basis

    # Direct DataFrame access (no AnnData slice — profiling showed `adata[:, basis]`
    # cost ~65% of the wrapper's wall time per call).
    var = adata.var
    var_cols = var.columns
    if f"{key}_gamma" not in var_cols:
        key = "fit"

    def _scalar(col: str, default: float) -> float:
        if col in var_cols:
            return float(var[col].values[idx])
        return default

    alpha = _scalar(f"{key}_alpha", 1.0)
    beta_unscaled = _scalar(f"{key}_beta", 1.0)
    gamma = _scalar(f"{key}_gamma", 1.0)
    scaling = _scalar(f"{key}_scaling", 1.0)
    t_val = _scalar(f"{key}_t_", 0.0)
    beta = beta_unscaled * scaling

    if "fit_u0" in var_cols:
        u0_offset = float(var["fit_u0"].values[idx])
        s0_offset = float(var["fit_s0"].values[idx])
    else:
        u0_offset, s0_offset = 0.0, 0.0

    if t is None or isinstance(t, bool) or len(t) < adata.n_obs:
        if key == "true":
            t = np.asarray(adata.obs[f"{key}_t"].values, dtype=np.float64)
        else:
            t = np.asarray(adata.layers[f"{key}_t"][:, idx], dtype=np.float64)
    else:
        t = np.asarray(t, dtype=np.float64)

    if extrapolate:
        tmax = float(np.max(t))
        t = np.concatenate([np.linspace(0.0, t_val, num=500), np.linspace(t_val, tmax, num=500)])

    if sort:
        t = np.sort(t)

    t = np.ascontiguousarray(t, dtype=np.float64)
    alpha_arr, u_arr, s_arr = _kernel(
        t, t_val, alpha, beta, gamma, scaling, u0_offset, s0_offset, False
    )
    return alpha_arr, u_arr, s_arr
