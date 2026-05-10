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
    if f"{key}_gamma" not in adata.var_keys():
        key = "fit"

    sub = adata[:, basis]
    alpha = float(sub.var[f"{key}_alpha"].values[0]) if f"{key}_alpha" in sub.var.keys() else 1.0
    beta_unscaled = (
        float(sub.var[f"{key}_beta"].values[0]) if f"{key}_beta" in sub.var.keys() else 1.0
    )
    gamma = float(sub.var[f"{key}_gamma"].values[0])
    scaling = (
        float(sub.var[f"{key}_scaling"].values[0])
        if f"{key}_scaling" in sub.var.keys()
        else 1.0
    )
    t_val = float(sub.var[f"{key}_t_"].values[0])
    beta = beta_unscaled * scaling

    if "fit_u0" in adata.var.keys():
        u0_offset = float(adata.var["fit_u0"].iloc[idx])
        s0_offset = float(adata.var["fit_s0"].iloc[idx])
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
        t = np.concatenate(
            [np.linspace(0.0, t_val, num=500), np.linspace(t_val, tmax, num=500)]
        )

    if sort:
        t = np.sort(t)

    t = np.ascontiguousarray(t, dtype=np.float64)
    alpha_arr, u_arr, s_arr = _kernel(
        t, t_val, alpha, beta, gamma, scaling, u0_offset, s0_offset, False
    )
    return alpha_arr, u_arr, s_arr
