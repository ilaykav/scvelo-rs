"""Rust-backed `scv.tl.differential_kinetic_test`.

Mirrors scvelo's public API but moves the per-gene (per-cluster) LRT
into Rust. The Rust kernel calls `assign_timepoints` once per gene
(bit-exact match to scvelo's `compute_divergence(mode='assign_timepoints')`),
then slices by cluster to compute distx sums + orth_distx + p-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._scvelo_rs import diff_kinetic_test_kernel as _kernel


def _ensure_velocity_genes(adata) -> np.ndarray:
    """Mask of velocity_genes (fall back to fitted genes if not present)."""
    var = adata.var
    if "velocity_genes" in var.columns:
        return var["velocity_genes"].values.astype(bool)
    if "fit_alpha" in var.columns:
        return ~np.isnan(var["fit_alpha"].values)
    return np.ones(adata.n_vars, dtype=bool)


def _get_connectivity_triplet(adata):
    """Return (data, indices, indptr) tuple (f64, i32, i32) from
    adata.obsp['connectivities'] CSR matrix, or (None, None, None)."""
    obsp = getattr(adata, "obsp", None)
    if obsp is None or "connectivities" not in obsp.keys():
        return None, None, None
    conn = obsp["connectivities"]
    if not hasattr(conn, "tocsr"):
        return None, None, None
    csr = conn.tocsr()
    return (
        np.ascontiguousarray(csr.data, dtype=np.float64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.indptr, dtype=np.int32),
    )


def differential_kinetic_test(
    adata,
    var_names="velocity_genes",
    groupby=None,
    use_raw: bool = False,
    add_key: str = "fit",
    copy: bool = False,
    min_cells: int = 10,
    **kwargs,
):
    """Drop-in for `scvelo.tl.differential_kinetic_test`. Rust-backed."""
    adata = adata.copy() if copy else adata

    # Resolve groupby
    if groupby is None:
        if "clusters" in adata.obs.columns:
            groupby = "clusters"
        elif "louvain" in adata.obs.columns:
            groupby = "louvain"
        else:
            raise ValueError("differential_kinetic_test: pass groupby=...")
    if groupby not in adata.obs.columns:
        raise ValueError(f"obs column {groupby!r} not found")

    # Resolve var_names
    if isinstance(var_names, str):
        if var_names == "velocity_genes":
            mask = _ensure_velocity_genes(adata)
            var_idx = np.where(mask)[0]
        elif var_names == "all":
            var_idx = np.arange(adata.n_vars)
        elif var_names in adata.var.columns:
            var_idx = np.where(adata.var[var_names].values.astype(bool))[0]
        elif var_names in adata.var_names:
            var_idx = np.array([adata.var_names.get_loc(var_names)])
        else:
            raise ValueError(f"var_names={var_names!r} not found")
    else:
        var_idx = np.array([adata.var_names.get_loc(g) for g in var_names])

    if f"{add_key}_alpha" not in adata.var.columns:
        raise ValueError(
            f"recover_dynamics must run before differential_kinetic_test "
            f"(missing var['{add_key}_alpha'])"
        )
    fit_alpha_all = adata.var[f"{add_key}_alpha"].values
    var_idx = var_idx[~np.isnan(fit_alpha_all[var_idx])]
    n_genes = len(var_idx)
    if n_genes == 0:
        raise ValueError("no fitted genes to test")

    n_cells = adata.n_obs

    # Cluster assignment as i32
    clusters_series = adata.obs[groupby]
    if not isinstance(clusters_series.dtype, pd.CategoricalDtype):
        clusters_series = clusters_series.astype("category")
    cluster_cats = list(clusters_series.cat.categories)
    n_clusters = len(cluster_cats)
    cluster_assign = np.ascontiguousarray(clusters_series.cat.codes.values, dtype=np.int32)

    # Per-gene fit params
    def _get_var(col, default=None):
        if col in adata.var.columns:
            return np.ascontiguousarray(
                adata.var[col].values[var_idx], dtype=np.float64
            )
        if default is None:
            raise KeyError(col)
        return np.full(n_genes, default, dtype=np.float64)

    alpha = _get_var(f"{add_key}_alpha")
    beta = _get_var(f"{add_key}_beta")
    gamma = _get_var(f"{add_key}_gamma")
    scaling = _get_var(f"{add_key}_scaling", default=1.0)
    t_ = _get_var(f"{add_key}_t_")
    std_u = _get_var(f"{add_key}_std_u", default=1.0)
    std_s = _get_var(f"{add_key}_std_s", default=1.0)
    if f"{add_key}_variance" in adata.var.columns:
        varx = _get_var(f"{add_key}_variance", default=1.0)
    else:
        varx = np.ones(n_genes, dtype=np.float64)

    # u0_ = u(t_, 0, alpha, beta), s0_ = s(t_, 0, 0, alpha, beta, gamma).
    # Sign-preserving guard against gamma == beta degenerate case.
    expu_t = np.exp(-beta * t_)
    exps_t = np.exp(-gamma * t_)
    u0_ = (alpha / beta) * (1.0 - expu_t)
    g_minus_b = gamma - beta
    g_minus_b = np.where(np.abs(g_minus_b) < 1e-300, np.sign(g_minus_b) * 1e-300 + 1e-300, g_minus_b)
    c_switch = alpha / g_minus_b
    s0_ = (alpha / gamma) * (1.0 - exps_t) + c_switch * (exps_t - expu_t)

    # Per-cell Mu/Ms slices, then u_scaled = Mu / scaling per gene
    if use_raw:
        u_layer, s_layer = "unspliced", "spliced"
    else:
        u_layer = "Mu" if "Mu" in adata.layers else "unspliced"
        s_layer = "Ms" if "Ms" in adata.layers else "spliced"

    Mu = np.ascontiguousarray(adata.layers[u_layer][:, var_idx], dtype=np.float64)
    Ms = np.ascontiguousarray(adata.layers[s_layer][:, var_idx], dtype=np.float64)
    u_scaled = np.ascontiguousarray(Mu / scaling[np.newaxis, :], dtype=np.float64)

    # Base weights: cells with valid moments (matches scvelo's initialize_weights
    # for `weighted=False` mode after recover_dynamics has run).
    weights = np.isfinite(Mu) & np.isfinite(Ms) & (Mu > 0) & (Ms > 0)
    weights = np.ascontiguousarray(weights, dtype=bool)

    # Connectivities for compute_divergence smoothing inside assign_timepoints
    conn_data, conn_indices, conn_indptr = _get_connectivity_triplet(adata)

    # Call Rust kernel
    pvals = _kernel(
        u_scaled,
        Ms,
        weights,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u,
        std_s,
        varx,
        cluster_assign,
        n_clusters,
        min_cells,
        True,  # fit_steady_states
        conn_data,
        conn_indices,
        conn_indptr,
    )  # (n_genes, n_clusters)

    # Write back to adata
    full_pvals = np.full((adata.n_vars, n_clusters), np.nan, dtype=np.float64)
    full_pvals[var_idx, :] = pvals
    adata.varm["fit_pvals_kinetics"] = full_pvals

    diff_str = np.empty(adata.n_vars, dtype=object)
    for j, vi in enumerate(var_idx):
        sig_clusters = [
            cluster_cats[c] for c in range(n_clusters) if pvals[j, c] < 1e-2
        ]
        diff_str[vi] = ",".join(map(str, sig_clusters)) if sig_clusters else ""
    adata.var["fit_diff_kinetics"] = diff_str

    pval_max = np.full(adata.n_vars, np.nan, dtype=np.float64)
    for j, vi in enumerate(var_idx):
        sig_mask = pvals[j, :] < 1e-2
        if sig_mask.any():
            pval_max[vi] = float(np.max(pvals[j, sig_mask]))
    adata.var["fit_pval_kinetics"] = pval_max

    adata.uns.setdefault("recover_dynamics", {})["fit_diff_kinetics"] = groupby

    return adata if copy else None
