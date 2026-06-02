"""Rust-backed bit-exact `scv.tl.differential_kinetic_test`.

The Rust kernel mirrors scvelo's `DynamicsRecovery.differential_kinetic_test`
per gene exactly - `initialize_diff_kinetics` (weights/std/outside_of_trajectory),
`get_variance`, `get_cluster_mse`, `get_orth_fit`, `get_pval_diff_kinetics` -
sliced by cluster mask. The Python side only marshalls arrays.
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
    """Build the SAME connectivity matrix scvelo uses inside `dm.connectivities`.

    `get_connectivities(adata)` row-normalizes the kNN adjacency and adds a
    self-loop. Raw `adata.obsp["connectivities"]` has row sums of 5-15 and
    no self-loops, so using it directly produces a completely different
    smoothing kernel (compute_divergence's argmin then disagrees with scvelo).
    """
    try:
        from scvelo.preprocessing.moments import get_connectivities

        conn = get_connectivities(adata)
    except Exception:
        return None, None, None
    if conn is None or conn is False:
        return None, None, None
    if not hasattr(conn, "tocsr"):
        return None, None, None
    csr = conn.tocsr()
    return (
        np.ascontiguousarray(csr.data, dtype=np.float64),
        np.ascontiguousarray(csr.indices, dtype=np.int32),
        np.ascontiguousarray(csr.indptr, dtype=np.int32),
    )


def _resolve_var_idx(adata, var_names, add_key) -> np.ndarray:
    """Replicate scvelo's var_names resolution path."""
    if isinstance(var_names, str):
        if var_names == "velocity_genes":
            return np.where(_ensure_velocity_genes(adata))[0]
        if var_names == "all":
            return np.arange(adata.n_vars)
        if var_names in adata.var.columns:
            return np.where(adata.var[var_names].values.astype(bool))[0]
        if var_names in adata.var_names:
            return np.array([adata.var_names.get_loc(var_names)])
        raise ValueError(f"var_names={var_names!r} not found")
    return np.array([adata.var_names.get_loc(g) for g in var_names])


def differential_kinetic_test(
    data,
    var_names="velocity_genes",
    groupby=None,
    use_raw=None,
    return_model=None,
    add_key: str = "fit",
    copy=None,
    min_cells: int = 10,
    **kwargs,
):
    """Drop-in for `scvelo.tl.differential_kinetic_test`. Rust-backed, bit-exact."""
    adata = data.copy() if copy else data

    # Resolve groupby (matches scvelo's fallback chain).
    if groupby is None:
        if "clusters" in adata.obs.columns:
            groupby = "clusters"
        elif "louvain" in adata.obs.columns:
            groupby = "louvain"
        else:
            raise ValueError("differential_kinetic_test: pass groupby=...")
    if groupby not in adata.obs.columns:
        raise ValueError(f"obs column {groupby!r} not found")

    var_idx = _resolve_var_idx(adata, var_names, add_key)
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

    # Cluster assignment.
    clusters_series = adata.obs[groupby]
    if not isinstance(clusters_series.dtype, pd.CategoricalDtype):
        clusters_series = clusters_series.astype("category")
    cluster_cats = list(clusters_series.cat.categories)
    n_clusters = len(cluster_cats)
    cluster_assign = np.ascontiguousarray(
        clusters_series.cat.codes.values, dtype=np.int32
    )

    # Per-gene fit params. scvelo's load_pars stores beta as fit_beta /
    # scaling (line 585 of _em_model_core.py); load_pars then multiplies
    # back to recover the internal beta. We pass the internal beta to Rust.
    def _get_var(col, default=None):
        if col in adata.var.columns:
            return np.ascontiguousarray(
                adata.var[col].values[var_idx], dtype=np.float64
            )
        if default is None:
            raise KeyError(col)
        return np.full(n_genes, default, dtype=np.float64)

    alpha = _get_var(f"{add_key}_alpha")
    fit_beta = _get_var(f"{add_key}_beta")
    gamma = _get_var(f"{add_key}_gamma")
    scaling = _get_var(f"{add_key}_scaling", default=1.0)
    t_ = _get_var(f"{add_key}_t_")
    beta_internal = fit_beta * scaling  # matches load_pars

    # Raw Mu, Ms. scvelo's `load_pars` reads adata.layers["Mu"][:, idx] (or
    # "unspliced" with use_raw=True). The Rust kernel does u/scaling internally.
    if use_raw:
        u_layer, s_layer = "unspliced", "spliced"
    else:
        u_layer = "Mu" if "Mu" in adata.layers else "unspliced"
        s_layer = "Ms" if "Ms" in adata.layers else "spliced"
    u_raw = np.ascontiguousarray(adata.layers[u_layer][:, var_idx], dtype=np.float64)
    s_raw = np.ascontiguousarray(adata.layers[s_layer][:, var_idx], dtype=np.float64)

    conn_data, conn_indices, conn_indptr = _get_connectivity_triplet(adata)

    pvals = _kernel(
        u_raw,
        s_raw,
        alpha,
        beta_internal,
        gamma,
        scaling,
        t_,
        cluster_assign,
        n_clusters,
        min_cells,
        conn_data,
        conn_indices,
        conn_indptr,
    )  # (n_genes, n_clusters)

    # Writeback - matches scvelo's exact var/varm/uns schema.
    # scvelo stores varm["fit_pvals_kinetics"] as an (n_vars,) recarray of
    # float32 per-cluster fields (one field per cluster category).
    full_pvals_f32 = np.full((adata.n_vars, n_clusters), np.nan, dtype=np.float32)
    full_pvals_f32[var_idx, :] = pvals.astype(np.float32)
    dtype = [(str(name), "float32") for name in cluster_cats]
    adata.varm[f"{add_key}_pvals_kinetics"] = np.rec.fromarrays(
        full_pvals_f32.T, dtype=dtype
    ).T

    # var["fit_diff_kinetics"]: comma-separated list of significant cluster names.
    diff_str = np.empty(adata.n_vars, dtype=object)
    diff_str[:] = None
    for j, vi in enumerate(var_idx):
        sig = [cluster_cats[c] for c in range(n_clusters) if pvals[j, c] < 1e-2]
        diff_str[vi] = ",".join(str(x) for x in sig) if sig else ""
    adata.var[f"{add_key}_diff_kinetics"] = diff_str

    # var["fit_pval_kinetics"]: max pval among significant clusters (NaN if none).
    pval_max = np.full(adata.n_vars, np.nan, dtype=np.float64)
    for j, vi in enumerate(var_idx):
        sig_mask = pvals[j, :] < 1e-2
        if sig_mask.any():
            pval_max[vi] = float(np.max(pvals[j, sig_mask]))
    adata.var[f"{add_key}_pval_kinetics"] = pval_max

    adata.uns.setdefault("recover_dynamics", {})["fit_diff_kinetics"] = groupby

    return adata if copy else None
