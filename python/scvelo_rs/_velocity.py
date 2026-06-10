"""Drop-in replacements for `scvelo.tl.velocity` and `scvelo.tl.velocity_graph`.

Both route the hot loops through Rust. `velocity` covers deterministic mode;
stochastic mode falls back to scvelo (depends on second-order moments).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import issparse

from ._scvelo_rs import velocity_graph_kernel, velocity_kernel


def velocity(
    data,
    vkey: str = "velocity",
    mode: str = "stochastic",
    fit_offset: bool = False,
    fit_offset2: bool = False,
    filter_genes: bool = False,
    groups=None,
    groupby: str | None = None,
    groups_for_fit=None,
    constrain_ratio=None,
    use_raw: bool = False,
    use_latent_time: bool | None = None,
    perc=None,
    min_r2: float = 1e-2,
    min_likelihood: float = 1e-3,
    r2_adjusted: bool | None = None,
    use_highly_variable: bool = True,
    diff_kinetics: bool | None = None,
    copy: bool = False,
    **kwargs: Any,
):
    """Drop-in for `scvelo.tl.velocity`.

    Deterministic mode runs in Rust (incl. extreme-quantile percentile
    trimming). Other modes (`stochastic`, `dynamical`) fall through to scvelo
    upstream because they depend on second-order moments / EM fits we haven't
    fully ported.
    """
    if mode != "deterministic":
        return _fallback_velocity(
            data,
            vkey=vkey,
            mode=mode,
            fit_offset=fit_offset,
            fit_offset2=fit_offset2,
            filter_genes=filter_genes,
            groups=groups,
            groupby=groupby,
            groups_for_fit=groups_for_fit,
            constrain_ratio=constrain_ratio,
            use_raw=use_raw,
            use_latent_time=use_latent_time,
            perc=perc,
            min_r2=min_r2,
            min_likelihood=min_likelihood,
            r2_adjusted=r2_adjusted,
            use_highly_variable=use_highly_variable,
            diff_kinetics=diff_kinetics,
            copy=copy,
            **kwargs,
        )

    if groups_for_fit is not None or groups is not None or filter_genes:
        # Group-based fitting and gene filtering not yet ported.
        return _fallback_velocity(
            data,
            vkey=vkey,
            mode=mode,
            fit_offset=fit_offset,
            fit_offset2=fit_offset2,
            filter_genes=filter_genes,
            groups=groups,
            groupby=groupby,
            groups_for_fit=groups_for_fit,
            constrain_ratio=constrain_ratio,
            use_raw=use_raw,
            use_latent_time=use_latent_time,
            perc=perc,
            min_r2=min_r2,
            min_likelihood=min_likelihood,
            r2_adjusted=r2_adjusted,
            use_highly_variable=use_highly_variable,
            diff_kinetics=diff_kinetics,
            copy=copy,
            **kwargs,
        )

    adata = data.copy() if copy else data

    # scvelo's `tl.velocity()` substitutes `perc=None → [5, 95]` internally.
    if perc is None:
        perc = [5, 95]

    # scvelo's LinearRegression: with fit_intercept=False AND perc as a
    # list/tuple, it collapses `percentile` to the UPPER bound only (taking
    # `perc[1]`). So `[5, 95]` becomes single-percentile=95 in that branch.
    if not fit_offset and isinstance(perc, (list, tuple, np.ndarray)) and len(perc) == 2:
        perc_lo, perc_hi = float(perc[1]), None
    elif not isinstance(perc, (list, tuple, np.ndarray)):
        perc_lo, perc_hi = float(perc), None
    elif len(perc) == 1:
        perc_lo, perc_hi = float(perc[0]), None
    else:
        perc_lo, perc_hi = float(perc[0]), float(perc[1])

    Ms_layer = adata.layers["spliced" if use_raw else "Ms"]
    Mu_layer = adata.layers["unspliced" if use_raw else "Mu"]
    Ms = Ms_layer.toarray() if issparse(Ms_layer) else np.asarray(Ms_layer)
    Mu = Mu_layer.toarray() if issparse(Mu_layer) else np.asarray(Mu_layer)
    Ms = np.ascontiguousarray(Ms, dtype=np.float64)
    Mu = np.ascontiguousarray(Mu, dtype=np.float64)

    constrain_lo = constrain_hi = None
    if constrain_ratio is not None:
        if np.size(constrain_ratio) < 2:
            constrain_lo, constrain_hi = None, float(constrain_ratio)
        else:
            constrain_lo, constrain_hi = float(constrain_ratio[0]), float(constrain_ratio[1])

    gamma, offset, r2, residual, velocity_genes = velocity_kernel(
        Ms,
        Mu,
        fit_offset,
        float(min_r2),
        1e-2,
        constrain_lo,
        constrain_hi,
        perc_lo,
        perc_hi,
    )

    velocity_genes = np.asarray(velocity_genes)
    if use_highly_variable and "highly_variable" in adata.var.columns:
        velocity_genes &= adata.var["highly_variable"].to_numpy().astype(bool)

    if int(np.sum(velocity_genes)) < 2:
        thresh = float(np.percentile(np.asarray(r2), 80))
        velocity_genes = np.asarray(r2) > thresh

    adata.layers[vkey] = np.asarray(residual, dtype=np.float64)
    adata.var[f"{vkey}_gamma"] = np.asarray(gamma)
    adata.var[f"{vkey}_offset"] = np.asarray(offset)
    adata.var[f"{vkey}_r2"] = np.asarray(r2)
    adata.var[f"{vkey}_genes"] = velocity_genes
    adata.var[f"{vkey}_qreg_ratio"] = np.asarray(gamma)

    params = adata.uns.get(f"{vkey}_params", {})
    params["mode"] = mode
    params["fit_offset"] = fit_offset
    params["perc"] = perc
    adata.uns[f"{vkey}_params"] = params

    return adata if copy else None


def _fallback_velocity(data, **kwargs):
    """Fall through to scvelo upstream for non-deterministic / advanced modes."""
    import scvelo as scv

    fn = getattr(scv.tl, "velocity_original", None) or scv.tl.velocity
    return fn(data, **kwargs)


def velocity_graph(
    data,
    vkey: str = "velocity",
    xkey: str = "Ms",
    tkey=None,
    basis=None,
    n_neighbors: int | None = None,
    n_recurse_neighbors: int | None = None,
    random_neighbors_at_max=None,
    sqrt_transform: bool | None = None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties: bool | None = None,
    approx=None,
    mode_neighbors: str = "distances",
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
):
    """Drop-in for `scvelo.tl.velocity_graph`. Rayon-parallel cosine kernel."""
    from scvelo.preprocessing.neighbors import (
        get_n_neighs,
        get_neighs,
        neighbors,
        verify_neighbors,
    )
    from scvelo.tools.utils import get_indices
    from scvelo.tools.velocity_graph import vals_to_csr

    adata = data.copy() if copy else data
    verify_neighbors(adata)
    if vkey not in adata.layers.keys():
        # Use our wrapper so the auto-velocity path runs through Rust when
        # eligible (deterministic mode); otherwise it falls through to scvelo.
        velocity(adata, vkey=vkey)
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    # Gene subset selection (mirrors scvelo's VelocityGraph.__init__).
    subset = np.ones(adata.n_vars, bool)
    if gene_subset is not None:
        var_names_subset = adata.var_names.isin(gene_subset)
        subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
    elif f"{vkey}_genes" in adata.var.keys():
        subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

    xkey = xkey if xkey in adata.layers.keys() else "spliced"

    X_layer = adata.layers[xkey]
    V_layer = adata.layers[vkey]
    X = np.array(X_layer.toarray()[:, subset] if issparse(X_layer) else X_layer[:, subset])
    V = np.array(V_layer.toarray()[:, subset] if issparse(V_layer) else V_layer[:, subset])

    nans = np.isnan(np.sum(V, axis=0))
    if np.any(nans):
        X = X[:, ~nans]
        V = V[:, ~nans]

    if approx is True and X.shape[1] > 100:
        from scvelo.preprocessing.neighbors import pca

        X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
        X = np.array(X_pca, dtype=np.float32)
        V = (V - V.mean(0)).dot(PCs.T)
        V[V.sum(1) == 0] = 0
    else:
        X = np.array(X, dtype=np.float32)
        V = np.array(V, dtype=np.float32)

    if sqrt_transform is None:
        uns_key = f"{vkey}_params"
        if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
            sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
    if sqrt_transform:
        V = np.sqrt(np.abs(V)) * np.sign(V)
    V -= np.nanmean(V, axis=1)[:, None]

    if n_recurse_neighbors is None:
        if n_neighbors is not None or mode_neighbors == "connectivities":
            n_recurse_neighbors = 1
        else:
            n_recurse_neighbors = 2

    if "neighbors" not in adata.uns.keys():
        neighbors(adata)
    if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
        raise ValueError("Your neighbor graph seems to be corrupted. Recompute via pp.neighbors.")

    # Resolve neighbor indices array (n_cells, n_knn).
    if n_neighbors is None or n_neighbors <= get_n_neighs(adata):
        indices = get_indices(
            dist=get_neighs(adata, "distances"),
            n_neighbors=n_neighbors,
            mode_neighbors=mode_neighbors,
        )[0]
    else:
        if basis is None:
            basis_keys = ["X_pca", "X_tsne", "X_umap"]
            basis = [k for k in basis_keys if k in adata.obsm.keys()][-1]
        elif f"X_{basis}" in adata.obsm.keys():
            basis = f"X_{basis}"
        if isinstance(approx, str) and approx in adata.obsm.keys():
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nbrs.fit(adata.obsm[approx])
            indices = nbrs.kneighbors_graph(mode="connectivity").indices.reshape(
                (-1, n_neighbors + 1)
            )
        else:
            from scvelo import Neighbors

            nbrs = Neighbors(adata)
            nbrs.compute_neighbors(n_neighbors=n_neighbors, use_rep=basis, n_pcs=10)
            indices = get_indices(dist=nbrs.distances, mode_neighbors=mode_neighbors)[0]

    # tkey time-aware mode currently falls back to scvelo.
    if tkey is not None and tkey in adata.obs.keys():
        return _fallback_to_scvelo(
            data,
            vkey=vkey,
            xkey=xkey,
            tkey=tkey,
            basis=basis,
            n_neighbors=n_neighbors,
            n_recurse_neighbors=n_recurse_neighbors,
            random_neighbors_at_max=random_neighbors_at_max,
            sqrt_transform=sqrt_transform,
            variance_stabilization=variance_stabilization,
            gene_subset=gene_subset,
            compute_uncertainties=compute_uncertainties,
            approx=approx,
            mode_neighbors=mode_neighbors,
            copy=copy,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )

    # random_neighbors_at_max not yet supported in Rust (uses np.random.choice).
    if random_neighbors_at_max is not None:
        return _fallback_to_scvelo(
            data,
            vkey=vkey,
            xkey=xkey,
            tkey=tkey,
            basis=basis,
            n_neighbors=n_neighbors,
            n_recurse_neighbors=n_recurse_neighbors,
            random_neighbors_at_max=random_neighbors_at_max,
            sqrt_transform=sqrt_transform,
            variance_stabilization=variance_stabilization,
            gene_subset=gene_subset,
            compute_uncertainties=compute_uncertainties,
            approx=approx,
            mode_neighbors=mode_neighbors,
            copy=copy,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )

    indices_c = np.ascontiguousarray(indices, dtype=np.int32)
    X_c = np.ascontiguousarray(X, dtype=np.float32)
    V_c = np.ascontiguousarray(V, dtype=np.float32)

    rows, cols, vals = velocity_graph_kernel(X_c, V_c, indices_c, n_recurse_neighbors)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    vals = np.asarray(vals, dtype=np.float64)
    vals[np.isnan(vals)] = 0

    n_obs = X.shape[0]
    graph, graph_neg = vals_to_csr(vals, rows, cols, shape=(n_obs, n_obs), split_negative=True)

    adata.uns[f"{vkey}_graph"] = graph
    adata.uns[f"{vkey}_graph_neg"] = graph_neg

    confidence = graph.max(1).toarray().flatten()
    self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)
    adata.obs[f"{vkey}_self_transition"] = self_prob

    if compute_uncertainties:
        # Falls back to scvelo for the per-cell uncertainty computation
        # (which depends on second-order moments).
        from scvelo.tools.velocity_graph import velocity_graph as _scv_vg

        _scv_vg(
            adata,
            vkey=vkey,
            xkey=xkey,
            tkey=tkey,
            basis=basis,
            n_neighbors=n_neighbors,
            n_recurse_neighbors=n_recurse_neighbors,
            random_neighbors_at_max=random_neighbors_at_max,
            sqrt_transform=sqrt_transform,
            gene_subset=gene_subset,
            compute_uncertainties=True,
            approx=approx,
            mode_neighbors=mode_neighbors,
            copy=False,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=False,
        )

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = n_recurse_neighbors

    return adata if copy else None


def _fallback_to_scvelo(data, **kwargs):
    """Fall back to scvelo's implementation for unsupported feature combos."""
    import scvelo as scv

    fn = getattr(scv.tl, "velocity_graph_original", None) or scv.tl.velocity_graph
    return fn(data, **kwargs)
