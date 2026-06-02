"""Gastrulation E7.5 + differential-kinetics workflow.

Adapted from scvelo's DifferentialKinetics.ipynb (BSD-3-Clause). Exercises
the differential_kinetic_test path: recover_dynamics is called twice (once
on all genes, then again on the genes flagged as multi-kinetic across
cell types), and velocity is recomputed with `diff_kinetics=True`.
"""

from __future__ import annotations


def load_data():
    """Fetch the Pijuan-Sala 2019 E7.5 subset (~21k cells, cached)."""
    import scvelo as scv

    return scv.datasets.gastrulation_e75()


def run(lib, adata) -> None:
    """Execute the differential-kinetics pipeline in-place on `adata`.

    scvelo's `tl.velocity(diff_kinetics=True)` reads `groupby` from
    `adata.uns["recover_dynamics"]["fit_diff_kinetics"]`, but the second
    `recover_dynamics` call (on multi-kinetic genes) resets that uns key -
    so we alias `celltype` to `clusters` (scvelo's fallback default) up
    front and use `groupby="clusters"` consistently.
    """
    import numpy as np
    import scanpy as sc

    # Alias the dataset's celltype column to "clusters" so scvelo's
    # default groupby path works without uns juggling.
    if "celltype" in adata.obs.columns and "clusters" not in adata.obs.columns:
        adata.obs["clusters"] = adata.obs["celltype"]

    lib.pp.filter_genes(adata, min_shared_counts=20)
    lib.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # Cast Mu/Ms to f64 so both backends share the same numerical environment
    # (scvelo defaults to f32; scvelo_rs runs pure f64 internally).
    for k in ("Mu", "Ms"):
        if k in adata.layers:
            adata.layers[k] = np.asarray(adata.layers[k], dtype=np.float64)

    # First fit on every (HVG-filtered) gene.
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)

    # Identify genes with heterogeneous kinetics across cell-type groups.
    lib.tl.differential_kinetic_test(adata, groupby="clusters")

    # Re-fit recover_dynamics on the multi-kinetic genes per cluster, then
    # recompute velocity with per-cluster kinetics.
    multi_kinetic_mask = adata.var.get("fit_diff_kinetics", None)
    if multi_kinetic_mask is not None and multi_kinetic_mask.any():
        multi_kinetic_genes = adata.var_names[multi_kinetic_mask.astype(bool)].tolist()
        if multi_kinetic_genes:
            # Restore uns key (the next recover_dynamics call resets it).
            lib.tl.recover_dynamics(
                adata,
                var_names=multi_kinetic_genes,
                n_jobs=1,
                show_progress_bar=False,
            )
            adata.uns.setdefault("recover_dynamics", {})["fit_diff_kinetics"] = "clusters"
            lib.tl.velocity(adata, mode="dynamical", diff_kinetics=True)
