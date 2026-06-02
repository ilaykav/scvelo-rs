"""Pancreas tutorial - canonical scvelo dynamical-model walkthrough.

Adapted from theislab/scvelo_notebooks/Pancreas.ipynb (BSD-3-Clause).

The bench harness calls `load_data()` once, then `run(lib, adata)` for each
backend with a fresh `.copy()` of the same AnnData - so scvelo and scvelo-rs
operate on byte-identical input.
"""

from __future__ import annotations


def load_data():
    """Fetch the Bastidas-Ponce 2019 pancreas dataset (cached under ~/.scvelo)."""
    import scvelo as scv

    return scv.datasets.pancreas()


def run(lib, adata) -> None:
    """Execute the full dynamical-model pipeline in-place on `adata`.

    Preprocessing splits `filter_and_normalize` into its constituent steps
    because scvelo 0.3.4's combined helper forwards `**kwargs` (including
    `n_top_genes`) to `normalize_per_cell`, which rejects them. HVG selection
    therefore goes through scanpy directly.
    """
    import numpy as np
    import scanpy as sc

    lib.pp.filter_genes(adata, min_shared_counts=20)
    lib.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # Cast Mu/Ms to f64 - scvelo defaults to f32 (numpy preserves f32 inside
    # `np.std`, `np.mean`), while scvelo_rs runs pure f64 internally. The
    # mismatch produces ~25% drift on outlier genes per CLAUDE.md Phase 3.x.
    # Casting both backends to f64 puts them in the same numerical
    # environment for bit-exact comparison.
    for k in ("Mu", "Ms"):
        if k in adata.layers:
            adata.layers[k] = np.asarray(adata.layers[k], dtype=np.float64)
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)
    lib.tl.latent_time(adata)
