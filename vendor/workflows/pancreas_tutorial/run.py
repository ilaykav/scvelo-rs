"""Pancreas tutorial — canonical scvelo dynamical-model walkthrough.

Adapted from theislab/scvelo_notebooks/Pancreas.ipynb (BSD-3-Clause).

The bench harness calls `load_data()` once, then `run(lib, adata)` for each
backend with a fresh `.copy()` of the same AnnData — so scvelo and scvelo-rs
operate on byte-identical input.
"""

from __future__ import annotations


def load_data():
    """Fetch the Bastidas-Ponce 2019 pancreas dataset (cached under ~/.scvelo)."""
    import scvelo as scv

    return scv.datasets.pancreas()


def run(lib, adata) -> None:
    """Execute the full dynamical-model pipeline in-place on `adata`."""
    lib.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)
    lib.tl.latent_time(adata)
