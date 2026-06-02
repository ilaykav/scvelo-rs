"""PBMC 68k pipeline - Zheng 2017 frozen PBMCs, full scvelo dynamical workflow.

Heavy mid-large benchmark (~68k cells immune compartment). BSD-3-Clause;
see ./README.md for citation and license.
"""

from __future__ import annotations


def load_data():
    """Fetch the Zheng 2017 PBMC 68k dataset (cached under ~/.scvelo)."""
    import scvelo as scv

    return scv.datasets.pbmc68k()


def run(lib, adata) -> None:
    """Execute the full dynamical-model pipeline in-place on `adata`."""
    import numpy as np
    import scanpy as sc

    lib.pp.filter_genes(adata, min_shared_counts=20)
    lib.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # Cast Mu/Ms to f64 so both backends share the same numerical environment.
    for k in ("Mu", "Ms"):
        if k in adata.layers:
            adata.layers[k] = np.asarray(adata.layers[k], dtype=np.float64)
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)
    lib.tl.latent_time(adata)
