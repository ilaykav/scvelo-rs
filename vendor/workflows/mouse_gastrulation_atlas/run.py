"""Mouse gastrulation atlas — atlas-scale scvelo dynamical pipeline.

Atlas-scale (~116,000 cells) workflow that follows the standard scvelo
dynamical-model shape on the Pijuan-Sala 2019 dataset. The bench harness
skips the stock-scvelo run here (documented OOM/timeout); only scvelo-rs
is timed.

BSD-3-Clause; see ./LICENSE.
"""

from __future__ import annotations


def load_data():
    """Fetch the Pijuan-Sala 2019 mouse gastrulation atlas (~400 MB, cached)."""
    import scvelo as scv

    return scv.datasets.gastrulation()


def run(lib, adata) -> None:
    """Execute the dynamical pipeline at atlas scale in-place."""
    # Preprocessing: split filter_and_normalize and use scanpy for HVGs
    # because scvelo 0.3.4's combined helper forwards n_top_genes
    # incorrectly through **kwargs.
    import scanpy as sc

    lib.pp.filter_genes(adata, min_shared_counts=20)
    lib.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)
