"""CellRank 2 hematopoiesis — scvelo dynamical pipeline + CellRank fate mapping.

Adapted from theislab/cellrank2_reproducibility and the official CellRank
tutorial (200_rna_velocity). BSD-3-Clause; see ./LICENSE.

The bench harness calls `load_data()` once, then `run(lib, adata)` for each
backend with a fresh `.copy()` of the same AnnData.
"""

from __future__ import annotations


def load_data():
    """Fetch the Setty 2019 CD34+ bone marrow dataset (cached under ~/.cache)."""
    import cellrank as cr

    return cr.datasets.bone_marrow()


def run(lib, adata) -> None:
    """Execute scvelo dynamical pipeline + CellRank fate mapping in-place."""
    import cellrank as cr

    lib.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    lib.pp.moments(adata, n_pcs=30, n_neighbors=30)
    lib.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    lib.tl.velocity(adata, mode="dynamical")
    lib.tl.velocity_graph(adata, show_progress_bar=False)

    # CellRank downstream — same code path under both backends. Not Rust-accelerated;
    # included so the bench reflects the full real-user workflow.
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    combined = 0.8 * vk + 0.2 * ck

    g = cr.estimators.GPCCA(combined)
    g.fit(n_states=5)
    g.predict_terminal_states()
    g.compute_fate_probabilities()
