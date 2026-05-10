"""End-to-end scvelo-rs pipeline on the pancreas tutorial dataset.

Runs the full RNA velocity workflow with Rust-backed kernels and prints a
summary at each step. This is what an existing scvelo user would write after
swapping `import scvelo as scv` for `import scvelo_rs as scv`.

Run:

    python examples/pancreas_end_to_end.py

Approximate runtime: ~10s on a laptop. Downloads the pancreas dataset on first
run (~30 MB cached under ~/.scvelo).
"""

from __future__ import annotations

import time
import warnings

import scvelo_rs as scv

warnings.filterwarnings("ignore")


def main() -> None:
    print("Loading pancreas dataset...")
    adata = scv.datasets.pancreas()
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    t0 = time.time()
    print("\nPreprocessing (filter_and_normalize + moments)...")
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nFitting dynamical model (Rust)...")
    t = time.time()
    scv.tl.recover_dynamics(adata, var_names="all", show_progress_bar=False)
    n_fit = int((~adata.var["fit_alpha"].isna()).sum())
    print(f"  fit {n_fit}/{adata.n_vars} genes in {time.time() - t:.1f}s")

    print("\nComputing velocities (Rust)...")
    t = time.time()
    scv.tl.velocity(adata, mode="dynamical")
    print(f"  done in {time.time() - t:.1f}s")

    print("\nBuilding velocity graph (Rust)...")
    t = time.time()
    scv.tl.velocity_graph(adata, show_progress_bar=False)
    print(f"  graph nnz = {adata.uns['velocity_graph'].nnz}, done in {time.time() - t:.1f}s")

    print("\nDownstream analysis (pure pass-through to scvelo):")
    t = time.time()
    scv.tl.velocity_pseudotime(adata)
    scv.tl.velocity_confidence(adata)
    print(f"  pseudotime + confidence in {time.time() - t:.1f}s")

    print("\nResults:")
    print(
        f"  fit_alpha range:      {adata.var['fit_alpha'].min():.3f} .. "
        f"{adata.var['fit_alpha'].max():.3f}"
    )
    print(f"  fit_likelihood mean:  {adata.var['fit_likelihood'].mean():.3f}")
    print(f"  velocity_confidence:  {adata.obs['velocity_confidence'].mean():.3f} mean")
    print("\nDone. To plot velocity arrows, install matplotlib then run:")
    print("    scv.pl.velocity_embedding_stream(adata, basis='umap')")


if __name__ == "__main__":
    main()
