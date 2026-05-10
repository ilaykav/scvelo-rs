"""Monkey-patch entry: speed up an existing scvelo pipeline with one extra import.

Useful when you have an existing notebook / script using `import scvelo as scv`
and don't want to touch the rest of the code. Importing `scvelo_rs.patch`
swaps `scv.tl.{recover_dynamics,velocity,velocity_graph}` to point at the Rust
kernels — every downstream call benefits transparently.

Originals are preserved at `scv.tl.<name>_original` for A/B comparison.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    import scvelo as scv

    # The one-line speedup. Apply BEFORE building your AnnData / running tl.
    import scvelo_rs.patch  # noqa: F401

    print("Patched. scv.tl.recover_dynamics now points at:",
          scv.tl.recover_dynamics.__module__)
    print("Original preserved at scv.tl.recover_dynamics_original:",
          scv.tl.recover_dynamics_original.__module__)

    # Standard scvelo workflow — unchanged.
    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata)

    print("\nRunning recover_dynamics through patched scv.tl ...")
    scv.tl.recover_dynamics(adata, show_progress_bar=False)
    print(f"  fit {(~adata.var['fit_alpha'].isna()).sum()} genes")

    print("\nTo revert (e.g. for A/B comparison):")
    print("  scvelo_rs.patch.revert()")


if __name__ == "__main__":
    main()
