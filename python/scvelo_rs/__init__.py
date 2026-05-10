"""Drop-in replacement for scvelo. Use exactly like scvelo:

    import scvelo_rs as scv
    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)
    scv.tl.recover_dynamics(adata)        # Rust kernel (134x faster)
    scv.tl.velocity(adata)                # Rust kernel for deterministic mode
    scv.tl.velocity_graph(adata)          # Rust kernel
    scv.tl.transition_matrix(adata)       # passes through to scvelo

Everything that scvelo exposes works through the same submodule namespaces
(`tl`, `pp`, `pl`, `datasets`). Hot loops route through Rust; everything else
passes through to scvelo unchanged. Patch-style adoption also works:

    import scvelo as scv
    import scvelo_rs.patch  # noqa: F401  — scv.tl.recover_dynamics et al. now Rust
"""

# Submodule namespaces — `tl`, `pp`, `pl`, `datasets` mirror scvelo.
from . import datasets, pl, pp, tl
from ._dynamics import recover_dynamics
from ._scvelo_rs import __version__, project_to_curve_kernel
from ._velocity import velocity, velocity_graph

__all__ = [
    "tl",
    "pp",
    "pl",
    "datasets",
    "recover_dynamics",
    "velocity",
    "velocity_graph",
    "project_to_curve_kernel",
    "__version__",
]
