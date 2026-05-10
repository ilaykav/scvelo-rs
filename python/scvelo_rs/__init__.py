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
(`tl`, `pp`, `pl`, `datasets`, `utils`). Hot loops route through Rust;
everything else passes through to scvelo unchanged. Patch-style adoption
also works:

    import scvelo as scv
    import scvelo_rs.patch  # noqa: F401  — scv.tl.recover_dynamics et al. now Rust
"""

# Submodule namespaces — `tl`, `pp`, `pl`, `datasets`, `utils` mirror scvelo.
# Top-level scvelo helpers — exposed as `scvelo_rs.<name>` to mirror `scv.<name>`.
import scvelo as _scv

from . import datasets, pl, pp, tl, utils
from ._dynamics import recover_dynamics
from ._scvelo_rs import __version__, project_to_curve_kernel
from ._velocity import velocity, velocity_graph

AnnData = _scv.AnnData
GridSpec = _scv.GridSpec
Neighbors = _scv.Neighbors
Velocity = _scv.Velocity
VelocityGraph = _scv.VelocityGraph
get_df = _scv.get_df
load = _scv.load
logging = _scv.logging
read_csv = _scv.read_csv
read_load = _scv.read_load
set_figure_params = _scv.set_figure_params
settings = _scv.settings

# Full-name submodule aliases (scvelo exposes both pp/preprocessing, tl/tools, pl/plotting).
preprocessing = pp
tools = tl
plotting = pl

del _scv

__all__ = [
    # submodules
    "tl",
    "pp",
    "pl",
    "datasets",
    "utils",
    "preprocessing",
    "tools",
    "plotting",
    # Rust-backed
    "recover_dynamics",
    "velocity",
    "velocity_graph",
    "project_to_curve_kernel",
    # scvelo top-level helpers
    "AnnData",
    "GridSpec",
    "Neighbors",
    "Velocity",
    "VelocityGraph",
    "get_df",
    "load",
    "logging",
    "read_csv",
    "read_load",
    "set_figure_params",
    "settings",
    # metadata
    "__version__",
]
