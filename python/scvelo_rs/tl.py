"""`scvelo_rs.tl` — drop-in for `scvelo.tl`.

Functions with Rust hot loops:
  - `recover_dynamics`         dynamical-model EM (full Rust)
  - `velocity_graph`           per-cell cosine kernel (full Rust)
  - `velocity`                 deterministic mode in Rust; stochastic / dynamical
                               fall through to scvelo

Pass-throughs to scvelo (mostly thin scipy.sparse / scanpy wrappers,
not bottlenecks):
  - `transition_matrix`, `terminal_states`, `velocity_pseudotime`,
    `velocity_embedding`, `velocity_confidence`, `velocity_confidence_transition`,
    `score_genes_cell_cycle`, `rank_velocity_genes`, `paga`, `differential_kinetic_test`,
    `latent_time`, `score_robustness`

Use::

    import scvelo_rs as scv
    scv.tl.recover_dynamics(adata)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.tl.transition_matrix(adata)   # scvelo upstream (no hot loop)
"""

from __future__ import annotations

# Pass-through every other function from scvelo.tl. These do not contain
# significant hot loops; the scipy.sparse and scanpy.tl.{DPT,PAGA} backing
# stores already release the GIL on heavy ops.
import scvelo as _scv

from ._dynamics import recover_dynamics
from ._velocity import velocity, velocity_graph

_PASSTHROUGH = (
    "transition_matrix",
    "terminal_states",
    "velocity_pseudotime",
    "velocity_embedding",
    "velocity_confidence",
    "velocity_confidence_transition",
    "score_genes_cell_cycle",
    "rank_velocity_genes",
    "paga",
    "differential_kinetic_test",
    "latent_time",
    "score_robustness",
    "velocity_clusters",
    "velocity_genes",
)

for _name in _PASSTHROUGH:
    if hasattr(_scv.tl, _name):
        # Bind the underlying scvelo function unmodified. If `scvelo_rs.patch`
        # has been imported, scvelo's `recover_dynamics`/`velocity`/
        # `velocity_graph` already point at our Rust paths, so any pass-through
        # transitively benefits.
        globals()[_name] = getattr(_scv.tl, _name)
del _name

del _scv

__all__ = list(_PASSTHROUGH) + ["recover_dynamics", "velocity", "velocity_graph"]
