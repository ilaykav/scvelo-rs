"""`scvelo_rs.tl` - drop-in for `scvelo.tl`.

Rust-backed:
  - `recover_dynamics`        dynamical-model EM (full Rust)
  - `velocity`                deterministic mode in Rust; stochastic /
                              dynamical fall through to scvelo
  - `velocity_graph`          per-cell cosine kernel (full Rust)

Everything else `scv.tl` exposes passes through dynamically. New `scv.tl`
additions in future scvelo releases flow through automatically.
"""

import scvelo as _scv

from ._diff_kinetic import differential_kinetic_test  # noqa: F401
from ._dynamics import recover_dynamics  # noqa: F401
from ._velocity import velocity, velocity_graph  # noqa: F401

_OVERRIDDEN = {
    "recover_dynamics",
    "velocity",
    "velocity_graph",
    "velocity_embedding",
    "recover_latent_time",
    "differential_kinetic_test",
}


# TODO(#5): port to Rust.
def velocity_embedding(*args, **kwargs):
    return _scv.tl.velocity_embedding(*args, **kwargs)


for _name in dir(_scv.tl):
    if not _name.startswith("_") and _name not in _OVERRIDDEN:
        globals()[_name] = getattr(_scv.tl, _name)
del _name

__all__ = sorted(n for n in globals() if not n.startswith("_") and n != "_scv")
