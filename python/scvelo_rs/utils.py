"""`scvelo_rs.utils` — drop-in for `scvelo.utils`.

Pass-through to scvelo. New helpers added by scvelo flow through
automatically.
"""

import scvelo as _scv

_OVERRIDDEN = {
    "compute_velocity_on_grid",
    "compute_dynamics",
}


# TODO(#5): port to Rust.
def compute_velocity_on_grid(*args, **kwargs):
    return _scv.utils.compute_velocity_on_grid(*args, **kwargs)


# TODO(#4): port to Rust.
def compute_dynamics(*args, **kwargs):
    return _scv.utils.compute_dynamics(*args, **kwargs)


for _name in dir(_scv.utils):
    if not _name.startswith("_") and _name not in _OVERRIDDEN:
        globals()[_name] = getattr(_scv.utils, _name)
del _name

__all__ = sorted(n for n in globals() if not n.startswith("_") and n != "_scv")
