"""`scvelo_rs.pp` — drop-in for `scvelo.pp`.

Pass-through to scvelo. New helpers added by scvelo flow through
automatically.
"""

import scvelo as _scv

from ._pp import neighbors  # noqa: F401

_OVERRIDDEN = {"neighbors"}

for _name in dir(_scv.pp):
    if not _name.startswith("_") and _name not in _OVERRIDDEN:
        globals()[_name] = getattr(_scv.pp, _name)
del _name
del _scv

__all__ = sorted(n for n in globals() if not n.startswith("_"))
