"""`scvelo_rs.pl` — pass-through to `scvelo.pl` (matplotlib plotting).

Plotting will stay in Python / matplotlib indefinitely; nothing to gain
from a Rust port here.
"""

from __future__ import annotations

import scvelo as _scv

# Re-export the entire scvelo.pl module surface.
_attrs = [n for n in dir(_scv.pl) if not n.startswith("_")]
for _name in _attrs:
    globals()[_name] = getattr(_scv.pl, _name)
del _name
del _scv
del annotations

__all__ = _attrs
del _attrs
