"""`scvelo_rs.datasets` — pass-through to `scvelo.datasets`."""

from __future__ import annotations

import scvelo as _scv

_attrs = [n for n in dir(_scv.datasets) if not n.startswith("_")]
for _name in _attrs:
    globals()[_name] = getattr(_scv.datasets, _name)
del _name
del _scv

__all__ = _attrs
del _attrs
