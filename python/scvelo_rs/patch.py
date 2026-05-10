"""Monkey-patch entry point.

After ::

    import scvelo as scv
    import scvelo_rs.patch    # noqa: F401

every call to `scv.tl.<function>` covered by scvelo-rs routes through Rust.
Originals are preserved at `<function>_original` for A/B comparison.
"""

from __future__ import annotations

import warnings

try:
    import scvelo as _scv
except ImportError as exc:
    raise ImportError(
        "scvelo_rs.patch requires scvelo. Install with `pip install scvelo`."
    ) from exc

from ._dynamics import recover_dynamics as _fast_recover_dynamics
from ._velocity import velocity as _fast_velocity
from ._velocity import velocity_graph as _fast_velocity_graph

# (attr_name, fast_replacement)
_PATCHED = (
    ("recover_dynamics", _fast_recover_dynamics),
    ("velocity", _fast_velocity),
    ("velocity_graph", _fast_velocity_graph),
)


def apply(verbose: bool = True) -> None:
    """Replace `scv.tl.<function>` with the Rust-backed versions. Idempotent."""
    for name, fast in _PATCHED:
        orig_attr = f"{name}_original"
        if not hasattr(_scv.tl, orig_attr):
            setattr(_scv.tl, orig_attr, getattr(_scv.tl, name))
        setattr(_scv.tl, name, fast)
    if verbose:
        names = ", ".join(n for n, _ in _PATCHED)
        warnings.warn(
            f"scvelo_rs.patch: scv.tl.{{{names}}} routed through Rust. "
            "Originals preserved at scv.tl.<name>_original.",
            stacklevel=2,
        )


def revert() -> None:
    """Restore upstream scvelo functions."""
    for name, _ in _PATCHED:
        orig_attr = f"{name}_original"
        if hasattr(_scv.tl, orig_attr):
            setattr(_scv.tl, name, getattr(_scv.tl, orig_attr))


apply(verbose=False)
