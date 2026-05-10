"""Smoke tests — imports, kernel call, AnnData glue, and signature parity."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_import():
    import scvelo_rs

    assert scvelo_rs.__version__


def test_kernel_module_loads():
    """The native module loads and exposes the expected entry points."""
    from scvelo_rs._scvelo_rs import (
        align_dynamics_kernel,
        per_gene_weights_kernel,
        recover_dynamics_kernel,
        splicing_dynamics_eval_kernel,
    )

    # Just verify they exist as callables.
    for fn in (
        recover_dynamics_kernel,
        per_gene_weights_kernel,
        align_dynamics_kernel,
        splicing_dynamics_eval_kernel,
    ):
        assert callable(fn)


def test_splicing_dynamics_eval_kernel_smoke():
    """Smallest possible call to confirm the splicing ODE kernel returns right shape."""
    from scvelo_rs._scvelo_rs import splicing_dynamics_eval_kernel

    t = np.linspace(0, 5, 50)
    u, s = splicing_dynamics_eval_kernel(t, 1.5, 0.5, 0.4, 0.0, 0.0)
    assert u.shape == (50,) and s.shape == (50,)
    # u should be monotonically increasing from 0 toward alpha/beta = 3.0
    assert u[0] < u[-1]
    assert 0.0 <= u[-1] <= 1.5 / 0.5


def test_signature_matches_scvelo():
    """Drop-in guarantee: every scvelo recover_dynamics kwarg is accepted by ours.

    Migration breaks if upstream adds a kwarg we don't accept, or if our
    defaults drift from theirs. This test fails loud when that happens.
    """
    scvelo = pytest.importorskip("scvelo")
    from scvelo_rs import recover_dynamics

    upstream = inspect.signature(scvelo.tl.recover_dynamics)
    ours = inspect.signature(recover_dynamics)

    upstream_params = {name: p for name, p in upstream.parameters.items() if name != "data"}
    our_params = {name: p for name, p in ours.parameters.items() if name != "data"}

    missing = set(upstream_params) - set(our_params) - {"kwargs"}
    extra_required = {
        name
        for name, p in our_params.items()
        if name not in upstream_params
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    assert not missing, f"missing kwargs vs scvelo: {missing}"
    assert not extra_required, f"new required kwargs not in scvelo: {extra_required}"

    drifted = []
    for name, up in upstream_params.items():
        if name in our_params and up.default is not inspect.Parameter.empty:
            ours_default = our_params[name].default
            if ours_default != up.default:
                drifted.append((name, up.default, ours_default))
    assert not drifted, f"default drift vs scvelo: {drifted}"


def test_monkey_patch_roundtrip():
    """apply/revert leaves scvelo's `recover_dynamics` pointing at the upstream
    version. Anchored on the preserved `recover_dynamics_original` attribute
    rather than capturing the live ref, so this stays robust to other tests
    importing `scvelo_rs.patch`.
    """
    scvelo = pytest.importorskip("scvelo")
    import scvelo_rs.patch

    original = getattr(scvelo.tl, "recover_dynamics_original", None)
    if original is None:
        scvelo_rs.patch.apply(verbose=False)
        original = scvelo.tl.recover_dynamics_original

    scvelo_rs.patch.apply(verbose=False)
    assert scvelo.tl.recover_dynamics is not original

    scvelo_rs.patch.revert()
    assert scvelo.tl.recover_dynamics is original
