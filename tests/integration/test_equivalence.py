"""scvelo-test-suite-style equivalence tests.

Mirrors scvelo's own test patterns: load a tutorial dataset, run recover_dynamics
through both scvelo and scvelo-rs, assert output schema parity + numerical
proximity. These are the assertions a scvelo user would expect to hold when
swapping our wrapper in.

Run via `pytest tests/test_scvelo_equivalence.py -v`.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest

# Per-gene relative tolerance for a fit to count as "matching" scvelo.
# Some NM-trajectory drift remains (~1-15% per param on ~half the genes), so
# we set a generous tolerance and assert that the MEDIAN drift is small.
PER_GENE_REL = 0.30  # up to 30% on outliers
MEDIAN_REL = 0.10  # half the genes within 10%


def _load_pancreas_slice(n_genes: int = 10):
    import scvelo as scv

    adata = scv.datasets.pancreas()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    return adata[:, :n_genes].copy()


@pytest.fixture(scope="module")
def pancreas_pair():
    """Run scvelo + scvelo_rs side-by-side once, share results across tests."""
    scvelo = pytest.importorskip("scvelo")
    import scvelo_rs

    adata = _load_pancreas_slice(n_genes=10)
    a_stock = adata.copy()
    a_rust = adata.copy()

    # Stock scvelo, t_max=False so output is the raw NM fit (no align rescale).
    scvelo.tl.recover_dynamics(
        a_stock,
        var_names="all",
        n_jobs=1,
        fit_connected_states=True,
        show_progress_bar=False,
        t_max=False,
    )
    # Our direct-call kernel, same flag.
    scvelo_rs.recover_dynamics(
        a_rust,
        var_names="all",
        n_jobs=1,
        fit_connected_states=True,
        show_progress_bar=False,
        t_max=False,
    )
    return a_stock, a_rust


# --- Schema parity ---------------------------------------------------------

REQUIRED_VAR_COLS = (
    "fit_alpha",
    "fit_beta",
    "fit_gamma",
    "fit_t_",
    "fit_scaling",
    "fit_likelihood",
    "fit_variance",
)
REQUIRED_LAYER_KEYS = ("fit_t", "fit_tau", "fit_tau_")


def test_schema_var_columns(pancreas_pair):
    a_stock, a_rust = pancreas_pair
    for col in REQUIRED_VAR_COLS:
        assert col in a_rust.var, f"missing {col} in adata.var"
        assert a_rust.var[col].shape == a_stock.var[col].shape, (
            f"{col}: shape {a_rust.var[col].shape} != stock {a_stock.var[col].shape}"
        )


def test_schema_layers(pancreas_pair):
    _a_stock, a_rust = pancreas_pair
    for k in REQUIRED_LAYER_KEYS:
        assert k in a_rust.layers, f"missing layer {k}"
        assert a_rust.layers[k].shape == a_rust.shape, (
            f"layer {k} shape {a_rust.layers[k].shape} != adata shape {a_rust.shape}"
        )


# --- NaN pattern parity ----------------------------------------------------


def test_nan_pattern_matches(pancreas_pair):
    a_stock, a_rust = pancreas_pair
    for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_"):
        nan_stock = np.isnan(a_stock.var[col].to_numpy())
        nan_rust = np.isnan(a_rust.var[col].to_numpy())
        assert np.array_equal(nan_stock, nan_rust), (
            f"{col}: NaN pattern differs in {(nan_stock != nan_rust).sum()} entries"
        )


# --- Numerical proximity ---------------------------------------------------


def _drift_stats(a_stock, a_rust, col):
    sv = a_stock.var[col].to_numpy()
    rs = a_rust.var[col].to_numpy()
    mask = ~np.isnan(sv) & ~np.isnan(rs)
    if mask.sum() == 0:
        return 0.0, 0.0
    rel = np.abs(sv[mask] - rs[mask]) / (np.abs(sv[mask]) + 1e-12)
    return float(rel.max()), float(np.median(rel))


# Per-column tolerances. fit_t_ has slightly higher per-gene drift because
# stages 4 and 6 (which optimise t_) are most sensitive to NM-trajectory
# variation; rates are tighter.
PER_COL_REL = {
    "fit_alpha": 0.30,
    "fit_beta": 0.30,
    "fit_gamma": 0.30,
    # `fit_t_` retains a wider tolerance: a small fraction of genes per
    # slice land in a different but equally-valid local minimum (NM
    # trajectory drift on f32 input layers). Most are within 10%.
    "fit_t_": 0.50,
    "fit_scaling": 0.10,
}


@pytest.mark.parametrize("col", list(PER_COL_REL.keys()))
def test_per_gene_rel_drift(pancreas_pair, col):
    a_stock, a_rust = pancreas_pair
    max_rel, median_rel = _drift_stats(a_stock, a_rust, col)
    print(f"\n  {col}: max_rel={max_rel:.4f}  median_rel={median_rel:.4f}")
    assert max_rel < PER_COL_REL[col], (
        f"{col}: max relative drift {max_rel:.4%} exceeds tolerance {PER_COL_REL[col]:.0%}"
    )
    assert median_rel < MEDIAN_REL, (
        f"{col}: median relative drift {median_rel:.4%} exceeds tolerance {MEDIAN_REL:.0%}"
    )


# --- Idempotency / determinism --------------------------------------------


def test_determinism():
    """Two runs of scvelo_rs.recover_dynamics on identical data give identical output."""
    import scvelo_rs

    adata = _load_pancreas_slice(n_genes=8)
    a1 = adata.copy()
    a2 = adata.copy()
    scvelo_rs.recover_dynamics(
        a1,
        var_names="all",
        n_jobs=1,
        fit_connected_states=True,
        show_progress_bar=False,
        t_max=False,
    )
    scvelo_rs.recover_dynamics(
        a2,
        var_names="all",
        n_jobs=1,
        fit_connected_states=True,
        show_progress_bar=False,
        t_max=False,
    )

    for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_"):
        v1 = a1.var[col].to_numpy()
        v2 = a2.var[col].to_numpy()
        nan_mask = np.isnan(v1) | np.isnan(v2)
        assert np.array_equal(np.isnan(v1), np.isnan(v2))
        assert np.array_equal(v1[~nan_mask], v2[~nan_mask]), (
            f"{col}: non-deterministic - runs differ on {(v1 != v2)[~nan_mask].sum()} cells"
        )
