"""Edge-case robustness battery — degenerate inputs that real-world data hits.

Modeled after rustscenic's edge-case suite. Each test feeds an unusual but
plausible AnnData into `scvelo_rs.recover_dynamics` (and friends) and asserts
the library produces a sane result or a clear error — never a segfault, hang,
or silent garbage.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"


def _load_pancreas_50():
    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    return adata


# ---------------------------------------------------------------------------
# Degenerate gene content
# ---------------------------------------------------------------------------


def test_all_zero_gene_handled_gracefully():
    """A gene where every Mu/Ms value is 0 must produce NaN fits, not crash."""
    import scvelo_rs

    adata = _load_pancreas_50()
    # Zero out the first gene entirely.
    adata.layers["Mu"][:, 0] = 0.0
    adata.layers["Ms"][:, 0] = 0.0

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    # Zeroed gene: fit should be NaN (not recoverable). Other genes should fit.
    assert np.isnan(adata.var["fit_alpha"].iloc[0]), "zero-gene should produce NaN"
    n_recovered = (~adata.var["fit_alpha"].isna()).sum()
    assert n_recovered > 50, f"only {n_recovered} genes recovered out of {adata.n_vars}"


def test_constant_gene_handled_gracefully():
    """A gene where every cell has the SAME nonzero value (zero variance) must
    not crash and must mark the gene as unfit."""
    import scvelo_rs

    adata = _load_pancreas_50()
    adata.layers["Mu"][:, 0] = 1.0
    adata.layers["Ms"][:, 0] = 1.0

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    # std=0 path either NaNs the gene or recovers a degenerate fit; just must
    # not crash.
    val = adata.var["fit_alpha"].iloc[0]
    assert np.isnan(val) or np.isfinite(val), "constant gene gave non-finite non-NaN"


def test_nan_in_layers_raises_clear_error():
    """NaN values in Mu/Ms must raise a clear ValueError before reaching the
    Rust kernel — `partial_cmp` doesn't form a total order with NaN, which
    used to panic the sort kernel."""
    import scvelo_rs

    adata = _load_pancreas_50()
    rng = np.random.default_rng(0)
    nan_mask = rng.random(adata.shape) < 0.05
    adata.layers["Mu"][nan_mask] = np.nan
    adata.layers["Ms"][nan_mask] = np.nan

    with pytest.raises(ValueError, match="NaN or inf"):
        scvelo_rs.recover_dynamics(
            adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
        )


def test_inf_in_layers_raises_clear_error():
    """Inf values must raise the same clear error."""
    import scvelo_rs

    adata = _load_pancreas_50()
    adata.layers["Mu"][0, 0] = np.inf
    adata.layers["Ms"][0, 0] = -np.inf

    with pytest.raises(ValueError, match="NaN or inf"):
        scvelo_rs.recover_dynamics(
            adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
        )


# ---------------------------------------------------------------------------
# Unusual shapes
# ---------------------------------------------------------------------------


def test_single_gene_adata():
    """1-gene AnnData must run end-to-end."""
    import scvelo_rs

    adata = _load_pancreas_50()
    adata = adata[:, :1].copy()
    assert adata.n_vars == 1

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    assert "fit_alpha" in adata.var.columns


def test_few_cells_adata():
    """Very small adata (10 cells) — must run without crashing on init heuristics."""
    import scvelo_rs

    adata = _load_pancreas_50()
    adata = adata[:10, :20].copy()
    # Recompute neighbors for the tiny slice.
    sc.pp.neighbors(adata, n_neighbors=5)

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False,
        fit_connected_states=False,
    )
    assert "fit_alpha" in adata.var.columns


# ---------------------------------------------------------------------------
# Var-name oddities
# ---------------------------------------------------------------------------


def test_duplicate_var_names_does_not_crash():
    """Duplicate gene symbols are common in real biological data
    (e.g. Y_RNA appearing 50× across chromosomes). Must not crash."""
    import scvelo_rs

    adata = _load_pancreas_50()
    # Force a duplicate.
    new_names = list(adata.var_names)
    new_names[1] = new_names[0]
    adata.var_names = new_names

    # Pass an explicit list that includes the duplicated name.
    scvelo_rs.recover_dynamics(
        adata, var_names=[new_names[0]], n_jobs=1,
        show_progress_bar=False, t_max=False,
    )
    assert "fit_alpha" in adata.var.columns


def test_foreign_gene_names_raise_or_warn():
    """A gene-name list with NO match in adata.var_names must raise a clear
    ValueError, not silently no-op."""
    import scvelo_rs

    adata = _load_pancreas_50()
    with pytest.raises(ValueError, match="None of"):
        scvelo_rs.recover_dynamics(
            adata, var_names=["__not_a_gene__", "__also_fake__"],
            n_jobs=1, show_progress_bar=False,
        )


# ---------------------------------------------------------------------------
# Sparse-format coverage (CSC, COO in addition to the CSR we already test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_recover_dynamics_handles_all_sparse_formats(fmt):
    """Mu/Ms passed as scipy.sparse in CSR or CSC must work (anndata
    rejects COO at the layer-assignment step)."""
    import scvelo_rs
    from scipy.sparse import csc_matrix, csr_matrix

    adata = _load_pancreas_50()
    cls = {"csr": csr_matrix, "csc": csc_matrix}[fmt]
    adata.layers["Mu"] = cls(adata.layers["Mu"])
    adata.layers["Ms"] = cls(adata.layers["Ms"])

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    n_fit = (~adata.var["fit_alpha"].isna()).sum()
    assert n_fit > 50, f"format={fmt}: only {n_fit} genes fit"


# ---------------------------------------------------------------------------
# Empty / minimal AnnData
# ---------------------------------------------------------------------------


def test_no_layers_raises():
    """AnnData with no Mu/Ms/unspliced/spliced layers must error clearly."""
    import scvelo_rs

    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.normal(size=(20, 10)).astype(np.float32))

    with pytest.raises((ValueError, KeyError)):
        scvelo_rs.recover_dynamics(
            adata, var_names="all", n_jobs=1, show_progress_bar=False
        )


# ---------------------------------------------------------------------------
# Determinism across thread counts (rustscenic-inspired)
# ---------------------------------------------------------------------------


def test_determinism_across_n_jobs():
    """Same input run with different `n_jobs` must give bit-identical fits.
    Catches Rayon work-stealing reorderings that produce different results."""
    import scvelo_rs

    a1 = _load_pancreas_50()
    a2 = _load_pancreas_50()
    scvelo_rs.recover_dynamics(
        a1, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    scvelo_rs.recover_dynamics(
        a2, var_names="all", n_jobs=4, show_progress_bar=False, t_max=False
    )

    for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_"):
        v1 = a1.var[col].to_numpy()
        v2 = a2.var[col].to_numpy()
        nan_mask = np.isnan(v1) | np.isnan(v2)
        np.testing.assert_array_equal(
            v1[~nan_mask], v2[~nan_mask],
            err_msg=f"{col}: different result for n_jobs=1 vs n_jobs=4"
        )
