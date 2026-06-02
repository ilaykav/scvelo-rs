"""Bit-exact equivalence test for `tl.velocity` (deterministic mode)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"


@pytest.mark.parametrize(
    "fixture",
    [
        "pancreas_50obs_preprocessed",
        "pancreas_100obs_preprocessed",
        "dentategyrus_50obs_preprocessed",
        "dentategyrus_100obs_preprocessed",
    ],
)
def test_velocity_deterministic_matches_scvelo(fixture):
    import scvelo as scv
    import scvelo_rs

    a_scv = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))
    a_rs = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))
    for adata in (a_scv, a_rs):
        adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
        adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)

    scv.tl.velocity(a_scv, mode="deterministic")
    scvelo_rs.velocity(a_rs, mode="deterministic")

    # gamma + r2 - primary signal, expect bit-exact at f64 (small ULP noise OK).
    np.testing.assert_allclose(
        a_scv.var["velocity_gamma"].to_numpy().astype(np.float64),
        a_rs.var["velocity_gamma"].to_numpy().astype(np.float64),
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        a_scv.var["velocity_r2"].to_numpy().astype(np.float64),
        a_rs.var["velocity_r2"].to_numpy().astype(np.float64),
        rtol=1e-9,
        atol=1e-12,
    )

    # velocity_genes mask: must match exactly.
    g_scv = a_scv.var["velocity_genes"].to_numpy().astype(bool)
    g_rs = a_rs.var["velocity_genes"].to_numpy().astype(bool)
    n_diff = int(np.sum(g_scv != g_rs))
    assert n_diff == 0, f"{fixture}: velocity_genes diff on {n_diff} genes"

    # residual layer (stored as f32).
    np.testing.assert_allclose(
        np.asarray(a_scv.layers["velocity"], dtype=np.float64),
        np.asarray(a_rs.layers["velocity"], dtype=np.float64),
        rtol=1e-5,
        atol=1e-7,
    )
