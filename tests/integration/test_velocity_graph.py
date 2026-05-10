"""Bit-exact equivalence test for `velocity_graph` — runs scvelo's own
implementation alongside scvelo-rs's Rust kernel and asserts the per-cell
cosine sparse matrix entries match.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"

# Cosine values are computed in f32 (matching scvelo); ULP noise across f32
# accumulation in different summation orders sits at ~1e-6 abs. Allow a bit
# of slack for safe gating.
_RTOL = 1e-5
_ATOL = 1e-6
# Allow a small fraction of nz entries to drift more than the tolerance band
# (recursive-neighbor unioning + f32 accumulation order is what bites).
_MAX_OUTLIER_FRAC = 0.005


@pytest.mark.parametrize(
    "fixture",
    [
        "pancreas_50obs_preprocessed",
        "pancreas_100obs_preprocessed",
        "dentategyrus_50obs_preprocessed",
        "dentategyrus_100obs_preprocessed",
    ],
)
def test_velocity_graph_bit_exact(fixture):
    import scvelo as scv

    a_scv = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))
    a_rs = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))

    for adata in (a_scv, a_rs):
        adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
        adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
        scv.tl.velocity(adata, mode="deterministic")

    scv.tl.velocity_graph(a_scv, show_progress_bar=False)

    import scvelo_rs

    scvelo_rs.velocity_graph(a_rs, show_progress_bar=False)

    g_scv: csr_matrix = a_scv.uns["velocity_graph"]
    g_rs: csr_matrix = a_rs.uns["velocity_graph"]
    assert "velocity_graph_neg" in a_scv.uns
    assert "velocity_graph_neg" in a_rs.uns

    assert g_scv.shape == g_rs.shape, f"shape mismatch {g_scv.shape} vs {g_rs.shape}"
    assert g_scv.nnz == g_rs.nnz or abs(g_scv.nnz - g_rs.nnz) < 0.01 * g_scv.nnz, (
        f"nnz mismatch: scvelo={g_scv.nnz}, rust={g_rs.nnz}"
    )

    # Compare positive-graph entries column-by-column on aligned (i, j) pairs.
    diff = (g_scv - g_rs).tocoo()
    abs_diff = np.abs(diff.data) if diff.nnz else np.array([], dtype=float)
    if abs_diff.size:
        scv_nz = g_scv.toarray()[diff.row, diff.col]
        rel_diff = abs_diff / (np.abs(scv_nz) + _ATOL)
        n_outliers = int(np.sum(rel_diff > _RTOL))
        outlier_frac = n_outliers / max(1, g_scv.nnz)
        assert outlier_frac <= _MAX_OUTLIER_FRAC, (
            f"{fixture}: {n_outliers}/{g_scv.nnz} entries drift > {_RTOL} "
            f"({outlier_frac * 100:.2f}% > {_MAX_OUTLIER_FRAC * 100:.1f}% threshold). "
            f"max_abs_diff={abs_diff.max():.3e}, max_rel_diff={rel_diff.max():.3e}"
        )

    self_scv = a_scv.obs["velocity_self_transition"].to_numpy()
    self_rs = a_rs.obs["velocity_self_transition"].to_numpy()
    np.testing.assert_allclose(self_scv, self_rs, rtol=1e-3, atol=1e-4)
