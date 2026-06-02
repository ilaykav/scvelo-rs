"""Dual-run equivalence test for `differential_kinetic_test`.

Runs scvelo's `tl.differential_kinetic_test` and scvelo_rs's drop-in side by
side on the same input, and asserts the writeback (`var['fit_diff_kinetics']`,
`var['fit_pval_kinetics']`, `varm['fit_pvals_kinetics']`,
`uns['recover_dynamics']['fit_diff_kinetics']`) matches scvelo within f64 ULP
noise. Also reports wall-time savings.

scvelo_rs's `differential_kinetic_test` recomputes per-gene LRT from raw
inputs via a Rust kernel that mirrors scvelo's `_em_model_core` flow
(initialize_diff_kinetics → get_variance → get_cluster_mse → get_orth_fit →
get_pval_diff_kinetics). The per-cluster sums use a numpy-equivalent pairwise
accumulator so results are f64-ULP equivalent to scvelo's `np.sum`; tiny
end-of-summation drift (~1 ULP) propagates through `norm.sf`, so per-cluster
pvals match scvelo within ~1e-15 abs but are not strictly `==`.

The `fit_diff_kinetics` string list is decided by the `p < 1e-2` threshold,
which is robust to sub-ULP drift - that assertion stays exact.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# Each fixture: (scvelo dataset loader, n_obs slice). `clusters` is required
# for differential_kinetic_test's groupby; the preprocessed h5ad fixtures
# drop obs columns during filtering, so we re-preprocess from the raw loader.
_FIXTURES = [
    ("pancreas", 200),
    ("dentategyrus", 200),
]


def _load_and_prep(loader_name: str, n_obs: int):
    """Load raw scvelo dataset, slice, preprocess, fit dynamical model."""
    import scvelo as scv

    adata = getattr(scv.datasets, loader_name)()
    adata = adata[:n_obs].copy()
    scv.pp.filter_and_normalize(adata, min_shared_counts=10)
    n = min(30, n_obs - 1)
    scv.pp.moments(adata, n_pcs=n, n_neighbors=n)
    # Cast to f64 so stock scvelo and our path agree at NM trajectory level
    # (recover_dynamics is bit-exact under f64 layers per Phase 3.10 results).
    for k in ("Mu", "Ms"):
        if k in adata.layers:
            adata.layers[k] = np.asarray(adata.layers[k], dtype=np.float64)
    scv.tl.recover_dynamics(adata, n_jobs=1, show_progress_bar=False)
    scv.tl.velocity(adata, mode="dynamical")
    if "clusters" not in adata.obs.columns and "celltype" in adata.obs.columns:
        adata.obs["clusters"] = adata.obs["celltype"]
    return adata


@pytest.mark.parametrize("loader,n_obs", _FIXTURES, ids=lambda v: str(v))
def test_diff_kinetic_bit_exact(loader, n_obs):
    import scvelo as scv
    import scvelo_rs

    base = _load_and_prep(loader, n_obs)
    fixture = f"{loader}_{n_obs}obs"
    if "clusters" not in base.obs.columns:
        pytest.skip(f"{fixture}: no 'clusters' obs column for groupby")

    # Baseline (scvelo).
    a = base.copy()
    t0 = time.time()
    scv.tl.differential_kinetic_test(a, groupby="clusters")
    scv_time = time.time() - t0

    # Drop-in (scvelo_rs).
    b = base.copy()
    t0 = time.time()
    scvelo_rs.tl.differential_kinetic_test(b, groupby="clusters")
    rs_time = time.time() - t0

    # 1. var['fit_diff_kinetics']: string-equality.
    sv_diff = a.var["fit_diff_kinetics"].astype(str).to_numpy()
    rs_diff = b.var["fit_diff_kinetics"].astype(str).to_numpy()
    assert np.array_equal(sv_diff, rs_diff), (
        f"{fixture}: fit_diff_kinetics differs (first mismatch at index "
        f"{int(np.argmax(sv_diff != rs_diff))})"
    )

    # 2. var['fit_pval_kinetics']: float compare within f64 ULP (NaN-aware).
    sv_p = a.var["fit_pval_kinetics"].to_numpy(dtype=np.float64)
    rs_p = b.var["fit_pval_kinetics"].to_numpy(dtype=np.float64)
    sv_nan, rs_nan = np.isnan(sv_p), np.isnan(rs_p)
    assert np.array_equal(sv_nan, rs_nan), f"{fixture}: NaN pattern differs in fit_pval_kinetics"
    valid = ~sv_nan
    f64_atol = 1e-12  # ~3 orders above f64 eps (2.2e-16); generous for accumulated rounding
    max_abs = float(np.max(np.abs(sv_p[valid] - rs_p[valid]))) if valid.any() else 0.0
    assert max_abs <= f64_atol, (
        f"{fixture}: fit_pval_kinetics max abs diff {max_abs:.3e} > {f64_atol:.0e}"
    )

    # 3. varm['fit_pvals_kinetics']: per-cluster matrix. scvelo stores as a
    # 2D recarray (n_vars, 1) with one float32 field per cluster. Stack the
    # fields to get a plain (n_vars, n_clusters) float64 matrix for compare.
    def _recarray_to_2d_f64(rec):
        arr = np.asarray(rec)
        if arr.dtype.names is not None:
            stacked = np.stack([arr[n] for n in arr.dtype.names], axis=-1).astype(np.float64)
        else:
            stacked = arr.astype(np.float64, copy=False)
        # Squeeze out the singleton "varm column" axis added by `np.rec...T`.
        return stacked.reshape(stacked.shape[0], -1)

    sv_m = _recarray_to_2d_f64(a.varm["fit_pvals_kinetics"])
    rs_m = _recarray_to_2d_f64(b.varm["fit_pvals_kinetics"])
    assert sv_m.shape == rs_m.shape, f"{fixture}: varm shape {rs_m.shape} != scvelo {sv_m.shape}"
    sv_nan_m, rs_nan_m = np.isnan(sv_m), np.isnan(rs_m)
    assert np.array_equal(sv_nan_m, rs_nan_m), (
        f"{fixture}: NaN pattern differs in varm['fit_pvals_kinetics']"
    )
    valid_m = ~sv_nan_m
    max_abs = float(np.max(np.abs(sv_m[valid_m] - rs_m[valid_m])))
    # scvelo writes as float32 recarray, so the writeback to varm rounds to f32.
    # That's lossy vs the f64 we compute internally - accept f32 ULP noise here.
    f32_ulp = float(np.finfo(np.float32).eps)
    assert max_abs <= f32_ulp, (
        f"{fixture}: fit_pvals_kinetics max abs diff {max_abs:.3e} > f32 eps {f32_ulp:.3e}"
    )

    # 4. uns key.
    assert (
        a.uns["recover_dynamics"]["fit_diff_kinetics"]
        == b.uns["recover_dynamics"]["fit_diff_kinetics"]
    )

    speedup = scv_time / max(rs_time, 1e-6)
    print(
        f"\n{fixture}: scvelo {scv_time:.2f}s | scvelo_rs {rs_time:.2f}s "
        f"| {speedup:.2f}x | n_genes={int((~np.isnan(sv_p)).sum()) + int(sv_nan.sum() == sv_p.size) * sv_p.size}"
    )
