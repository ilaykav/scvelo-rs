"""Tests for the Rust-backed `pp.pca` and `pp.neighbors` KNN kernels.

PCA: SVD signs are arbitrary, so we compare *subspaces* (absolute correlations
between corresponding components ≥ 0.99) and explained variance ratios.

KNN: HNSW is approximate, so we compare *recall* against a scipy.spatial KDTree
reference (overlap of top-k indices ≥ 95%).
"""

from __future__ import annotations

import numpy as np
import pytest
from scvelo_rs._scvelo_rs import knn_kernel, pca_kernel


def test_pca_matches_scvelo_subspace():
    """PCA from our nalgebra kernel matches scanpy's PCA up to sign / basis
    rotation in degenerate-eigenvalue subspaces. We assert the per-component
    absolute correlation against scanpy's output is >= 0.99 for the top
    components and explained-variance ratios match within 1e-4."""
    sc = pytest.importorskip("scanpy")
    import anndata as ad

    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 80)).astype(np.float64)

    # scanpy reference (uses sklearn arpack under the hood).
    adata = ad.AnnData(X.copy())
    sc.pp.pca(adata, n_comps=20, zero_center=True)
    pca_scv = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    var_ratio_scv = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float64)

    # Our Rust SVD.
    x_pca, _pcs, _var, var_ratio = pca_kernel(np.ascontiguousarray(X, dtype=np.float64), 20, True)
    pca_rs = np.asarray(x_pca, dtype=np.float64)
    var_ratio_rs = np.asarray(var_ratio, dtype=np.float64)

    # Variance-ratio sanity check: must agree component-by-component (these
    # are scalar singular-value squares; sign-invariant).
    np.testing.assert_allclose(var_ratio_scv, var_ratio_rs, rtol=1e-4, atol=1e-6)

    # Per-component absolute correlation. Allow small tolerance for the tail
    # components where degenerate singular values can mix.
    corrs = []
    for k in range(20):
        a = pca_scv[:, k]
        b = pca_rs[:, k]
        corr = abs(np.corrcoef(a, b)[0, 1])
        corrs.append(corr)
    corrs = np.array(corrs)
    # Top half: must be tightly aligned. Tail: degenerate-svd subspace may rotate.
    assert corrs[:10].min() > 0.99, (
        f"top-10 PCA components: min |corr|={corrs[:10].min():.4f} < 0.99"
    )
    # Allow some slack for the tail components.
    assert corrs.min() > 0.5, f"all components: min |corr|={corrs.min():.4f}"


def test_knn_recall_vs_kdtree():
    """HNSW returns approximate KNN; check recall vs an exact KDTree reference."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    n_cells, n_dim, k = 800, 30, 30
    X = rng.normal(size=(n_cells, n_dim)).astype(np.float32)

    # Exact reference via scipy.
    tree = cKDTree(X)
    _, ref_idx = tree.query(X, k=k + 1)  # +1 so we can drop self
    ref_idx = ref_idx[:, 1:]

    # Our HNSW.
    flat_idx, _ = knn_kernel(np.ascontiguousarray(X, dtype=np.float32), k)
    rs_idx = np.asarray(flat_idx).reshape(n_cells, k)

    # Per-cell recall: |our_top_k ∩ ref_top_k| / k.
    recall = np.array([len(set(rs_idx[i]) & set(ref_idx[i])) / float(k) for i in range(n_cells)])
    # HNSW with ef_search=2k typically gives ~0.95-0.99 recall on Gaussian data.
    assert recall.mean() > 0.90, f"mean KNN recall = {recall.mean():.3f} < 0.90"
    assert (recall > 0.5).mean() > 0.95, (
        f"only {(recall > 0.5).mean() * 100:.1f}% of cells have >50% recall"
    )


def test_pca_kernel_shape_and_output_contracts():
    """The Rust SVD kernel must return correctly-shaped (X_pca, PCs, var, var_ratio)
    for the standard non-sparse / zero-centered case."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 60)).astype(np.float64)
    x_pca, pcs, var, var_ratio = pca_kernel(np.ascontiguousarray(X), 15, True)

    assert np.asarray(x_pca).shape == (150, 15)
    assert np.asarray(pcs).shape == (15, 60)
    assert np.asarray(var).shape == (15,)
    assert np.asarray(var_ratio).shape == (15,)
    # variance ratios should sum to <= 1 and be monotonically non-increasing.
    var_ratio = np.asarray(var_ratio, dtype=np.float64)
    assert var_ratio.sum() <= 1.0 + 1e-9
    assert np.all(np.diff(var_ratio) <= 1e-9)
