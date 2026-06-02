"""scvelo_rs.pp - preprocessing primitives.

scvelo-rs mirrors `scvelo.pp`'s current surface. The Rust SVD and HNSW
primitives are callable directly via `scvelo_rs._scvelo_rs.pca_kernel` and
`scvelo_rs._scvelo_rs.knn_kernel`.
"""

from typing import Any

import scvelo as _scv


def neighbors(
    adata,
    n_neighbors: int = 30,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    knn: bool = True,
    random_state: int = 0,
    method: str = "umap",
    metric: str = "euclidean",
    metric_kwds=None,
    num_threads: int = -1,
    copy: bool = False,
    **kwargs: Any,
):
    """scvelo.pp.neighbors-compatible.

    TODO(#6): port to Rust.
    """
    return _scv.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        knn=knn,
        random_state=random_state,
        method=method,
        metric=metric,
        metric_kwds=metric_kwds,
        num_threads=num_threads,
        copy=copy,
        **kwargs,
    )


__all__ = ["neighbors"]
