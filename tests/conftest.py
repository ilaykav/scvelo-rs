"""Pytest conftest — build test fixtures on demand from upstream scvelo datasets.

`tests/_data/*.h5ad` is gitignored (regenerable). On first run (typically CI),
this conftest builds each fixture by loading the corresponding scvelo
tutorial dataset, running the standard preprocessing, and slicing to the
target obs count. Subsequent runs see the file on disk and skip the build.
"""

from __future__ import annotations

import warnings
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "_data"

# (fixture name) -> (scvelo dataset loader name, obs slice size)
_FIXTURES: dict[str, tuple[str, int]] = {
    "pancreas_50obs_preprocessed": ("pancreas", 50),
    "pancreas_100obs_preprocessed": ("pancreas", 100),
    "dentategyrus_50obs_preprocessed": ("dentategyrus", 50),
    "dentategyrus_100obs_preprocessed": ("dentategyrus", 100),
}


def _build_fixture(name: str, dataset: str, n_obs: int) -> None:
    import scvelo as scv

    adata = getattr(scv.datasets, dataset)()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=200)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata = adata[:n_obs].copy()
    adata.write(_DATA_DIR / f"{name}.h5ad")


def pytest_configure(config):
    _DATA_DIR.mkdir(exist_ok=True)
    missing = {n: m for n, m in _FIXTURES.items() if not (_DATA_DIR / f"{n}.h5ad").exists()}
    if not missing:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (dataset, n_obs) in missing.items():
            _build_fixture(name, dataset, n_obs)
