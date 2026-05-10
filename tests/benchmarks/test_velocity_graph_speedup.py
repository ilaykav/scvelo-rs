"""Speedup benchmark for `velocity_graph`. Skipped by default; opt in via
`pytest -m benchmark tests/benchmarks/test_velocity_graph_speedup.py -s`.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "fixture",
    [
        "pancreas_100obs_preprocessed",
        "dentategyrus_100obs_preprocessed",
    ],
)
def test_velocity_graph_speedup(fixture):
    import scvelo as scv
    import scvelo_rs

    a_scv = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))
    a_rs = sc.read(str(_DATA_DIR / f"{fixture}.h5ad"))
    for adata in (a_scv, a_rs):
        adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
        adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
        scv.tl.velocity(adata, mode="deterministic")

    t0 = time.time()
    scv.tl.velocity_graph(a_scv, show_progress_bar=False)
    t_scv = time.time() - t0

    t0 = time.time()
    scvelo_rs.velocity_graph(a_rs, show_progress_bar=False)
    t_rs = time.time() - t0

    speedup = t_scv / t_rs if t_rs > 0 else float("inf")
    print(f"\n{fixture}: scvelo {t_scv:.3f}s vs rust {t_rs:.3f}s -> {speedup:.2f}x speedup")
