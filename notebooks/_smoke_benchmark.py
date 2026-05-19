"""Smoke benchmark for the CI perf-regression gate (issue #7).

Runs the full mini-pipeline (recover_dynamics + velocity + velocity_graph) on a
5k x 50 synthetic atlas with scvelo-rs only and asserts wall time <= 60 s. Wall
time on a developer workstation is ~3 s; the 60 s ceiling is a loose bound chosen
to catch ~20x regressions on GitHub's 2-core runners without flakiness.

Run locally:
    python notebooks/_smoke_benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import scanpy as sc
import scvelo as scv
import scvelo_rs

warnings.filterwarnings("ignore")

WALL_BOUND_S = 60.0
N_CELLS = 5_000
N_GENES = 50


def main() -> int:
    adata = scv.datasets.simulation(random_seed=0, n_obs=N_CELLS, n_vars=N_GENES)
    scv.pp.filter_and_normalize(adata, min_shared_counts=5)
    sc.pp.log1p(adata)
    scv.pp.moments(adata, n_pcs=min(30, adata.n_vars - 1), n_neighbors=30)
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)

    t0 = time.time()
    scvelo_rs.tl.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    scvelo_rs.tl.velocity(adata, mode="deterministic")
    scvelo_rs.tl.velocity_graph(adata, show_progress_bar=False)
    wall = time.time() - t0

    print(f"smoke pipeline wall: {wall:.2f}s (bound: {WALL_BOUND_S}s)")
    if wall > WALL_BOUND_S:
        print(f"REGRESSION: smoke wall {wall:.2f}s exceeds bound {WALL_BOUND_S}s")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
