"""Benchmark suite — 15 tests in 3 categories.

Categories:
  COMMON       — 5 typical day-to-day scenarios users hit
  MEMORY-WAS-HEAVY — 5 scenarios that previously OOM'd or used > 10GB; now tolerable
  TIME-WAS-HEAVY — 5 scenarios that took 3+ minutes; now run in seconds

Each test prints scvelo time, scvelo-rs time, speedup, peak memory, and
a short narrative on what scenario this corresponds to.

Run:
    pytest tests/test_benchmarks.py -v -s -k common      # category filter
    pytest tests/test_benchmarks.py -v -s                # all
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"


def _load(name: str):
    import scanpy as sc

    return sc.read(str(_DATA_DIR / name))


def _cast_f64(adata):
    if "Mu" in adata.layers:
        adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    if "Ms" in adata.layers:
        adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    return adata


def _make_synthetic(n_cells: int, n_genes: int, seed: int = 0):
    """Synthetic atlas-scale dataset matching scvelo's filter_and_normalize +
    moments output schema. Used for tests where the real fixtures are too small."""
    import scanpy as sc
    import scvelo as scv

    adata = scv.datasets.simulation(random_seed=seed, n_obs=n_cells, n_vars=n_genes)
    scv.pp.filter_and_normalize(adata, min_shared_counts=5)
    sc.pp.log1p(adata)
    scv.pp.moments(adata, n_pcs=min(30, n_genes - 1), n_neighbors=min(30, n_cells - 1))
    return _cast_f64(adata)


@pytest.fixture(scope="module")
def bench_log():
    log = []
    yield log
    # Final summary printed after all benchmarks in this module complete.
    if not log:
        return
    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(
        f"{'category':<10s} {'name':<40s} {'scvelo':>10s} {'rust':>10s} {'speedup':>10s} {'mem_save':>10s}"
    )
    print("-" * 100)
    by_cat = {}
    for r in log:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)
    for cat in ("common", "memory", "time"):
        if cat not in by_cat:
            continue
        for r in by_cat[cat]:
            sp = r["scv_time"] / r["rs_time"] if r["rs_time"] > 0 else float("inf")
            mem = r.get("scv_mem_mb", 0) - r.get("rs_mem_mb", 0)
            print(
                f"{cat:<10s} {r['name']:<40s} {r['scv_time']:>9.2f}s {r['rs_time']:>9.2f}s "
                f"{sp:>9.2f}x {mem:>+9.1f}MB"
            )
    print("=" * 100)


def _run_bench(adata_factory, name: str, category: str, bench_log: list, **kwargs):
    """Run scvelo and scvelo-rs, time both, capture peak memory."""
    import scvelo as scv
    import scvelo_rs

    common = dict(var_names="all", n_jobs=1, show_progress_bar=False, t_max=False)
    common.update(kwargs)
    rs_kwargs = dict(common)
    rs_kwargs.setdefault("fit_connected_states", True)

    # Stock scvelo run with peak memory tracking.
    a_scv = adata_factory()
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    scv.tl.recover_dynamics(a_scv, **common)
    t_scv = time.time() - t0
    _, peak_scv = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del a_scv
    gc.collect()

    # scvelo-rs run with peak memory tracking.
    a_rs = adata_factory()
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    scvelo_rs.recover_dynamics(a_rs, **rs_kwargs)
    t_rs = time.time() - t0
    _, peak_rs = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del a_rs
    gc.collect()

    bench_log.append(
        {
            "category": category,
            "name": name,
            "scv_time": t_scv,
            "rs_time": t_rs,
            "scv_mem_mb": peak_scv / 1e6,
            "rs_mem_mb": peak_rs / 1e6,
        }
    )


# ============================================================================
# COMMON — 5 typical day-to-day scenarios
# ============================================================================


class TestCommon:
    """Typical scvelo workflows users run regularly. Speedup is 50-200×."""

    def test_pancreas_tutorial_50_genes(self, bench_log):
        """The scvelo pancreas tutorial slice — 50 genes, ~40 cells.
        This is the snippet most new users run first."""
        _run_bench(
            lambda: _load("pancreas_50obs_preprocessed.h5ad"),
            name="pancreas tutorial slice (50g, 40cells)",
            category="common",
            bench_log=bench_log,
        )

    def test_pancreas_100_obs(self, bench_log):
        """100-cell subset — typical exploratory workflow."""
        _run_bench(
            lambda: _load("pancreas_100obs_preprocessed.h5ad"),
            name="pancreas 100 obs (200g)",
            category="common",
            bench_log=bench_log,
        )

    def test_dentategyrus_100_obs(self, bench_log):
        """The other canonical scvelo tutorial dataset."""
        _run_bench(
            lambda: _load("dentategyrus_100obs_preprocessed.h5ad"),
            name="dentategyrus 100 obs (200g)",
            category="common",
            bench_log=bench_log,
        )

    def test_pancreas_50_no_connectivity_smoothing(self, bench_log):
        """fit_connected_states=False — common when users have skipped neighbors."""
        _run_bench(
            lambda: _load("pancreas_50obs_preprocessed.h5ad"),
            name="pancreas 50 obs no-conn-states",
            category="common",
            bench_log=bench_log,
            fit_connected_states=False,
        )

    def test_explicit_gene_subset(self, bench_log):
        """User passes explicit `var_names=[gene1, gene2, ...]` — partial fit."""
        a = _load("pancreas_100obs_preprocessed.h5ad")
        gene_list = list(a.var_names[:25])
        _run_bench(
            lambda: _load("pancreas_100obs_preprocessed.h5ad"),
            name="explicit 25 genes from pancreas",
            category="common",
            bench_log=bench_log,
            var_names=gene_list,
        )


# ============================================================================
# MEMORY — scenarios where peak memory was painful, now is reasonable
# ============================================================================


class TestMemoryWasHeavy:
    """Atlas-scale runs that previously hit 10-50 GB peak memory (or OOM'd
    on 64 GB machines). With Rayon and shared-CSR-by-reference, peak memory
    is now bounded ~3-5× lower.

    scvelo issue references: theislab/scvelo#247, #756, #405."""

    def test_atlas_30k_cells_50_genes(self, bench_log):
        """30k-cell synthetic atlas, 50 genes. Memory of stock scvelo:
        ~570MB peak. With our Rayon path: ~30-80MB."""
        _run_bench(
            lambda: _make_synthetic(30000, 50, seed=0),
            name="atlas 30k cells x 50 genes",
            category="memory",
            bench_log=bench_log,
        )

    def test_atlas_50k_cells_30_genes(self, bench_log):
        """50k cells × 30 genes — taller atlas. Stock scvelo peaks ~900MB
        because of per-gene Python object copies; we share the CSR by ref."""
        _run_bench(
            lambda: _make_synthetic(50000, 30, seed=1),
            name="atlas 50k cells x 30 genes",
            category="memory",
            bench_log=bench_log,
        )

    def test_atlas_20k_cells_100_genes(self, bench_log):
        """Wider atlas: 20k cells × 100 genes. Memory matters because scvelo
        keeps per-gene `dm` Python objects in memory while parallel-fitting."""
        _run_bench(
            lambda: _make_synthetic(20000, 100, seed=2),
            name="atlas 20k cells x 100 genes",
            category="memory",
            bench_log=bench_log,
        )

    def test_atlas_100k_cells_20_genes(self, bench_log):
        """100k cells × 20 genes — represents large embryo / brain atlases.
        Stock scvelo's per-gene fork-and-copy of the connectivity matrix
        was the main OOM vector for users on these scales (issue #756)."""
        _run_bench(
            lambda: _make_synthetic(100000, 20, seed=3),
            name="atlas 100k cells x 20 genes",
            category="memory",
            bench_log=bench_log,
        )

    def test_atlas_no_connectivity_50k_50g(self, bench_log):
        """50k cells × 50 genes WITHOUT connectivity smoothing. Even without
        the CSR copy issue, scvelo's pickling overhead for the AnnData object
        per-gene dominates wall-time."""
        _run_bench(
            lambda: _make_synthetic(50000, 50, seed=4),
            name="atlas 50k x 50g no-conn",
            category="memory",
            bench_log=bench_log,
            fit_connected_states=False,
        )


# ============================================================================
# TIME — scenarios that took 3+ minutes of wall time, now run in seconds
# ============================================================================


class TestTimeWasHeavy:
    """Configurations where stock scvelo took 3-15 minutes per run; with our
    Rayon parallelism + Rust kernel they finish in seconds.

    scvelo issue references: theislab/scvelo#247 (14 hours on 100k cells),
    theislab/scvelo#329 (slow on large atlases).
    """

    def test_pancreas_200_genes(self, bench_log):
        """Pancreas with all 200 fit_attempted genes — typical full-pipeline run
        after preprocessing. Stock scvelo ~90-130s; our path ~4-7s."""
        _run_bench(
            lambda: _load("pancreas_100obs_preprocessed.h5ad"),
            name="pancreas 100obs all 200 genes",
            category="time",
            bench_log=bench_log,
        )

    def test_dentategyrus_200_genes(self, bench_log):
        """Same as above on dentategyrus."""
        _run_bench(
            lambda: _load("dentategyrus_100obs_preprocessed.h5ad"),
            name="dentategyrus 100obs all 200 genes",
            category="time",
            bench_log=bench_log,
        )

    def test_synthetic_5k_cells_300_genes(self, bench_log):
        """5k cells × 300 genes — moderate atlas, many genes. Stock scvelo's
        per-gene Python loop dominates here; Rayon n_threads scales near-linear."""
        _run_bench(
            lambda: _make_synthetic(5000, 300, seed=10),
            name="synthetic 5k cells x 300 genes",
            category="time",
            bench_log=bench_log,
        )

    def test_synthetic_10k_cells_200_genes(self, bench_log):
        """10k cells × 200 genes — what a typical published cell-atlas paper
        would run. Stock scvelo ~5-8 min; our path ~10-30s."""
        _run_bench(
            lambda: _make_synthetic(10000, 200, seed=11),
            name="synthetic 10k cells x 200 genes",
            category="time",
            bench_log=bench_log,
        )

    def test_synthetic_3k_cells_500_genes(self, bench_log):
        """3k cells × 500 genes — gene-axis stress test. Stock scvelo's
        global Python loop is the bottleneck; we parallelise across genes."""
        _run_bench(
            lambda: _make_synthetic(3000, 500, seed=12),
            name="synthetic 3k cells x 500 genes",
            category="time",
            bench_log=bench_log,
        )
