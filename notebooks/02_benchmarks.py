"""Benchmark suite — 10 measurements split into speed and memory categories.

Run as a script:
    python notebooks/02_benchmarks.py [--quick] [--long-only]

  --quick      run only the 6 quick benchmarks (default in CI).
  --long-only  run only the 4 long-running benchmarks.
  (default)    run all 10.

Outputs `notebooks/_artifacts/benchmark_table.md` and `_artifacts/benchmark_results.json`.

The 10 benchmarks:

  Speed (wall time):
    speed_recover_dynamics_5k        — 5k × 50    quick
    speed_velocity_20k               — 20k × 100  quick
    speed_velocity_graph_20k         — 20k × 100  quick
    speed_full_pipeline_50k          — 50k × 100  LONG
    speed_recover_dynamics_100k      — 100k × 30  LONG

  Memory (peak heap):
    mem_recover_dynamics_5k          — 5k × 50    quick
    mem_velocity_graph_20k           — 20k × 100  quick
    mem_steady_state_layers          — 5k × 200   quick
    mem_full_pipeline_50k            — 50k × 100  LONG
    mem_oom_crash_100k               — 100k × 30  LONG (selling-point demo)
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic atlas factory.
# ---------------------------------------------------------------------------

_FIXTURE_CACHE: dict[tuple[int, int, int], object] = {}


def make_atlas(n_cells: int, n_genes: int, seed: int = 0):
    """Synthetic AnnData mimicking `scv.datasets.simulation` post-`pp.moments`.
    Cached per (n_cells, n_genes, seed) within a single run."""
    key = (n_cells, n_genes, seed)
    if key in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[key]

    import scanpy as sc
    import scvelo as scv

    adata = scv.datasets.simulation(random_seed=seed, n_obs=n_cells, n_vars=n_genes)
    scv.pp.filter_and_normalize(adata, min_shared_counts=5)
    sc.pp.log1p(adata)
    scv.pp.moments(adata, n_pcs=min(30, n_genes - 1), n_neighbors=min(30, n_cells - 1))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    _FIXTURE_CACHE[key] = adata
    return adata


# ---------------------------------------------------------------------------
# Per-step measurement.
# ---------------------------------------------------------------------------


def measure(fn: Callable, *args, **kwargs):
    """Run fn; return (wall_seconds, peak_rss_delta_mb, ok, return_or_exception).

    Uses psutil RSS sampled in a background thread so Rust-side allocations
    (which tracemalloc misses) are counted. Subtracts pre-call RSS to report
    peak heap *added* by the call.
    """
    proc = psutil.Process()
    gc.collect()
    rss_baseline = proc.memory_info().rss
    peak = [rss_baseline]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                cur = proc.memory_info().rss
                if cur > peak[0]:
                    peak[0] = cur
            except psutil.Error:
                pass
            stop.wait(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        ok = True
    except MemoryError as e:
        out = e
        ok = False
    wall = time.time() - t0
    stop.set()
    sampler.join(timeout=1.0)

    peak_delta_mb = max(0.0, (peak[0] - rss_baseline) / 1e6)
    return wall, peak_delta_mb, ok, out


# ---------------------------------------------------------------------------
# Operation runners — each takes adata + library and runs one or more steps.
# ---------------------------------------------------------------------------


def _run_ops(lib, adata, ops: list[str]):
    """Run a sequence of pipeline operations in-place."""
    for op in ops:
        if op == "recover_dynamics":
            lib.tl.recover_dynamics(
                adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
            )
        elif op == "velocity":
            lib.tl.velocity(adata, mode="deterministic")
        elif op == "velocity_graph":
            # velocity_graph requires `velocity` first if missing.
            if "velocity" not in adata.layers:
                lib.tl.velocity(adata, mode="deterministic")
            lib.tl.velocity_graph(adata, show_progress_bar=False)
        else:
            raise ValueError(f"unknown op {op!r}")


# ---------------------------------------------------------------------------
# Benchmark spec.
# ---------------------------------------------------------------------------


@dataclass
class Bench:
    name: str
    category: str  # "speed" or "memory"
    long: bool
    n_cells: int
    n_genes: int
    ops: list[str]
    description: str = ""
    skip_scvelo_above: int | None = None  # if n_cells exceeds this, expect scvelo to OOM

    def label(self) -> str:
        suffix = " (LONG)" if self.long else ""
        return f"{self.name}{suffix}"


BENCHMARKS: list[Bench] = [
    # === SPEED (5) ===
    Bench(
        name="speed_recover_dynamics_5k",
        category="speed",
        long=False,
        n_cells=5_000,
        n_genes=50,
        ops=["recover_dynamics"],
        description="recover_dynamics wall time, 5k × 50",
    ),
    Bench(
        name="speed_velocity_20k",
        category="speed",
        long=False,
        n_cells=20_000,
        n_genes=100,
        ops=["recover_dynamics", "velocity"],
        description="velocity wall time after recover_dynamics warm-up",
    ),
    Bench(
        name="speed_velocity_graph_20k",
        category="speed",
        long=False,
        n_cells=20_000,
        n_genes=100,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="velocity_graph wall time, 20k × 100",
    ),
    Bench(
        name="speed_full_pipeline_50k",
        category="speed",
        long=True,
        n_cells=50_000,
        n_genes=100,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="end-to-end pipeline, 50k cells",
    ),
    Bench(
        name="speed_recover_dynamics_100k",
        category="speed",
        long=True,
        n_cells=100_000,
        n_genes=30,
        ops=["recover_dynamics"],
        description="recover_dynamics at atlas scale, 100k cells",
    ),
    # === MEMORY (5) ===
    Bench(
        name="mem_recover_dynamics_5k",
        category="memory",
        long=False,
        n_cells=5_000,
        n_genes=50,
        ops=["recover_dynamics"],
        description="peak heap during recover_dynamics, 5k × 50",
    ),
    Bench(
        name="mem_velocity_graph_20k",
        category="memory",
        long=False,
        n_cells=20_000,
        n_genes=100,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="peak heap during velocity_graph (n_recurse_neighbors expansion)",
    ),
    Bench(
        name="mem_steady_state_layers",
        category="memory",
        long=False,
        n_cells=5_000,
        n_genes=200,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="peak heap with many genes (fit_t/fit_tau full-shape layers)",
    ),
    Bench(
        name="mem_full_pipeline_50k",
        category="memory",
        long=True,
        n_cells=50_000,
        n_genes=100,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="peak heap, end-to-end pipeline, 50k cells",
    ),
    Bench(
        name="mem_oom_crash_100k",
        category="memory",
        long=True,
        n_cells=100_000,
        n_genes=30,
        ops=["recover_dynamics", "velocity", "velocity_graph"],
        description="selling-point demo: scvelo's joblib forks OOM, we hold shared memory",
    ),
]

assert len(BENCHMARKS) == 10
assert sum(1 for b in BENCHMARKS if b.long) == 4
assert sum(1 for b in BENCHMARKS if b.category == "speed") == 5
assert sum(1 for b in BENCHMARKS if b.category == "memory") == 5


# ---------------------------------------------------------------------------
# Run one benchmark against both backends.
# ---------------------------------------------------------------------------


def run_one(bench: Bench) -> dict:
    import scvelo as scv
    import scvelo_rs

    print(f"\n=== {bench.label()} — {bench.n_cells:,} × {bench.n_genes}, ops={bench.ops} ===")
    print(f"    {bench.description}")

    base = make_atlas(bench.n_cells, bench.n_genes)

    out = {
        "name": bench.name,
        "category": bench.category,
        "long": bench.long,
        "n_cells": bench.n_cells,
        "n_genes": bench.n_genes,
        "ops": bench.ops,
    }

    for backend, lib in (("scvelo", scv), ("scvelo_rs", scvelo_rs)):
        a = base.copy()
        wall, peak_mb, ok, _ = measure(_run_ops, lib, a, bench.ops)
        if not ok:
            print(f"  {backend:<11s}  OOM after {wall:.1f}s")
            out[backend] = {"status": "OOM", "wall_s": round(wall, 2)}
            continue
        out[backend] = {
            "status": "ok",
            "wall_s": round(wall, 2),
            "peak_mb": round(peak_mb, 1),
        }
        print(f"  {backend:<11s}  wall={wall:>8.2f}s  peak={peak_mb:>7.1f}MB")
        del a
        gc.collect()

    if isinstance(out.get("scvelo"), dict) and isinstance(out.get("scvelo_rs"), dict):
        s, r = out["scvelo"], out["scvelo_rs"]
        if s.get("status") == "ok" and r.get("status") == "ok":
            out["speedup_x"] = round(s["wall_s"] / max(r["wall_s"], 1e-3), 2)
            out["mem_saved_mb"] = round(s["peak_mb"] - r["peak_mb"], 1)
            print(f"  -> speedup {out['speedup_x']:.2f}x, mem_saved {out['mem_saved_mb']:+.1f} MB")
        elif s.get("status") == "OOM" and r.get("status") == "ok":
            print(f"  -> scvelo OOM, scvelo-rs survived ({r['wall_s']}s, {r['peak_mb']}MB)")

    return out


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def write_markdown(results: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# scvelo-rs benchmark suite\n\n")
        f.write("10 measurements: 5 speed + 5 memory, 4 marked LONG.\n\n")

        for category, header in (("speed", "Speed (wall time)"), ("memory", "Memory (peak heap)")):
            f.write(f"## {header}\n\n")
            f.write("| benchmark | cells | genes | ops | scvelo | scvelo-rs | ratio |\n")
            f.write("|---|---:|---:|---|---|---|---:|\n")
            for r in results:
                if r["category"] != category:
                    continue
                s = r.get("scvelo", {})
                rs = r.get("scvelo_rs", {})

                def fmt(d, key, unit):
                    if not isinstance(d, dict):
                        return "—"
                    if d.get("status") == "OOM":
                        return "**OOM**"
                    return f"{d.get(key, '—')} {unit}"

                if category == "speed":
                    scv_cell = fmt(s, "wall_s", "s")
                    rs_cell = fmt(rs, "wall_s", "s")
                    ratio = r.get("speedup_x", "—")
                    ratio_str = f"{ratio}×" if isinstance(ratio, (int, float)) else "—"
                else:
                    scv_cell = fmt(s, "peak_mb", "MB")
                    rs_cell = fmt(rs, "peak_mb", "MB")
                    saved = r.get("mem_saved_mb", "—")
                    ratio_str = f"{saved:+} MB" if isinstance(saved, (int, float)) else "—"

                long_marker = " (LONG)" if r["long"] else ""
                f.write(
                    f"| {r['name']}{long_marker} | {r['n_cells']:,} | {r['n_genes']} | "
                    f"{','.join(r['ops'])} | {scv_cell} | {rs_cell} | {ratio_str} |\n"
                )
            f.write("\n")

        f.write("Generated by `notebooks/02_benchmarks.py`.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run only quick benchmarks")
    ap.add_argument("--long-only", action="store_true", help="run only LONG benchmarks")
    args = ap.parse_args()

    if args.quick:
        selected = [b for b in BENCHMARKS if not b.long]
    elif args.long_only:
        selected = [b for b in BENCHMARKS if b.long]
    else:
        selected = list(BENCHMARKS)

    print(
        f"Running {len(selected)} benchmarks "
        f"({sum(1 for b in selected if not b.long)} quick, "
        f"{sum(1 for b in selected if b.long)} long)"
    )

    out_dir = Path(__file__).parent / "_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for bench in selected:
        try:
            results.append(run_one(bench))
        except MemoryError:
            print(f"  {bench.name}: even fixture build OOM'd — skipping")
            results.append(
                {
                    "name": bench.name,
                    "category": bench.category,
                    "long": bench.long,
                    "n_cells": bench.n_cells,
                    "n_genes": bench.n_genes,
                    "ops": bench.ops,
                    "scvelo": {"status": "fixture_oom"},
                    "scvelo_rs": {"status": "fixture_oom"},
                }
            )

    md_path = out_dir / "benchmark_table.md"
    json_path = out_dir / "benchmark_results.json"
    write_markdown(results, md_path)
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nresults -> {md_path}")
    print(f"raw     -> {json_path}")


if __name__ == "__main__":
    main()
