"""Benchmark suite — 13 measurements across speed, memory, and vendor categories.

Run as a script:
    python notebooks/02_benchmarks.py [--quick | --long | --vendor-only]

  --quick         run only the 6 quick benchmarks (~15 min on GH 2-core).
  --long          run all benchmarks including 3 vendor workflows (~6-12 h).
  --vendor-only   run only the 3 vendor (real-world) workflows.
  (default)       same as --long.

Outputs `notebooks/_artifacts/benchmark_table.md` and `_artifacts/benchmark_results.json`.

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
    mem_oom_crash_100k               — 100k × 30  LONG

  Vendor (real-world workflows, all LONG):
    vendor_pancreas_tutorial         — scvelo pancreas tutorial (3.7k cells)
    vendor_cellrank2_hematopoiesis   — CellRank fate mapping (24k cells)
    vendor_mouse_gastrulation_atlas  — atlas scale (116k cells); scvelo SKIPPED
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
import sys
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

# Make `vendor/...` importable when running this script from the notebooks/ dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Runner hardware capture — stamped into JSON output and markdown header
# so future regenerations are self-labeling and CI artifacts are comparable.
# ---------------------------------------------------------------------------


def _hardware_info() -> dict:
    """Capture the runner's CPU / RAM / platform so result tables are honest."""
    try:
        import cpuinfo  # py-cpuinfo, optional

        cpu_model = cpuinfo.get_cpu_info().get("brand_raw") or "unknown"
    except Exception:
        cpu_model = platform.processor() or "unknown"

    return {
        "cpu_model": cpu_model,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
    }


def _hardware_line(hw: dict) -> str:
    return (
        f"{hw['cpu_model']}, "
        f"{hw['cpu_count_physical']}C/{hw['cpu_count_logical']}T, "
        f"{hw['total_ram_gb']} GB RAM, "
        f"{hw['platform']}, "
        f"Python {hw['python']} ({hw['runner']})"
    )


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
    category: str  # "speed", "memory", or "vendor"
    long: bool
    n_cells: int
    n_genes: int
    # Inline ops list (used for speed/memory). Vendor benches leave this empty
    # and set `workflow` instead.
    ops: list[str] = field(default_factory=list)
    # When set, dispatch to `vendor.workflows.<workflow>.run` instead of `_run_ops`.
    workflow: str | None = None
    description: str = ""
    skip_scvelo_above: int | None = None  # if n_cells exceeds this, expect scvelo to OOM
    # When True, the scvelo run is skipped entirely (documented OOM/timeout)
    # and reported as `{"status": "skipped", "reason": skip_scvelo_reason}`.
    skip_scvelo: bool = False
    skip_scvelo_reason: str = ""

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
    # === VENDOR (3) — real-world workflows; see vendor/workflows/ ===
    Bench(
        name="vendor_pancreas_tutorial",
        category="vendor",
        long=True,
        n_cells=3_696,
        n_genes=2_000,
        workflow="pancreas_tutorial",
        description=(
            "scvelo Pancreas.ipynb tutorial (Bastidas-Ponce 2019): "
            "filter_and_normalize -> moments -> recover_dynamics -> velocity -> "
            "velocity_graph -> latent_time"
        ),
    ),
    Bench(
        name="vendor_cellrank2_hematopoiesis",
        category="vendor",
        long=True,
        n_cells=24_000,
        n_genes=2_000,
        workflow="cellrank2_hematopoiesis",
        description=(
            "CellRank 2 hematopoiesis (Setty 2019 bone marrow): scvelo dynamical "
            "pipeline + CellRank VelocityKernel + GPCCA fate probabilities"
        ),
    ),
    Bench(
        name="vendor_mouse_gastrulation_atlas",
        category="vendor",
        long=True,
        n_cells=116_000,
        n_genes=2_000,
        workflow="mouse_gastrulation_atlas",
        skip_scvelo=True,
        skip_scvelo_reason=(
            "documented OOM/timeout on stock scvelo at this scale "
            "(scvelo issues #247, #756, #405; ~14 h on 100k cells per CLAUDE.md)"
        ),
        description=(
            "Pijuan-Sala 2019 mouse gastrulation atlas (~116k cells): scvelo "
            "dynamical pipeline at atlas scale; stock scvelo run skipped"
        ),
    ),
]

assert len(BENCHMARKS) == 13
assert sum(1 for b in BENCHMARKS if b.long) == 7
assert sum(1 for b in BENCHMARKS if b.category == "speed") == 5
assert sum(1 for b in BENCHMARKS if b.category == "memory") == 5
assert sum(1 for b in BENCHMARKS if b.category == "vendor") == 3


# ---------------------------------------------------------------------------
# Run one benchmark against both backends.
# ---------------------------------------------------------------------------


_VENDOR_DATA_CACHE: dict[str, object] = {}


def _vendor_load_data(workflow_name: str):
    """Load (and cache for a single run) the AnnData fixture for a vendor workflow."""
    if workflow_name in _VENDOR_DATA_CACHE:
        return _VENDOR_DATA_CACHE[workflow_name]
    mod = importlib.import_module(f"vendor.workflows.{workflow_name}")
    adata = mod.load_data()
    _VENDOR_DATA_CACHE[workflow_name] = adata
    return adata


def _run_workflow(workflow_name: str, lib, adata):
    """Dispatch a vendor workflow's `run(lib, adata)`."""
    mod = importlib.import_module(f"vendor.workflows.{workflow_name}")
    mod.run(lib, adata)


def run_one(bench: Bench) -> dict:
    import scvelo as scv
    import scvelo_rs

    label_ops = bench.ops if bench.workflow is None else f"workflow={bench.workflow}"
    print(f"\n=== {bench.label()} — {bench.n_cells:,} × {bench.n_genes}, {label_ops} ===")
    print(f"    {bench.description}")

    if bench.workflow is not None:
        # Vendor benches load real data via the workflow module.
        base = _vendor_load_data(bench.workflow)
        timed_fn = _run_workflow
        timed_args = (bench.workflow,)
    else:
        base = make_atlas(bench.n_cells, bench.n_genes)
        timed_fn = _run_ops
        timed_args = ()

    out = {
        "name": bench.name,
        "category": bench.category,
        "long": bench.long,
        "n_cells": bench.n_cells,
        "n_genes": bench.n_genes,
        "ops": bench.ops,
        "workflow": bench.workflow,
    }

    for backend, lib in (("scvelo", scv), ("scvelo_rs", scvelo_rs)):
        if backend == "scvelo" and bench.skip_scvelo:
            out[backend] = {"status": "skipped", "reason": bench.skip_scvelo_reason}
            print(f"  {backend:<11s}  SKIPPED — {bench.skip_scvelo_reason}")
            continue

        a = base.copy()
        if bench.workflow is not None:
            wall, peak_mb, ok, _ = measure(timed_fn, *timed_args, lib, a)
        else:
            wall, peak_mb, ok, _ = measure(timed_fn, lib, a, bench.ops)

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
        elif s.get("status") == "skipped" and r.get("status") == "ok":
            print(f"  -> scvelo skipped, scvelo-rs survived ({r['wall_s']}s, {r['peak_mb']}MB)")

    return out


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def write_markdown(results: list[dict], out_path: Path, hardware: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# scvelo-rs benchmark suite\n\n")
        f.write(f"**Run on:** {_hardware_line(hardware)}\n\n")
        f.write(
            "> Measured single-threaded with `n_jobs=1`. GitHub-hosted CI runners "
            "(2 cores) show smaller speedups than developer workstations — these "
            "numbers illustrate the gap rather than serving as a hardware-neutral "
            "benchmark.\n\n"
        )
        f.write("13 measurements: 5 speed + 5 memory + 3 vendor (real workflows).\n\n")

        def fmt(d, key, unit):
            if not isinstance(d, dict):
                return "—"
            if d.get("status") == "OOM":
                return "**OOM**"
            if d.get("status") == "skipped":
                return "**SKIPPED**"
            return f"{d.get(key, '—')} {unit}"

        sections = (
            ("speed", "Speed (wall time)"),
            ("memory", "Memory (peak heap)"),
            ("vendor", "Vendor workflows (real-world end-to-end)"),
        )
        for category, header in sections:
            f.write(f"## {header}\n\n")
            if category == "vendor":
                f.write("| workflow | cells | genes | scvelo | scvelo-rs | speedup |\n")
                f.write("|---|---:|---:|---|---|---:|\n")
            else:
                f.write("| benchmark | cells | genes | ops | scvelo | scvelo-rs | ratio |\n")
                f.write("|---|---:|---:|---|---|---|---:|\n")

            for r in results:
                if r["category"] != category:
                    continue
                s = r.get("scvelo", {})
                rs = r.get("scvelo_rs", {})

                if category == "speed":
                    scv_cell = fmt(s, "wall_s", "s")
                    rs_cell = fmt(rs, "wall_s", "s")
                    ratio = r.get("speedup_x", "—")
                    ratio_str = f"{ratio}×" if isinstance(ratio, (int, float)) else "—"
                    long_marker = " (LONG)" if r["long"] else ""
                    f.write(
                        f"| {r['name']}{long_marker} | {r['n_cells']:,} | {r['n_genes']} | "
                        f"{','.join(r['ops'])} | {scv_cell} | {rs_cell} | {ratio_str} |\n"
                    )
                elif category == "memory":
                    scv_cell = fmt(s, "peak_mb", "MB")
                    rs_cell = fmt(rs, "peak_mb", "MB")
                    saved = r.get("mem_saved_mb", "—")
                    ratio_str = f"{saved:+} MB" if isinstance(saved, (int, float)) else "—"
                    long_marker = " (LONG)" if r["long"] else ""
                    f.write(
                        f"| {r['name']}{long_marker} | {r['n_cells']:,} | {r['n_genes']} | "
                        f"{','.join(r['ops'])} | {scv_cell} | {rs_cell} | {ratio_str} |\n"
                    )
                else:
                    # vendor — column layout differs (no ops, speedup column instead)
                    scv_cell = fmt(s, "wall_s", "s")
                    rs_cell = fmt(rs, "wall_s", "s")
                    ratio = r.get("speedup_x", "—")
                    if isinstance(ratio, (int, float)):
                        ratio_str = f"{ratio}×"
                    elif s.get("status") == "skipped" and rs.get("status") == "ok":
                        ratio_str = "**∞ (scvelo skipped)**"
                    elif s.get("status") == "OOM" and rs.get("status") == "ok":
                        ratio_str = "**∞ (scvelo OOM)**"
                    else:
                        ratio_str = "—"
                    f.write(
                        f"| {r['name']} (LONG) | {r['n_cells']:,} | {r['n_genes']} | "
                        f"{scv_cell} | {rs_cell} | {ratio_str} |\n"
                    )
            f.write("\n")

        f.write("Generated by `notebooks/02_benchmarks.py`.\n")


def main():
    ap = argparse.ArgumentParser()
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--quick", action="store_true", help="run only the 6 quick benchmarks")
    tier.add_argument(
        "--long",
        action="store_true",
        help="run all 13 benchmarks (quick + long + vendor). Same as no flag.",
    )
    tier.add_argument(
        "--vendor-only",
        action="store_true",
        help="run only the 3 vendor (real-world) workflows",
    )
    args = ap.parse_args()

    if args.quick:
        # Quick excludes long. Vendor is always long.
        selected = [b for b in BENCHMARKS if not b.long]
    elif args.vendor_only:
        selected = [b for b in BENCHMARKS if b.category == "vendor"]
    else:
        # default and --long both run everything (cumulative).
        selected = list(BENCHMARKS)

    hardware = _hardware_info()
    print(f"Hardware: {_hardware_line(hardware)}")
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
    write_markdown(results, md_path, hardware)
    json_path.write_text(
        json.dumps({"hardware": hardware, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nresults -> {md_path}")
    print(f"raw     -> {json_path}")


if __name__ == "__main__":
    main()
