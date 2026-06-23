# scvelo-rs

[![CI](https://github.com/ilaykav/scvelo-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ilaykav/scvelo-rs/actions/workflows/ci.yml)
[![Wheels](https://github.com/ilaykav/scvelo-rs/actions/workflows/wheels.yml/badge.svg)](https://github.com/ilaykav/scvelo-rs/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/scvelo-rs.svg?color=blue)](https://pypi.org/project/scvelo-rs/)
[![Python versions](https://img.shields.io/pypi/pyversions/scvelo-rs.svg)](https://pypi.org/project/scvelo-rs/)
[![License](https://img.shields.io/pypi/l/scvelo-rs.svg)](https://github.com/ilaykav/scvelo-rs/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/scvelo-rs.svg)](https://pypi.org/project/scvelo-rs/)

A Rust + PyO3 port of [scVelo](https://github.com/theislab/scvelo).

Rust implementation of heavy-weight bottlenecks like `recover_dynamics` & `velocity_graph` - bit-exact to the original scVelo.

## Highlights

- **End-to-end on real workflows.** The canonical scvelo Pancreas tutorial
  (3,696 cells, full pipeline incl. PCA, KNN, latent time) runs **11.5×
  faster** - 15 min on stock scvelo becomes 1.3 min on scvelo-rs. The Rust
  `recover_dynamics` kernel is doing the heavy lifting (\~160× isolated);
  the remaining workflow time is preprocessing and downstream analysis
  that pass through unchanged. See [Benchmarks](#benchmarks) and
  [`vendor/`](vendor/) for the full end-to-end numbers.
- **Bit-exact** equivalence to scVelo on 99.9% of genes - the residual
  drift is at f64 ULP scale (per-gene Pearson r = 1.0000 across
  `fit_alpha`, `fit_beta`, `fit_gamma`, `fit_t_`).
- **Drop-in**: import `scvelo_rs.patch` and every downstream call to
  `scv.tl.{recover_dynamics, velocity, velocity_graph}` routes to Rust.
  Or `import scvelo_rs as scv` for the full API.
- **Cross-platform wheels** for Linux x86_64/aarch64, macOS arm64,
  Windows x86_64. Single `abi3-py310` wheel covers Python 3.10–3.13.
- **CPU-only.** Runs anywhere Python runs - laptop, HPC, Docker, ARM.
  No CUDA. No Numba. No JIT warmup.

```python
# Option 1 - drop-in import.
import scvelo_rs as scv

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata); scv.pp.moments(adata)
scv.tl.recover_dynamics(adata)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis="umap")
```

```python
# Option 2 - monkey-patch. Keep your existing `import scvelo as scv`;
# only the three patched hot paths get the Rust kernel.
import scvelo as scv
import scvelo_rs.patch  # noqa: F401   # patches scv.tl.{recover_dynamics, velocity, velocity_graph}

scv.tl.recover_dynamics(adata)         # bit-exact, no other code change needed
```

## Installation

```bash
pip install scvelo-rs
```

scVelo and scanpy are runtime dependencies (used for plotting, dataset
I/O, DPT/PAGA pass-through). They are pulled in automatically.

## Quick start

Three usage patterns, in order of how invasive the migration is.

### 1. Monkey-patch (zero code changes)

Add one line at the top of your existing scVelo script:

```python
import scvelo as scv
import scvelo_rs.patch  # noqa: F401   # swaps scv.tl.{recover_dynamics, velocity, velocity_graph}

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)
scv.tl.recover_dynamics(adata)         # now Rust-backed
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
```

Originals are preserved at `scv.tl.<name>_original` for A/B comparison.

### 2. Drop-in import

Replace `import scvelo as scv` with `import scvelo_rs as scv`. The
`scvelo_rs.{tl, pp, pl, datasets}` namespaces expose scVelo's full public
API; the hot loops route through Rust, everything else passes through
scVelo unchanged.

```python
import scvelo_rs as scv

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)
scv.tl.recover_dynamics(adata)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis="umap")
```

### 3. Direct call

```python
import scvelo_rs
scvelo_rs.recover_dynamics(adata)      # same signature as scv.tl.recover_dynamics
```

See [`examples/`](examples/) for runnable end-to-end scripts.


## Benchmarks

Measured on a developer workstation, single-threaded `n_jobs=1`. GitHub-hosted CI
runners (2 cores) will show smaller speedups - these numbers illustrate the gap
rather than serving as a hardware-neutral benchmark. The full suite lives in
[`notebooks/02_benchmarks.py`](notebooks/02_benchmarks.py) and stamps the runner's
CPU/RAM into the regenerated table automatically.

### Real-world end-to-end workflows

This is the bar the project is held to: **published scvelo pipelines, on the
real datasets and downstream tools people actually use, run end-to-end on both
backends with identical input** - not a microbench of one kernel. It's what a
user sees when they swap `import scvelo as scv` for `import scvelo_rs as scv` on
their own analysis. All five are registered in
[`notebooks/02_benchmarks.py`](notebooks/02_benchmarks.py) (`category="vendor"`),
run on the CI cron (every other day), and live in [`vendor/workflows/`](vendor/workflows/).

| workflow | cells | tissue · downstream | speedup | numerical match |
|---|---:|---|---|---|
| [Pancreas tutorial](vendor/workflows/pancreas_tutorial) (Bastidas-Ponce 2019) | 3,696 | mouse pancreas · `latent_time` | **11.5×** (919 s → 80 s) | near-bit-exact |
| [Differential kinetics](vendor/workflows/gastrulation_e75_diffkinetics) (Pijuan-Sala E7.5) | \~21k | mouse embryo · `differential_kinetic_test` + per-cluster refit | **21.6×** full pipeline; **593×** on the test step alone | `differential_kinetic_test` **bit-exact** |
| [CellRank-2 hematopoiesis](vendor/workflows/cellrank2_hematopoiesis) (Setty 2019) | \~24k | human bone marrow · CellRank `VelocityKernel` + GPCCA fate mapping | \~**8.9×** end-to-end | near-bit-exact |
| [PBMC 68k](vendor/workflows/pbmc68k_pipeline) (Zheng 2017) | \~68k | human PBMCs · `latent_time` | \~5–8 h → \~10–20 min (cron) | near-bit-exact |
| [Mouse gastrulation atlas](vendor/workflows/mouse_gastrulation_atlas) (Pijuan-Sala 2019) | \~116k | mouse embryo · atlas-scale dynamical | stock OOM/timeout → \~15–30 min | scvelo-rs-only |

Why these are representative, not cherry-picked: the pancreas tutorial is the
canonical scvelo intro; CellRank fate-mapping is the single most common
downstream consumer of `recover_dynamics`; PBMC 68k (Zheng 2017, \~5,000
citations) is the standard high-volume stress dataset; and the 116k
gastrulation atlas is exactly where stock scvelo is documented to OOM/time out
(scvelo issues #247, #756, #405) - the bench reports those as `SKIPPED` and
produces a scvelo-rs-only number.

**Numerical equivalence.** `differential_kinetic_test` is **bit-exact** vs
scvelo given identical fits (`fit_diff_kinetics` matches 2000/2000 string-equal;
per-cluster p-values agree to f64 ULP, \~1e-16). `recover_dynamics` / `velocity`
are **near-bit-exact**: with input layers cast to `float64`, per-gene parameters
match scvelo to a median of 0 and ≤\~3e-3 relative on a small set of
Nelder-Mead saddle-point outlier genes (≈1% of fitted genes); that residual
propagates into `velocity` / `fit_t` in the full pipeline. `fit_scaling`,
`fit_std_u/s`, and `fit_likelihood` are bit-exact (the latter matches to
3.6e-16 after porting scvelo's `get_likelihood(weighted='upper')`). See the
phase log in `CLAUDE.md` for the per-gene breakdown.

### Latest CI run (auto-updated every other day)

The tables below are regenerated every other day by the
[benchmarks workflow](.github/workflows/benchmarks.yml) on a GitHub-hosted
runner and committed here automatically - reproducible CI measurements, not
hand-entered numbers. They cover the `--long` tier (synthetic micro-benchmarks
isolating each kernel, plus the pancreas and CellRank vendor workflows); the
atlas-scale `extra-long` benches exceed CI's 6 h job cap and are summarised in
the real-world table above. A rolling per-run retrospective (last 100 runs, one
compact JSON line each) is kept in
[`notebooks/_artifacts/benchmark_history.jsonl`](notebooks/_artifacts/benchmark_history.jsonl).

<!-- BENCHMARKS:START - auto-generated by .github/workflows/benchmarks.yml; do not edit by hand -->

**Run on:** AMD EPYC 9V74 80-Core Processor, 2C/4T, 16.8 GB RAM, Linux-6.17.0-1018-azure-x86_64-with-glibc2.39, Python 3.12.13 (github-actions)  
**Generated:** 2026-06-23 08:31 UTC

> Measured single-threaded with `n_jobs=1`. GitHub-hosted CI runners (2 cores) show smaller speedups than developer workstations - these numbers illustrate the gap rather than serving as a hardware-neutral benchmark.

13 measurements: 6 speed + 5 memory + 2 vendor (real workflows).

#### Speed (wall time)

| benchmark | cells | genes | ops | scvelo | scvelo-rs | ratio | bit-exact |
|---|---:|---:|---|---|---|---:|---|
| speed_recover_dynamics_5k | 5,000 | 50 | recover_dynamics | 25.09 s | 4.02 s | 6.24× | - |
| speed_velocity_20k | 20,000 | 100 | recover_dynamics,velocity | 201.08 s | 36.26 s | 5.55× | - |
| speed_velocity_graph_20k | 20,000 | 100 | recover_dynamics,velocity,velocity_graph | 208.46 s | 37.7 s | 5.53× | - |
| speed_full_pipeline_50k (LONG) | 50,000 | 100 | recover_dynamics,velocity,velocity_graph | 609.85 s | 107.58 s | 5.67× | ✓ PASS (14/14) |
| speed_recover_dynamics_100k (LONG) | 100,000 | 30 | recover_dynamics | 341.31 s | 72.36 s | 4.72× | ✓ PASS (13/13) |
| speed_compute_dynamics_5k | 5,000 | 50 | compute_dynamics | 0.06 s | 0.01 s | 6.0× | - |

#### Memory (peak heap)

| benchmark | cells | genes | ops | scvelo | scvelo-rs | ratio | bit-exact |
|---|---:|---:|---|---|---|---:|---|
| mem_recover_dynamics_5k | 5,000 | 50 | recover_dynamics | 79.8 MB | 4.4 MB | +75.4 MB | - |
| mem_velocity_graph_20k | 20,000 | 100 | recover_dynamics,velocity,velocity_graph | 1718.7 MB | 525.8 MB | +1192.9 MB | - |
| mem_steady_state_layers | 5,000 | 200 | recover_dynamics,velocity,velocity_graph | 177.3 MB | 0.0 MB | +177.3 MB | - |
| mem_full_pipeline_50k (LONG) | 50,000 | 100 | recover_dynamics,velocity,velocity_graph | 4685.9 MB | 1351.8 MB | +3334.1 MB | - |
| mem_oom_crash_100k (LONG) | 100,000 | 30 | recover_dynamics,velocity,velocity_graph | 6957.4 MB | 1599.7 MB | +5357.7 MB | - |

#### Vendor workflows (real-world end-to-end)

| workflow | cells | genes | scvelo | scvelo-rs | speedup | bit-exact |
|---|---:|---:|---|---|---:|---|
| vendor_pancreas_tutorial (LONG) | 3,696 | 2000 | 421.26 s | 72.7 s | 5.79× | ✓ PASS (15/15) |
| vendor_cellrank2_hematopoiesis (LONG) | 24,000 | 2000 | 347.26 s | 118.24 s | 2.94× | ✓ PASS (15/15) |

<!-- BENCHMARKS:END -->

## Build from source

Requires Rust 1.75+ and Python 3.10+.

```bash
git clone https://github.com/ilaykav/scvelo-rs
cd scvelo-rs
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
maturin develop --release
pytest tests/unit tests/integration
```

The Rust crates `nalgebra` (SVD for PCA) and `hnsw_rs` (HNSW for KNN)
are pure-Rust - no OpenBLAS, no vcpkg, no system C libraries.
Cross-platform builds work out of the box.

## Documentation

A full Sphinx site (Quick start, Installation, Migration from scVelo,
Architecture, Numerical parity, Benchmarks) is in the works for v0.2.
Until then, this `README`, the [`CHANGELOG`](CHANGELOG.md), and the
runnable scripts under [`examples/`](examples/) and
[`notebooks/`](notebooks/) cover the same ground.

## Contributing

Bug reports, PRs, and benchmark contributions welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) - the short version is:

```bash
git clone https://github.com/ilaykav/scvelo-rs
cd scvelo-rs
pip install -e ".[dev]"
maturin develop --release
pytest tests/unit tests/integration
```

Bit-exact equivalence to scVelo is the contract for every Rust-backed
function. PRs that move per-gene drift above `1e-9` need a documented
reason.

## License

Released under [BSD-3-Clause](LICENSE). The
Rust kernels are independent reimplementations of theislab's published
algorithms - credit for the underlying methods belongs to
La Manno et al. 2018 (RNA velocity, *Nature*,
[doi:10.1038/s41586-018-0414-6](https://doi.org/10.1038/s41586-018-0414-6))
and Bergen et al. 2020 (scVelo, *Nat Biotechnol*,
[doi:10.1038/s41587-020-0591-3](https://doi.org/10.1038/s41587-020-0591-3)).

## Citing this work

`scvelo-rs` is a faithful port: the *method* is Bergen et al. 2020,
the *implementation* is this repository. **Always cite the original
scVelo paper as the primary reference**; cite the version of
`scvelo-rs` you used as a software dependency (`pip show scvelo-rs`
or `scvelo_rs.__version__`).

```bibtex
@article{bergen2020generalizing,
  title   = {Generalizing RNA velocity to transient cell states through dynamical modeling},
  author  = {Bergen, Volker and Lange, Marius and Peidli, Stefan and
             Wolf, F. Alexander and Theis, Fabian J.},
  journal = {Nature Biotechnology},
  year    = {2020},
  doi     = {10.1038/s41587-020-0591-3}
}

@software{scvelo_rs,
  title   = {scvelo-rs: a Rust acceleration of scVelo's dynamical model},
  author  = {Kavitzky, Ilay},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ilaykav/scvelo-rs},
  note    = {Rust + PyO3 port of Bergen et al. 2020 (doi:10.1038/s41587-020-0591-3)}
}
```

Authored and maintained by Ilay Kavitzky. Contribution guidelines are
in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Reporting bugs and feature requests

Open an issue at
[github.com/ilaykav/scvelo-rs/issues](https://github.com/ilaykav/scvelo-rs/issues).

**Bug reports** - include:

- `scvelo-rs` version (`pip show scvelo-rs`)
- OS and Python version
- A minimum reproducer (a small `.h5ad` slice + the calls that fail
  is usually enough)
- What you expected vs what you got

**Parity issues** (a fitted parameter or velocity vector differs from
the original scvelo): include both runs' values for the affected
gene/cell, the relative drift, and which fixture you ran on.

**Feature requests** - describe the workflow you can't do today, not
just the API you'd like. Atlas-scale parity reports are especially
welcome.

For anything else, direct mail: ilay.kavitzky@gmail.com.
