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
  faster** — 15 min on stock scvelo becomes 1.3 min on scvelo-rs. The Rust
  `recover_dynamics` kernel is doing the heavy lifting (~160× isolated);
  the remaining workflow time is preprocessing and downstream analysis
  that pass through unchanged. See [Benchmarks](#benchmarks) and
  [`vendor/`](vendor/) for the full end-to-end numbers.
- **Bit-exact** equivalence to scVelo on 99.9% of genes — the residual
  drift is at f64 ULP scale (per-gene Pearson r = 1.0000 across
  `fit_alpha`, `fit_beta`, `fit_gamma`, `fit_t_`).
- **Drop-in**: import `scvelo_rs.patch` and every downstream call to
  `scv.tl.{recover_dynamics, velocity, velocity_graph}` routes to Rust.
  Or `import scvelo_rs as scv` for the full API.
- **Cross-platform wheels** for Linux x86_64/aarch64, macOS arm64,
  Windows x86_64. Single `abi3-py310` wheel covers Python 3.10–3.13.
- **CPU-only.** Runs anywhere Python runs — laptop, HPC, Docker, ARM.
  No CUDA. No Numba. No JIT warmup.

```python
# Option 1 — drop-in import.
import scvelo_rs as scv

adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata); scv.pp.moments(adata)
scv.tl.recover_dynamics(adata)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(adata, basis="umap")
```

```python
# Option 2 — monkey-patch. Keep your existing `import scvelo as scv`;
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
runners (2 cores) will show smaller speedups — these numbers illustrate the gap
rather than serving as a hardware-neutral benchmark. The full suite lives in
[`notebooks/02_benchmarks.py`](notebooks/02_benchmarks.py) and stamps the runner's
CPU/RAM into the regenerated table automatically.

### Real-world end-to-end workflows

The numbers below come from running the official scvelo Pancreas tutorial
end-to-end — preprocessing, `recover_dynamics`, `velocity`, `velocity_graph`,
and `latent_time` — on both backends with identical input. This is what a user
sees when they swap `import scvelo as scv` for `import scvelo_rs as scv` on
their actual workflow, not a microbench of one kernel.

| workflow | cells | genes (HVGs) | scvelo | scvelo-rs | end-to-end speedup |
|---|---:|---:|---:|---:|---:|
| Pancreas tutorial (Bastidas-Ponce 2019) | 3,696 | 2,000 | **919 s** (~15 min) | **80 s** (~1.3 min) | **11.5×** |

CellRank-2 hematopoiesis (24k cells) and Pijuan-Sala 2019 mouse gastrulation
atlas (116k cells) workflows are scaffolded in [`vendor/workflows/`](vendor/)
and run on the weekly CI cron. At atlas scale stock scvelo is documented to
OOM/time out (scvelo issues #247, #756, #405); the bench reports those as
`SKIPPED` and produces a scvelo-rs-only number.

### Microbenchmarks (synthetic atlases)

These isolate individual kernels and skip the non-accelerated parts of a
real workflow. Useful for understanding *where* the time goes; less
representative of end-to-end user experience than the table above.

### Wall time

| benchmark | cells | genes | scvelo | scvelo-rs | speedup |
|---|---:|---:|---:|---:|---:|
| recover_dynamics (5k cells, 50 genes)                     |   5,000 |  50 |   43.24 s |  1.03 s | **42.0×** |
| recover_dynamics + velocity (20k, 100)                    |  20,000 | 100 |  380.02 s | 10.09 s | **37.7×** |
| recover_dynamics + velocity + velocity_graph (20k, 100)   |  20,000 | 100 |  389.54 s | 10.79 s | **36.1×** |
| full pipeline (50k, 100)                                  |  50,000 | 100 | 1202.26 s | 34.35 s | **35.0×** |
| recover_dynamics (100k, 30)                               | 100,000 |  30 |  671.65 s | 22.85 s | **29.4×** |

### Peak memory

| benchmark | cells | genes | scvelo | scvelo-rs | saved |
|---|---:|---:|---:|---:|---:|
| recover_dynamics (5k, 50)                                 |   5,000 |  50 |  108.4 MB |   35.0 MB |    73 MB |
| velocity_graph (20k, 100)                                 |  20,000 | 100 | 1727.5 MB |  626.5 MB | 1,101 MB |
| steady-state layers (5k, 200)                             |   5,000 | 200 |  252.1 MB |   66.2 MB |   186 MB |
| full pipeline (50k, 100)                                  |  50,000 | 100 | 4831.6 MB | 1879.3 MB | 2,952 MB |
| recover_dynamics + velocity_graph (100k, 30)              | 100,000 |  30 | 7074.4 MB | 2794.9 MB | 4,280 MB |

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
are pure-Rust — no OpenBLAS, no vcpkg, no system C libraries.
Cross-platform builds work out of the box.

## Documentation

A full Sphinx site (Quick start, Installation, Migration from scVelo,
Architecture, Numerical parity, Benchmarks) is in the works for v0.2.
Until then, this `README`, the [`CHANGELOG`](CHANGELOG.md), and the
runnable scripts under [`examples/`](examples/) and
[`notebooks/`](notebooks/) cover the same ground.

## Contributing

Bug reports, PRs, and benchmark contributions welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) — the short version is:

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
algorithms — credit for the underlying methods belongs to
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

**Bug reports** — include:

- `scvelo-rs` version (`pip show scvelo-rs`)
- OS and Python version
- A minimum reproducer (a small `.h5ad` slice + the calls that fail
  is usually enough)
- What you expected vs what you got

**Parity issues** (a fitted parameter or velocity vector differs from
the original scvelo): include both runs' values for the affected
gene/cell, the relative drift, and which fixture you ran on.

**Feature requests** — describe the workflow you can't do today, not
just the API you'd like. Atlas-scale parity reports are especially
welcome.

For anything else, direct mail: ilay.kavitzky@gmail.com.
