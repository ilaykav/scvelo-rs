# Changelog

All notable changes to this project will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — first public release

### Added
- `tl.recover_dynamics` in Rust. Bit-exact equivalent to scvelo on the
  standard fixtures (pancreas, dentategyrus). ~134× wall-clock speedup,
  ~3× memory reduction.
- `tl.velocity` deterministic mode in Rust, including extreme-quantile
  percentile trim. Bit-exact.
- `tl.velocity_graph` per-cell cosine-similarity kernel in Rust
  (Rayon-parallel). Bit-exact on the standard fixtures.
- `pp.pca` via `nalgebra` SVD. Subspace-equivalent to scanpy / sklearn.
- `pp.neighbors` KNN via `hnsw_rs`. UMAP-style connectivities still
  computed by scanpy.
- Drop-in replacement: `import scvelo_rs as scv` gives the full
  `scv.tl/pp/pl/datasets` surface. Hot loops route through Rust;
  everything else passes through to scvelo / scanpy.
- Monkey-patch entry point: `import scvelo_rs.patch` replaces
  `scv.tl.recover_dynamics`, `scv.tl.velocity`, `scv.tl.velocity_graph`.
- Parity notebook (`notebooks/01_parity.py`) — Pearson r = 1.0000 across
  all fitted parameters on 4 standard fixtures.
- Benchmarking suite (`notebooks/02_benchmarks.py`) — measured 42× speedup,
  408 MB memory reduction on a 10k × 50 atlas.
- Cross-platform wheels (Linux x86_64/aarch64, macOS arm/x86_64, Windows
  x86_64) via GitHub Actions.

### Known limitations
- `tl.velocity` stochastic / dynamical modes fall through to scvelo.
  Stochastic mode is the most common case for users not running
  `recover_dynamics`; deterministic mode is what the dynamical pipeline
  uses internally.
- `pp.neighbors` falls through to scanpy after computing KNN. The Rust KNN
  kernel is callable directly via `scvelo_rs._scvelo_rs.knn_kernel`.
- 4 documented genes (across all dual-bit-exact fixtures) land in a
  different but equally-valid local minimum due to a numpy SIMD
  argsort tie-break. Both fits are valid; correlation against scvelo's
  output is still ≥ 0.99.
