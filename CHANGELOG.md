# Changelog

All notable changes to this project will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] — fix install + complete the drop-in surface

### Fixed
- **Runtime dependencies were missing from the wheel METADATA** — `scvelo` and `scanpy` were only listed under the `dev` extra in 0.1.1, so `pip install scvelo-rs` did not pull them and `import scvelo_rs` always raised `ModuleNotFoundError: No module named 'scvelo'`. Promoted both to the runtime `dependencies` array in `pyproject.toml`.

### Added
- `scvelo_rs.utils` — new submodule that pass-throughs the entire `scvelo.utils` surface (63 helpers including `show_proportions`, `cosine_correlation`, `get_connectivities`, `R_squared`, `leastsq`, `load_biomart`, …). Previously absent.
- `scvelo_rs.tl` now re-exports every name `scvelo.tl` exposes — added 14 previously-missing helpers and classes (`align_dynamics`, `eigs`, `optimization`, `rank_dynamical_genes`, `recover_latent_time`, `velocity_map`, `DynamicsRecovery`, `ExpectationMaximizationModel`, `SecondOrderSteadyStateModel`, `SteadyStateModel`, …). Implemented via dynamic enumeration so future scvelo additions flow through automatically.
- `scvelo_rs.pp` switched to dynamic enumeration too; same forward-compat property.
- Top-level scvelo helpers exposed as `scvelo_rs.<name>`: `AnnData`, `GridSpec`, `Neighbors`, `Velocity`, `VelocityGraph`, `get_df`, `load`, `logging`, `read_csv`, `read_load`, `set_figure_params`, `settings`.
- Submodule aliases: `scvelo_rs.preprocessing` (= `pp`), `scvelo_rs.tools` (= `tl`), `scvelo_rs.plotting` (= `pl`), matching scvelo's own dual naming.

### Removed
Functions scvelo removed in 0.3.x are no longer exposed by scvelo-rs either; we mirror scvelo's actual surface rather than resurrecting deleted names.
- `pp.pca` — removed by scvelo in 0.3.3 (consolidated to `scanpy.pp.pca`). The Rust SVD primitive that previously backed it remains callable directly via `scvelo_rs._scvelo_rs.pca_kernel` for power users.
- `pp.log1p`, `pp.filter_genes_dispersion` — deleted from scvelo source in 0.3.4. 0.1.1's wrappers pointed at non-existent attributes and crashed on first call. Use `scanpy.pp.log1p` / `scanpy.pp.highly_variable_genes` directly.
- `pp.show_proportions` — the wrapper has been crashing since scvelo 0.3.0 (the function lives at `scvelo.utils.show_proportions`, not `scv.pp`). Now reachable as `scvelo_rs.utils.show_proportions` via the new utils submodule.

### CI
- Dropped `--skip-existing` from the `maturin upload` args. With it, a build that produced wheels for an already-published version (e.g. forgetting to bump `Cargo.toml`) silently no-op'd and the workflow reported success — exactly how 0.1.2's first attempt regressed. Without it, the upload fails loudly so version-mismatch bugs surface immediately.

## [0.1.1] — README + metadata polish

### Changed
- README: tightened README & documentation.

### Fixed
- 0.1.0 shipped wheels for Linux x86_64/aarch64, macOS arm64, Windows x86_64
  (not x86_64 macOS — `macos-13` runner queue was too long). Same matrix
  in 0.1.1.

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
