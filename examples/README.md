# scvelo-rs examples

Self-contained scripts demonstrating the drop-in usage. They double as
acceptance tests for the install — if any of them errors, something's
wrong with your environment.

| script | what it does | runtime |
|---|---|---:|
| [`pancreas_end_to_end.py`](pancreas_end_to_end.py) | Full pipeline (`filter_and_normalize` → `moments` → `recover_dynamics` → `velocity` → `velocity_graph` → `velocity_pseudotime` → `velocity_confidence`) on the pancreas tutorial dataset, all Rust-backed via `import scvelo_rs as scv`. | ~10s |
| [`monkey_patch_existing_pipeline.py`](monkey_patch_existing_pipeline.py) | How to apply `scvelo_rs.patch` to an existing `import scvelo as scv` pipeline without touching downstream code. Shows revert via `scvelo_rs.patch.revert()`. | ~5s |

Run any example with:

```bash
python examples/pancreas_end_to_end.py
```

For numerical parity vs upstream scVelo on the same fixtures, see
[`notebooks/01_parity.py`](../notebooks/01_parity.py). For wall-time and
peak-memory benchmarks against upstream, see
[`notebooks/02_benchmarks.py`](../notebooks/02_benchmarks.py).
