# vendor/

Real-world scvelo workflows vendored from published sources, used by the
`vendor` category in `notebooks/02_benchmarks.py` to measure end-to-end
speedup against stock scvelo on representative pipelines.

Each subdirectory under `workflows/` contains a `run.py` exposing
`load_data() -> AnnData` and `run(lib, adata) -> None`; the bench harness
calls these once per backend with identical input. Scientific context,
citations, and results for every workflow are documented in the top-level
[README](../README.md#real-world-end-to-end-workflows).

## Current workflows

| Workflow | Cells | Tissue | Downstream tool | Upstream source |
|---|---:|---|---|---|
| `pancreas_tutorial` | 3,696 | Mouse pancreas | vanilla scvelo + latent_time | [theislab/scvelo_notebooks](https://github.com/theislab/scvelo_notebooks) |
| `gastrulation_e75_diffkinetics` | ~21,000 | Mouse embryo (E7.5) | `differential_kinetic_test` + per-cluster refit | scvelo `DifferentialKinetics.ipynb` + Pijuan-Sala 2019 |
| `cellrank2_hematopoiesis` | ~24,000 | Human bone marrow | CellRank VelocityKernel + GPCCA | [theislab/cellrank2_reproducibility](https://github.com/theislab/cellrank2_reproducibility) |
| `pbmc68k_pipeline` | ~68,000 | Human PBMCs | vanilla scvelo + latent_time | scvelo + Zheng 2017 |
| `mouse_gastrulation_atlas` | ~116,000 | Mouse embryo | atlas-scale scvelo dynamical | scvelo + Pijuan-Sala 2019 |

All workflows use public h5ad fetched via `scv.datasets.*` or `cr.datasets.*`
(no auth, no manual figshare lookups). Data caches under `~/.scvelo`.

## Bench-time dependencies

These workflows pull in scvelo, scanpy and CellRank. Install with:

```bash
pip install scvelo scanpy cellrank psutil py-cpuinfo
```

The benchmarks workflow (`.github/workflows/benchmarks.yml`) installs these
automatically.

## Adding a new workflow

1. Create `vendor/workflows/<name>/run.py` exposing `load_data() -> AnnData`
   and `run(lib, adata) -> None`.
2. Add a `Bench(category="vendor", workflow="<name>", long=True, …)` entry
   to `BENCHMARKS` in `notebooks/02_benchmarks.py`.
3. Document the dataset, citation, pipeline, and result in the top-level
   [README](../README.md#real-world-end-to-end-workflows).
4. If the workflow is known to OOM or time out on stock scvelo, set
   `skip_scvelo=True` with a `skip_scvelo_reason`.

## Upstream attribution

All vendored workflows are derivative of upstream code released under
BSD-3-Clause (scvelo and CellRank), cited per workflow in the top-level README.
