---
name: Parity issue
about: scvelo-rs returns a numerically different result than upstream scVelo
labels: parity
---

## Affected output

<!-- Which column / layer / array differs? e.g. `adata.var["fit_alpha"]`,
`adata.layers["velocity"]`, `adata.uns["velocity_graph"]`. -->

## Drift you observed

<!-- A few representative values from each side. Numbers are most useful. -->

| metric / gene | upstream scvelo | scvelo-rs | abs diff | rel diff |
|---|---|---|---|---|
|  |  |  |  |  |

## Reproducer

```python
import scvelo as scv
import scvelo_rs

adata = ...  # how you build/load the input
# scvelo run
# scvelo-rs run
# code that prints the diff
```

## Fixture

- Dataset: <!-- pancreas / dentategyrus / your own atlas / synthetic -->
- Cells × genes:
- Did you cast `Mu` and `Ms` to `np.float64` before calling? <!-- yes / no
  (scvelo's f32 inputs cause documented sub-ULP drift; f64 is bit-exact). -->

## Environment

- `scvelo-rs` version:
- `scvelo` version:
- Python / OS:
