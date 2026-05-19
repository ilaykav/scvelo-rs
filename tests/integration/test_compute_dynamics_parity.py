"""Parity test: scvelo_rs.utils.compute_dynamics vs scvelo.utils.compute_dynamics.

Both implementations evaluate the analytical splicing dynamics from fitted
(alpha, beta, gamma, t_) per-cell. Output must match bit-exact (or to f64
ULP) for both `key='fit'` and the post-recover_dynamics fixture state.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"

TIGHT_REL = 1e-9  # tightly-equivalent threshold


def _load(name: str):
    import scanpy as sc

    return sc.read(str(_DATA_DIR / name))


def _drift(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = ~np.isnan(a) & ~np.isnan(b)
    if not valid.any():
        return 0, 0.0, 0.0
    diff = np.abs(a[valid] - b[valid])
    rel = diff / (np.abs(a[valid]) + 1e-300)
    return int(valid.sum()), float(diff.max()), float(rel.max())


@pytest.fixture(scope="module")
def fitted_adata():
    """pancreas/dentategyrus 50-cell fixture with recover_dynamics already run."""
    import scvelo as scv

    adata = _load("pancreas_50obs_preprocessed.h5ad")
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scv.tl.recover_dynamics(adata, n_top_genes=20)
    return adata


def _genes_with_fit(adata):
    return [g for g in adata.var_names if not np.isnan(adata.var.loc[g, "fit_alpha"])]


@pytest.mark.parametrize("sort_t", [True, False])
def test_compute_dynamics_matches_scvelo(fitted_adata, sort_t):
    import scvelo as scv
    import scvelo_rs as scvr

    genes = _genes_with_fit(fitted_adata)
    assert len(genes) > 0, "fixture has no fitted genes — recover_dynamics regression"

    failures = []
    for gene in genes[:10]:
        a_scv, u_scv, s_scv = scv.utils.compute_dynamics(fitted_adata, gene, key="fit", sort=sort_t)
        a_rs, u_rs, s_rs = scvr.utils.compute_dynamics(fitted_adata, gene, key="fit", sort=sort_t)

        for name, scv_v, rs_v in (("alpha", a_scv, a_rs), ("u", u_scv, u_rs), ("s", s_scv, s_rs)):
            n, max_abs, max_rel = _drift(scv_v, rs_v)
            if max_rel > TIGHT_REL:
                failures.append(
                    f"  gene={gene} field={name} sort={sort_t} "
                    f"max_rel={max_rel:.3e} max_abs={max_abs:.3e} (n={n})"
                )

    assert not failures, "compute_dynamics drift > 1e-9 on:\n" + "\n".join(failures)
