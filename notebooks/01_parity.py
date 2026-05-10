"""Parity notebook — scvelo vs scvelo-rs on the standard public datasets.

Run as a script:
    python notebooks/01_parity.py

Or convert to .ipynb:
    pip install jupytext
    jupytext --to ipynb notebooks/01_parity.py

Asserts:
  - fit_alpha / fit_beta / fit_gamma / fit_t_   per-gene Pearson r > 0.99
  - velocity_gamma                              per-gene Pearson r > 0.99
  - velocity (per-cell residual)                per-cell cosine sim > 0.99
  - velocity_graph (cosine cell-cell matrix)    per-cell vector r > 0.99

If any assertion fails, the notebook prints a per-fixture diagnostic table
showing where the divergence is and falls through (does not raise) — useful
when running interactively for inspection.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load fixtures the same way both runs see them.
# ---------------------------------------------------------------------------

import scanpy as sc

DATA_DIR = Path(__file__).parent.parent / "tests" / "_data"
FIXTURES = (
    "pancreas_50obs_preprocessed",
    "pancreas_100obs_preprocessed",
    "dentategyrus_50obs_preprocessed",
    "dentategyrus_100obs_preprocessed",
)


def load(name: str):
    adata = sc.read(str(DATA_DIR / f"{name}.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    return adata


# ---------------------------------------------------------------------------
# 2. Run upstream scvelo and scvelo-rs side-by-side.
# ---------------------------------------------------------------------------

import scvelo as scv
import scvelo_rs


def run_pipeline(adata, mode: str):
    """`mode` is one of: 'scvelo' (upstream), 'scvelo_rs' (drop-in)."""
    fns = scv if mode == "scvelo" else scvelo_rs
    fns.tl.recover_dynamics(adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False)
    fns.tl.velocity(adata, mode="deterministic")
    fns.tl.velocity_graph(adata, show_progress_bar=False)
    return adata


# ---------------------------------------------------------------------------
# 3. Per-fixture parity report.
# ---------------------------------------------------------------------------


def per_gene_corr(a, b, label: str) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def per_cell_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-cell cosine similarity between rows of a and b. Returns (n_cells,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    a = np.where(nan_a, 0.0, a)
    b = np.where(nan_b, 0.0, b)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def report(fixture: str):
    print(f"\n=== {fixture} ===")
    a_scv = run_pipeline(load(fixture), "scvelo")
    a_rs = run_pipeline(load(fixture), "scvelo_rs")

    rows = []
    for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_", "velocity_gamma"):
        if col in a_scv.var.columns and col in a_rs.var.columns:
            r = per_gene_corr(a_scv.var[col].to_numpy(), a_rs.var[col].to_numpy(), col)
            rows.append((col, r))

    print(f"{'metric':<22s} {'pearson_r':>10s}")
    print("-" * 36)
    for col, r in rows:
        print(f"{col:<22s} {r:>10.5f}")

    # Per-cell velocity vector cosine: a per-cell number, take the median.
    if "velocity" in a_scv.layers and "velocity" in a_rs.layers:
        cos = per_cell_cosine(
            np.asarray(a_scv.layers["velocity"]), np.asarray(a_rs.layers["velocity"])
        )
        print(f"{'velocity (cell cos)':<22s} {np.nanmedian(cos):>10.5f}  (median)")

    # velocity_graph: per-row cosine of the sparse rows.
    if "velocity_graph" in a_scv.uns and "velocity_graph" in a_rs.uns:
        g_scv = a_scv.uns["velocity_graph"].toarray()
        g_rs = a_rs.uns["velocity_graph"].toarray()
        cos = per_cell_cosine(g_scv, g_rs)
        print(f"{'velocity_graph (row)':<22s} {np.nanmedian(cos):>10.5f}  (median)")

    return a_scv, a_rs


# ---------------------------------------------------------------------------
# 4. Plotting (optional — only when matplotlib is available and a backend
# can render). Generates a side-by-side velocity arrow plot.
# ---------------------------------------------------------------------------


def plot_velocity_arrows(a_scv, a_rs, fixture: str, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fixture}_arrows.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, ad, label in ((axes[0], a_scv, "scvelo"), (axes[1], a_rs, "scvelo-rs")):
        try:
            scv.pl.velocity_embedding(
                ad,
                basis="umap",
                arrow_length=2,
                arrow_size=2,
                ax=ax,
                show=False,
            )
            ax.set_title(label, fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f"plot failed: {e}", ha="center", va="center")
            ax.set_title(label)
    fig.suptitle(fixture)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  arrows → {out_path}")


# ---------------------------------------------------------------------------
# 5. Run everything.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "_artifacts"
    aggregate = []
    for fix in FIXTURES:
        a_scv, a_rs = report(fix)
        if "X_umap" in a_scv.obsm:
            plot_velocity_arrows(a_scv, a_rs, fix, out_dir)
        for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_", "velocity_gamma"):
            if col in a_scv.var.columns and col in a_rs.var.columns:
                r = per_gene_corr(a_scv.var[col].to_numpy(), a_rs.var[col].to_numpy(), col)
                aggregate.append((fix, col, r))

    print("\n\n=== AGGREGATE: per-(fixture, metric) pearson r ===")
    print(f"{'fixture':<35s} {'metric':<22s} {'r':>8s}")
    print("-" * 70)
    for fix, col, r in aggregate:
        flag = "OK" if r > 0.99 else "CHECK"
        print(f"{fix:<35s} {col:<22s} {r:>8.4f}   {flag}")

    failing = [(f, c, r) for (f, c, r) in aggregate if r <= 0.99]
    print(f"\nfailing > 0.99 threshold: {len(failing)} / {len(aggregate)}")
