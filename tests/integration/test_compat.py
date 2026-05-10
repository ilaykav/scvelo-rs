"""Compatibility tests — verify scvelo-rs behaves like scvelo for the
common usage patterns existing scvelo users have in their pipelines.

Each test asserts a contract that an existing scvelo user would expect to
hold after swapping `import scvelo as scv` for `import scvelo_rs as scv`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

warnings.filterwarnings("ignore")

_DATA_DIR = Path(__file__).parent.parent / "_data"


# ---------------------------------------------------------------------------
# Public-surface compatibility — `scvelo_rs as scv` exposes same API shape.
# ---------------------------------------------------------------------------


def test_public_surface_matches_scvelo():
    """Every name we promise (`tl`, `pp`, `pl`, `datasets`) must resolve."""
    import scvelo_rs as scv

    # Required submodules.
    for sub in ("tl", "pp", "pl", "datasets"):
        assert hasattr(scv, sub), f"missing submodule: scvelo_rs.{sub}"

    # Required tl functions (the ones existing pipelines call).
    for fn in (
        "recover_dynamics",
        "velocity",
        "velocity_graph",
        "transition_matrix",
        "terminal_states",
        "velocity_pseudotime",
        "velocity_embedding",
        "velocity_confidence",
        "rank_velocity_genes",
        "paga",
        "score_genes_cell_cycle",
        "latent_time",
    ):
        assert hasattr(scv.tl, fn), f"missing scvelo_rs.tl.{fn}"
        assert callable(getattr(scv.tl, fn))

    # Required pp functions.
    for fn in (
        "filter_and_normalize",
        "moments",
        "neighbors",
        "pca",
        "log1p",
        "filter_genes",
        "normalize_per_cell",
    ):
        assert hasattr(scv.pp, fn), f"missing scvelo_rs.pp.{fn}"
        assert callable(getattr(scv.pp, fn))

    # Required dataset loaders.
    for ds in ("pancreas", "dentategyrus", "bonemarrow", "simulation", "toy_data"):
        assert hasattr(scv.datasets, ds), f"missing scvelo_rs.datasets.{ds}"
        assert callable(getattr(scv.datasets, ds))


def test_recover_dynamics_signature_matches_scvelo():
    """Signature parity — every kwarg scvelo accepts, we accept (or `**kwargs`)."""
    import inspect

    import scvelo as scv
    import scvelo_rs

    upstream = inspect.signature(scv.tl.recover_dynamics)
    ours = inspect.signature(scvelo_rs.recover_dynamics)

    # Every named upstream kwarg should be present in ours.
    upstream_kwargs = {
        n
        for n, p in upstream.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    ours_kwargs = {
        n for n, p in ours.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in ours.parameters.values())

    missing = upstream_kwargs - ours_kwargs
    assert not missing or has_var_kwargs, f"recover_dynamics missing kwargs: {missing}"


# ---------------------------------------------------------------------------
# Patch path — applied/reverted cleanly, no leakage between tests.
# ---------------------------------------------------------------------------


def test_patch_apply_revert_idempotent():
    """`scvelo_rs.patch.apply()` and `revert()` must round-trip cleanly."""
    import scvelo as scv
    import scvelo_rs.patch

    # Establish baseline.
    scvelo_rs.patch.revert()
    assert hasattr(scv.tl, "recover_dynamics_original"), (
        "original was preserved on initial patch import"
    )

    upstream_recover = scv.tl.recover_dynamics_original

    # Apply twice — must be idempotent.
    scvelo_rs.patch.apply(verbose=False)
    after_apply = scv.tl.recover_dynamics
    scvelo_rs.patch.apply(verbose=False)
    assert scv.tl.recover_dynamics is after_apply, "apply not idempotent"

    # Revert restores upstream.
    scvelo_rs.patch.revert()
    assert scv.tl.recover_dynamics is upstream_recover, "revert didn't restore original"

    # Re-apply still works.
    scvelo_rs.patch.apply(verbose=False)
    assert scv.tl.recover_dynamics is not upstream_recover


def test_patch_covers_all_three_functions():
    """`scvelo_rs.patch` must patch all three Rust-backed entry points."""
    import scvelo as scv
    import scvelo_rs.patch

    scvelo_rs.patch.apply(verbose=False)

    # Each patched function must point at our `scvelo_rs._*` modules.
    assert scv.tl.recover_dynamics.__module__.startswith("scvelo_rs.")
    assert scv.tl.velocity.__module__.startswith("scvelo_rs.")
    assert scv.tl.velocity_graph.__module__.startswith("scvelo_rs.")

    # And originals are preserved.
    assert hasattr(scv.tl, "recover_dynamics_original")
    assert hasattr(scv.tl, "velocity_original")
    assert hasattr(scv.tl, "velocity_graph_original")


# ---------------------------------------------------------------------------
# Output-schema compatibility — adata fields downstream code depends on.
# ---------------------------------------------------------------------------


def test_recover_dynamics_writes_full_var_schema():
    """All `fit_*` columns scvelo writes must be present after our run."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )

    expected_cols = {
        "fit_alpha",
        "fit_beta",
        "fit_gamma",
        "fit_t_",
        "fit_scaling",
        "fit_std_u",
        "fit_std_s",
        "fit_likelihood",
        "fit_u0",
        "fit_s0",
        "fit_pval_steady",
        "fit_steady_u",
        "fit_steady_s",
        "fit_variance",
    }
    actual = set(adata.var.columns)
    missing = expected_cols - actual
    assert not missing, f"recover_dynamics missing var columns: {missing}"


def test_recover_dynamics_writes_full_layer_schema():
    """`fit_t`, `fit_tau`, `fit_tau_` layers must be present and full-shape."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )

    for layer in ("fit_t", "fit_tau", "fit_tau_"):
        assert layer in adata.layers, f"missing layer {layer}"
        assert adata.layers[layer].shape == adata.shape, (
            f"layer {layer} shape {adata.layers[layer].shape} != adata.shape {adata.shape}"
        )


def test_velocity_writes_genes_and_residuals():
    """`tl.velocity` deterministic output schema must match scvelo's."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scvelo_rs.velocity(adata, mode="deterministic")

    assert "velocity" in adata.layers
    assert adata.layers["velocity"].shape == adata.shape
    for col in (
        "velocity_gamma",
        "velocity_offset",
        "velocity_r2",
        "velocity_genes",
        "velocity_qreg_ratio",
    ):
        assert col in adata.var.columns, f"missing {col}"

    # `velocity_genes` must be a bool mask, not float.
    assert adata.var["velocity_genes"].dtype == bool, (
        f"velocity_genes dtype is {adata.var['velocity_genes'].dtype}, expected bool"
    )

    # uns params reflect mode.
    assert adata.uns.get("velocity_params", {}).get("mode") == "deterministic"


def test_velocity_graph_writes_csr():
    """`tl.velocity_graph` writes a CSR sparse matrix to adata.uns."""
    import scvelo_rs
    from scipy.sparse import csr_matrix, issparse

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scvelo_rs.velocity(adata, mode="deterministic")
    scvelo_rs.velocity_graph(adata, show_progress_bar=False)

    assert "velocity_graph" in adata.uns
    g = adata.uns["velocity_graph"]
    assert issparse(g) and isinstance(g, csr_matrix), f"velocity_graph not CSR: {type(g)}"
    assert g.shape == (adata.n_obs, adata.n_obs)
    assert "velocity_graph_neg" in adata.uns
    assert "velocity_self_transition" in adata.obs.columns


# ---------------------------------------------------------------------------
# Stochastic mode pass-through — currently routed to scvelo upstream.
# Make sure it still works (i.e., doesn't crash, produces a velocity layer).
# ---------------------------------------------------------------------------


def test_velocity_stochastic_falls_through_cleanly(monkeypatch):
    """`mode='stochastic'` must dispatch to scvelo upstream (we don't reimplement
    second-order moments). Verifies the routing, not scvelo's own behavior."""
    import scvelo as scv
    import scvelo_rs
    import scvelo_rs._velocity

    called = {}

    def fake_scv_velocity(*args, **kwargs):
        called["mode"] = kwargs.get("mode")
        called["vkey"] = kwargs.get("vkey")
        return None

    # Patch both the original-preserved name and scv.tl.velocity itself —
    # _fallback_velocity prefers velocity_original if present.
    monkeypatch.setattr(scv.tl, "velocity_original", fake_scv_velocity, raising=False)
    monkeypatch.setattr(scv.tl, "velocity", fake_scv_velocity, raising=False)

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    scvelo_rs.velocity(adata, mode="stochastic")
    assert called["mode"] == "stochastic", "stochastic mode didn't reach upstream scvelo"


# ---------------------------------------------------------------------------
# Downstream chains — recover_dynamics → velocity → velocity_graph
# → transition_matrix → terminal_states → velocity_pseudotime should all run.
# ---------------------------------------------------------------------------


def test_full_dynamical_pipeline_via_drop_in():
    """End-to-end: the typical scvelo dynamical-model pipeline runs through
    `import scvelo_rs as scv` without any error."""
    import scvelo_rs as scv

    adata = sc.read(str(_DATA_DIR / "pancreas_100obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)

    # Use PCA basis (already populated) since fixture has no UMAP.
    scv.tl.recover_dynamics(adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata, show_progress_bar=False)
    scv.tl.velocity_embedding(adata, basis="pca")

    assert "velocity_pca" in adata.obsm
    assert adata.obsm["velocity_pca"].shape[0] == adata.n_obs


# ---------------------------------------------------------------------------
# Error handling — bad inputs should give clear errors, not segfaults.
# ---------------------------------------------------------------------------


def test_recover_dynamics_missing_layers_raises():
    """Missing `Mu`/`Ms` AND missing `unspliced`/`spliced` should raise
    cleanly, not segfault."""
    import anndata as ad
    import scvelo_rs

    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.normal(size=(20, 10)).astype(np.float32))

    with pytest.raises((ValueError, KeyError)):
        scvelo_rs.recover_dynamics(
            adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
        )


def test_recover_dynamics_unsupported_kwargs_raise():
    """Unsupported kwargs must raise NotImplementedError — never silently
    differ from scvelo's behavior."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)

    with pytest.raises(NotImplementedError):
        scvelo_rs.recover_dynamics(adata, fit_basal_transcription=True, show_progress_bar=False)
    with pytest.raises(NotImplementedError):
        scvelo_rs.recover_dynamics(adata, return_model=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Determinism — same input twice must give bit-exact same output.
# ---------------------------------------------------------------------------


def test_recover_dynamics_deterministic():
    import scvelo_rs

    adata1 = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata2 = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    for adata in (adata1, adata2):
        adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
        adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    scvelo_rs.recover_dynamics(
        adata1, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )
    scvelo_rs.recover_dynamics(
        adata2, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )

    for col in ("fit_alpha", "fit_beta", "fit_gamma", "fit_t_", "fit_likelihood"):
        v1 = adata1.var[col].to_numpy()
        v2 = adata2.var[col].to_numpy()
        nan_mask = np.isnan(v1) & np.isnan(v2)
        if not nan_mask.all():
            np.testing.assert_array_equal(
                v1[~nan_mask], v2[~nan_mask], err_msg=f"{col}: non-deterministic across runs"
            )


# ---------------------------------------------------------------------------
# Sparse-layer support — scvelo accepts sparse Mu/Ms; we must too.
# ---------------------------------------------------------------------------


def test_recover_dynamics_handles_sparse_layers():
    """If Mu/Ms come in as sparse (CSR), we must still produce sane output."""
    import scvelo_rs
    from scipy.sparse import csr_matrix

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = csr_matrix(np.asarray(adata.layers["Mu"], dtype=np.float64))
    adata.layers["Ms"] = csr_matrix(np.asarray(adata.layers["Ms"], dtype=np.float64))

    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, show_progress_bar=False, t_max=False
    )

    assert "fit_alpha" in adata.var.columns
    n_attempted = (~np.isnan(adata.var["fit_alpha"].to_numpy())).sum()
    assert n_attempted > 0, "no genes fit when Mu/Ms passed as sparse"


# ---------------------------------------------------------------------------
# var_names selectors — "all", "velocity_genes", explicit list.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression tests — bugs surfaced and fixed by the compat suite.
# ---------------------------------------------------------------------------


def test_use_raw_true_accepted():
    """`use_raw=True` must work (scvelo supports it; we used to falsely reject)."""
    import scvelo_rs
    from scipy.sparse import issparse

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    # use_raw=True means scvelo reads `unspliced`/`spliced` from adata.layers,
    # NOT from adata.raw.layers. Densify in case the fixture stores them sparse.
    for k in ("unspliced", "spliced"):
        layer = adata.layers[k]
        if issparse(layer):
            layer = layer.toarray()
        adata.layers[k] = np.asarray(layer, dtype=np.float64)
    scvelo_rs.recover_dynamics(
        adata, var_names="all", n_jobs=1, use_raw=True, show_progress_bar=False, t_max=False
    )
    assert "fit_alpha" in adata.var.columns


def test_recover_dynamics_empty_var_names_list_raises():
    """An empty list must raise ValueError, not silently produce all-NaN fits."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    with pytest.raises(ValueError, match="empty"):
        scvelo_rs.recover_dynamics(adata, var_names=[], n_jobs=1, show_progress_bar=False)


def test_recover_dynamics_no_matching_var_names_raises():
    """An explicit list with no matches must raise, not silently no-op."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    with pytest.raises(ValueError, match="None of the"):
        scvelo_rs.recover_dynamics(
            adata,
            var_names=["__not_a_real_gene__"],
            n_jobs=1,
            show_progress_bar=False,
        )


def test_recover_dynamics_velocity_genes_all_false_warns():
    """When `velocity_genes` exists but is all-False, must warn and fall back
    instead of silently producing a no-op pipeline."""
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    adata.var["velocity_genes"] = np.zeros(adata.n_vars, dtype=bool)

    with pytest.warns(UserWarning, match="all False"):
        scvelo_rs.recover_dynamics(
            adata,
            var_names="velocity_genes",
            n_jobs=1,
            show_progress_bar=False,
            t_max=False,
        )
    # After fallback to all genes, fits must populate.
    assert (~np.isnan(adata.var["fit_alpha"].to_numpy())).sum() > 0


def test_velocity_graph_uses_our_velocity_when_auto_filling(monkeypatch):
    """When `velocity` layer is missing, `velocity_graph` must call OUR Rust
    velocity wrapper, not bypass to scvelo's directly. Regression: previously
    `velocity_graph` did `from scvelo.tools.velocity import velocity` which
    imports scvelo's function unconditionally, ignoring the patch surface."""
    import scvelo_rs
    import scvelo_rs._velocity as _v

    auto_called = {"ours": False}

    def spy_velocity(adata, vkey="velocity", **kwargs):
        auto_called["ours"] = True
        # Stub out: write a dummy velocity layer so velocity_graph proceeds.
        adata.layers[vkey] = np.zeros_like(adata.layers["Ms"])
        adata.var[f"{vkey}_genes"] = np.ones(adata.n_vars, dtype=bool)
        adata.uns[f"{vkey}_params"] = {"mode": "deterministic"}

    monkeypatch.setattr(_v, "velocity", spy_velocity)

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)
    # No `velocity` layer pre-computed → graph wrapper must auto-fill via
    # our wrapper.
    scvelo_rs.velocity_graph(adata, show_progress_bar=False)
    assert auto_called["ours"], "velocity_graph bypassed our velocity wrapper"


@pytest.mark.parametrize("var_names_arg", ["all", "explicit_first_20", "velocity_genes"])
def test_recover_dynamics_var_names_modes(var_names_arg):
    import scvelo_rs

    adata = sc.read(str(_DATA_DIR / "pancreas_50obs_preprocessed.h5ad"))
    adata.layers["Mu"] = np.asarray(adata.layers["Mu"], dtype=np.float64)
    adata.layers["Ms"] = np.asarray(adata.layers["Ms"], dtype=np.float64)

    if var_names_arg == "explicit_first_20":
        var_names = list(adata.var_names[:20])
    else:
        var_names = var_names_arg

    scvelo_rs.recover_dynamics(
        adata, var_names=var_names, n_jobs=1, show_progress_bar=False, t_max=False
    )
    assert "fit_alpha" in adata.var.columns
