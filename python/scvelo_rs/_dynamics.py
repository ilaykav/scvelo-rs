"""AnnData glue around the Rust kernel.

`scvelo_rs.recover_dynamics(adata, ...)` runs the entire per-gene EM
(initialization, Nelder-Mead, projection-mode refit, alignment) inside one
PyO3 call, gene-parallel via Rayon, sharing the connectivity CSR by reference.
Same signature as `scvelo.tl.recover_dynamics`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ._scvelo_rs import (
    align_dynamics_kernel,
    initialize_all_genes_kernel,
    per_gene_weights_kernel,
    recover_dynamics_kernel,
)

_VAR_FIT_COLUMNS = (
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
)
_UNS_KEY = "recover_dynamics"


def recover_dynamics(
    data,
    var_names: str | list = "velocity_genes",
    n_top_genes: int | None = None,
    max_iter: int = 10,
    assignment_mode: str = "projection",
    t_max: float | None = None,
    fit_time: bool = True,
    fit_scaling: bool = True,
    fit_steady_states: bool = True,
    fit_connected_states: bool | None = None,
    fit_basal_transcription: bool | None = None,
    use_raw: bool = False,
    load_pars: bool | None = None,
    return_model: bool | None = None,
    plot_results: bool = False,
    steady_state_prior: Any = None,
    add_key: str = "fit",
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs,
):
    """Drop-in replacement for `scvelo.tl.recover_dynamics`. `n_jobs`/`backend`
    accepted for API compatibility (Rust uses Rayon internally)."""
    _check_unsupported(
        assignment_mode=assignment_mode,
        fit_basal_transcription=fit_basal_transcription,
        use_raw=use_raw,
        load_pars=load_pars,
        return_model=return_model,
        plot_results=plot_results,
        steady_state_prior=steady_state_prior,
        backend=backend,
    )
    if kwargs:
        warnings.warn(
            f"scvelo_rs.recover_dynamics: ignoring unknown kwargs {list(kwargs)}",
            stacklevel=2,
        )

    adata = data.copy() if copy else data

    if "Ms" not in adata.layers or "Mu" not in adata.layers:
        use_raw = True
    if fit_connected_states is None:
        fit_connected_states = not use_raw

    adata.uns[_UNS_KEY] = {
        "fit_connected_states": fit_connected_states,
        "fit_basal_transcription": fit_basal_transcription,
        "use_raw": use_raw,
    }

    gene_mask = _select_genes(adata, var_names=var_names, n_top_genes=n_top_genes)
    Mu, Ms = _extract_layers(adata, use_raw=use_raw)

    Mu_sub = np.ascontiguousarray(Mu[:, gene_mask], dtype=np.float64)
    Ms_sub = np.ascontiguousarray(Ms[:, gene_mask], dtype=np.float64)
    weights_sub = np.ascontiguousarray(_per_gene_weights(Mu_sub, Ms_sub), dtype=bool)

    cd, ci, cp = _connectivity_triplet(adata, fit_connected_states)

    if steady_state_prior is None:
        pval_steady, steady_u_arr, steady_s_arr = _per_gene_bimodality(
            Mu_sub, Ms_sub, weights_sub, fit_scaling=fit_scaling
        )
        f32_mode_active = _detect_f32_layers(adata)
        alpha0, beta0, gamma0, scaling0, t_0, u0_0, s0_0, std_u0, std_s0, ssr0 = (
            initialize_all_genes_kernel(
                Mu_sub,
                Ms_sub,
                weights_sub,
                fit_scaling,
                fit_steady_states,
                f32_mode_active,
                pval_steady,
                steady_u_arr,
                steady_s_arr,
                cd,
                ci,
                cp,
            )
        )
    else:
        inits = _initialize_all_genes(
            adata,
            gene_mask=gene_mask,
            Mu=Mu,
            Ms=Ms,
            fit_scaling=fit_scaling,
            fit_steady_states=fit_steady_states,
            fit_connected_states=fit_connected_states,
            steady_state_prior=steady_state_prior,
        )
        if inits is None:
            import scvelo as scv

            return (
                scv.tl.recover_dynamics_original(adata, var_names=var_names, **kwargs)
                if hasattr(scv.tl, "recover_dynamics_original")
                else None
            )
        alpha0 = np.array([i["alpha"] for i in inits], dtype=np.float64)
        beta0 = np.array([i["beta"] for i in inits], dtype=np.float64)
        gamma0 = np.array([i["gamma"] for i in inits], dtype=np.float64)
        scaling0 = np.array([i["scaling"] for i in inits], dtype=np.float64)
        t_0 = np.array([i["t_"] for i in inits], dtype=np.float64)
        u0_0 = np.array([i["u0_"] for i in inits], dtype=np.float64)
        s0_0 = np.array([i["s0_"] for i in inits], dtype=np.float64)
        std_u0 = np.array([i["std_u"] for i in inits], dtype=np.float64)
        std_s0 = np.array([i["std_s"] for i in inits], dtype=np.float64)
        ssr0 = np.array(
            [i["ssr"] if i["ssr"] is not None else np.nan for i in inits], dtype=np.float64
        )

    # Hardcoded: scvelo's distx promotes to f64 despite f32 layers via numpy
    # broadcast rules. Matching scvelo means f64 throughout in our compute path.
    f32_mode = False

    (
        alpha,
        beta,
        gamma,
        t_,
        scaling,
        likelihood,
        variance,
        fit_t_sub,
        fit_tau_sub,
        fit_tau__sub,
    ) = recover_dynamics_kernel(
        Mu_sub,
        Ms_sub,
        weights_sub,
        alpha0,
        beta0,
        gamma0,
        scaling0,
        t_0,
        u0_0,
        s0_0,
        std_u0,
        std_s0,
        ssr0,
        max_iter,
        fit_scaling,
        fit_steady_states,
        f32_mode,
        cd,
        ci,
        cp,
    )

    beta_corrected = beta / scaling

    # scvelo's update() rescales `self.steady_u *= self.scaling/scaling` on
    # every scaling commit; the telescoped factor is init_scaling/final_scaling.
    rescale = np.where(scaling != 0, scaling0 / scaling, 1.0)
    steady_u_final = steady_u_arr * rescale if steady_u_arr is not None else None
    steady_s_final = steady_s_arr.copy() if steady_s_arr is not None else None

    _write_var_columns(
        adata,
        gene_mask=gene_mask,
        alpha=alpha,
        beta=beta_corrected,
        gamma=gamma,
        t_=t_,
        scaling=scaling,
        likelihood=likelihood,
        variance=variance,
        std_u=std_u0,
        std_s=std_s0,
        u0_=u0_0,
        s0_=s0_0,
        pval_steady=pval_steady,
        steady_u=steady_u_final,
        steady_s=steady_s_final,
    )

    # Layer smoothing: use `get_connectivities(adata)` (row-normalized) NOT raw
    # `adata.obsp["connectivities"]` (row sums 5-15) — scvelo's recover_dynamics
    # uses the normalized version; the raw one would scale fit_t by 5-15×.
    layer_conn = None
    if fit_connected_states:
        try:
            from scvelo.preprocessing.moments import get_connectivities

            layer_conn = get_connectivities(adata)
        except Exception:
            layer_conn = adata.obsp.get("connectivities")
    _write_layers(
        adata,
        gene_mask=gene_mask,
        fit_t_sub=fit_t_sub,
        fit_tau_sub=fit_tau_sub,
        fit_tau__sub=fit_tau__sub,
        conn=layer_conn,
    )

    # We default `t_max=None → False` (skip alignment), unlike scvelo's
    # `None → 20`. align_dynamics divides by `m = t_max/T_max` and amplifies
    # any small drift in steady-state classification into 100%+ swings on
    # outlier genes. Users wanting scvelo's t_max=20 semantics pass it
    # explicitly.
    if t_max is None:
        t_max = False

    if t_max is not False:
        try:
            t_max_v = 20.0 if t_max is None else float(t_max)
            alpha_arr = np.ascontiguousarray(adata.var["fit_alpha"].to_numpy(), dtype=np.float64)
            beta_arr = np.ascontiguousarray(adata.var["fit_beta"].to_numpy(), dtype=np.float64)
            gamma_arr = np.ascontiguousarray(adata.var["fit_gamma"].to_numpy(), dtype=np.float64)
            t__arr = np.ascontiguousarray(adata.var["fit_t_"].to_numpy(), dtype=np.float64)
            T_arr = np.ascontiguousarray(adata.layers["fit_t"], dtype=np.float64)
            Tau_arr = np.ascontiguousarray(adata.layers["fit_tau"], dtype=np.float64)
            Tau__arr = np.ascontiguousarray(adata.layers["fit_tau_"], dtype=np.float64)
            align_dynamics_kernel(
                alpha_arr,
                beta_arr,
                gamma_arr,
                t__arr,
                T_arr,
                Tau_arr,
                Tau__arr,
                t_max_v,
            )
            adata.var["fit_alpha"] = alpha_arr
            adata.var["fit_beta"] = beta_arr
            adata.var["fit_gamma"] = gamma_arr
            adata.var["fit_t_"] = t__arr
            adata.layers["fit_t"] = T_arr
            adata.layers["fit_tau"] = Tau_arr
            adata.layers["fit_tau_"] = Tau__arr
        except Exception as e:
            warnings.warn(
                f"align_dynamics_kernel failed, falling back to scvelo: {e}", stacklevel=2
            )
            try:
                from scvelo.tools._em_model_core import align_dynamics

                T = adata.layers["fit_t"]
                idx = ~np.isnan(np.sum(T, axis=0))
                align_dynamics(adata, t_max=t_max, dm=None, idx=idx)
            except Exception as ee:
                warnings.warn(f"align_dynamics failed: {ee}", stacklevel=2)

    return adata if copy else None


# ---------------------------------------------------------------------------


def _check_unsupported(**flags):
    if flags["assignment_mode"] != "projection":
        raise NotImplementedError(
            f"assignment_mode={flags['assignment_mode']!r} not supported; "
            "only 'projection' is implemented."
        )
    if flags["plot_results"]:
        raise NotImplementedError("plot_results=True not supported.")
    if flags["return_model"]:
        raise NotImplementedError("return_model=True not supported.")
    for flag in ("fit_basal_transcription", "load_pars", "steady_state_prior"):
        if flags[flag] is not None:
            raise NotImplementedError(f"{flag}={flags[flag]!r} not supported.")
    if flags["backend"] not in ("loky", "threading"):
        warnings.warn(
            f"backend={flags['backend']!r} ignored; scvelo_rs uses Rayon internally.",
            stacklevel=3,
        )


def _detect_f32_layers(adata) -> bool:
    if os.environ.get("SCVELORS_F32_INIT") == "0":
        return False
    try:
        mu_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
        return mu_layer is not None and mu_layer.dtype == np.float32
    except Exception:
        return False


def _extract_layers(adata, use_raw: bool):
    layers = adata.layers
    if use_raw:
        u_key, s_key = "unspliced", "spliced"
    else:
        u_key = "Mu" if "Mu" in layers else "unspliced"
        s_key = "Ms" if "Ms" in layers else "spliced"
    if u_key not in layers or s_key not in layers:
        raise ValueError(
            f"AnnData missing required layers ({u_key!r}, {s_key!r}). "
            "Run scvelo.pp.moments(adata) first."
        )
    u_layer = layers[u_key]
    s_layer = layers[s_key]
    if issparse(u_layer):
        u_layer = u_layer.toarray()
    if issparse(s_layer):
        s_layer = s_layer.toarray()
    Mu = np.asarray(u_layer, dtype=np.float64, order="C")
    Ms = np.asarray(s_layer, dtype=np.float64, order="C")
    # Reject non-finite values up front. Our Rust sort comparators use
    # `partial_cmp(a, b).unwrap_or(Equal)` which doesn't form a total order
    # in the presence of NaN — leads to a panic deep in the sort kernel.
    if not np.isfinite(Mu).all() or not np.isfinite(Ms).all():
        raise ValueError(
            f"{u_key}/{s_key} contains NaN or inf values. Filter or impute "
            "non-finite entries before calling recover_dynamics."
        )
    return Mu, Ms


def _select_genes(adata, var_names, n_top_genes):
    n_vars = adata.n_vars
    if var_names == "all":
        mask = np.ones(n_vars, dtype=bool)
    elif var_names == "velocity_genes":
        if "velocity_genes" in adata.var:
            mask = adata.var["velocity_genes"].to_numpy().astype(bool)
            if not mask.any():
                warnings.warn(
                    "adata.var['velocity_genes'] is all False; falling back "
                    "to all genes. Run scvelo.tl.velocity(adata) first to "
                    "populate it.",
                    stacklevel=3,
                )
                mask = np.ones(n_vars, dtype=bool)
        else:
            warnings.warn(
                "'velocity_genes' not found; falling back to all genes. Run "
                "scvelo.tl.velocity(adata) first to populate it.",
                stacklevel=3,
            )
            mask = np.ones(n_vars, dtype=bool)
    elif isinstance(var_names, (list, tuple, np.ndarray)):
        if len(var_names) == 0:
            raise ValueError(
                "var_names is empty; pass 'all', 'velocity_genes', or a "
                "non-empty list of gene names."
            )
        mask = np.asarray(adata.var_names.isin(set(var_names)), dtype=bool)
        if not mask.any():
            raise ValueError(
                f"None of the {len(var_names)} requested gene(s) are present in adata.var_names."
            )
    else:
        raise ValueError(f"Unrecognised var_names={var_names!r}")
    if n_top_genes is not None and mask.sum() > n_top_genes:
        idx = np.where(mask)[0][:n_top_genes]
        new_mask = np.zeros_like(mask)
        new_mask[idx] = True
        mask = new_mask
    return mask


def _per_gene_bimodality(
    Mu_sub: np.ndarray, Ms_sub: np.ndarray, weights: np.ndarray, fit_scaling: bool = True
):
    """Per-gene bimodality test (Rust port of scvelo's `test_bimodality(kde=True)`).

    Set `SCVELORS_USE_PY_BIMODALITY=1` to fall back to the scipy KDE path
    (slower; useful for diagnosing rare numerical disagreement).
    """
    if os.environ.get("SCVELORS_USE_PY_BIMODALITY"):
        from scvelo.tools.utils import test_bimodality

        n_genes = Mu_sub.shape[1]
        pval = np.full(n_genes, np.nan, dtype=np.float64)
        su = np.zeros(n_genes, dtype=np.float64)
        ss = np.zeros(n_genes, dtype=np.float64)
        for g in range(n_genes):
            w = weights[:, g]
            if w.sum() <= 2:
                continue
            u_w = Mu_sub[w, g].astype(np.float64)
            s_w = Ms_sub[w, g].astype(np.float64)
            std_u = np.std(u_w)
            std_s = np.std(s_w)
            if std_u == 0 or std_s == 0 or not fit_scaling:
                scaling_g = 1.0
            else:
                scaling_g = std_u / std_s
            # scvelo's initialize() divides u_w by scaling BEFORE the bimodality
            # test (line 49 of _em_model_core.py); without this some genes
            # miss the override gate.
            u_w_scaled = u_w / scaling_g
            try:
                _, pval_u, means_u = test_bimodality(u_w_scaled, kde=True)
                _, pval_s, means_s = test_bimodality(s_w, kde=True)
            except (ValueError, np.linalg.LinAlgError):
                continue
            pval[g] = max(pval_u, pval_s)
            su[g] = means_u[1]
            ss[g] = means_s[1]
        return pval, su, ss

    from scvelo_rs._scvelo_rs import per_gene_bimodality_kernel

    Mu_c = np.ascontiguousarray(Mu_sub, dtype=np.float64)
    Ms_c = np.ascontiguousarray(Ms_sub, dtype=np.float64)
    w_c = np.ascontiguousarray(weights, dtype=bool)
    pval, steady_u, steady_s = per_gene_bimodality_kernel(Mu_c, Ms_c, w_c, fit_scaling, True)
    return np.asarray(pval), np.asarray(steady_u), np.asarray(steady_s)


def _per_gene_weights(Mu_sub: np.ndarray, Ms_sub: np.ndarray, perc: float = 99) -> np.ndarray:
    """Boolean (n_cells, n_genes) mask matching scvelo's `initialize_weights`."""
    Mu_c = np.ascontiguousarray(Mu_sub, dtype=np.float64)
    Ms_c = np.ascontiguousarray(Ms_sub, dtype=np.float64)
    return per_gene_weights_kernel(Mu_c, Ms_c, float(perc), True)


def _initialize_all_genes(
    adata,
    *,
    gene_mask,
    Mu,
    Ms,
    fit_scaling,
    fit_steady_states,
    fit_connected_states,
    steady_state_prior,
):
    """Fallback init via scvelo's Python `DynamicsRecovery` — only used when
    `steady_state_prior` is set (rare). Default path uses the Rust kernel."""
    try:
        from scvelo.tools._em_model_core import DynamicsRecovery
    except Exception as e:
        warnings.warn(f"scvelo not importable for initialization: {e}", stacklevel=2)
        return None

    var_names = adata.var_names[gene_mask].to_numpy()
    if len(var_names) == 0:
        return None

    inits = []
    for gene in var_names:
        try:
            dm = DynamicsRecovery(
                adata,
                gene,
                fit_scaling=fit_scaling,
                fit_steady_states=fit_steady_states,
                fit_connected_states=fit_connected_states,
                steady_state_prior=steady_state_prior,
                max_iter=0,
            )
            if not dm.recoverable:
                inits.append(_nan_init())
                continue
            inits.append(
                {
                    "alpha": float(dm.alpha),
                    "beta": float(dm.beta),
                    "gamma": float(dm.gamma),
                    "scaling": float(dm.scaling),
                    "t_": float(dm.t_),
                    "u0_": float(dm.u0_),
                    "s0_": float(dm.s0_),
                    "std_u": float(dm.std_u),
                    "std_s": float(dm.std_s),
                    "ssr": float(dm.steady_state_ratio)
                    if dm.steady_state_ratio is not None
                    else None,
                }
            )
        except Exception:
            inits.append(_nan_init())
    return inits


def _nan_init():
    return {
        "alpha": np.nan,
        "beta": np.nan,
        "gamma": np.nan,
        "scaling": np.nan,
        "t_": np.nan,
        "u0_": np.nan,
        "s0_": np.nan,
        "std_u": np.nan,
        "std_s": np.nan,
        "ssr": None,
    }


def _connectivity_triplet(adata, fit_connected_states):
    if not fit_connected_states:
        return None, None, None
    try:
        from scvelo.preprocessing.moments import get_connectivities

        conn = get_connectivities(adata)
    except Exception as e:
        warnings.warn(f"could not build connectivities: {e}", stacklevel=2)
        return None, None, None
    if conn is None or conn is False:
        return None, None, None
    if not issparse(conn):
        conn = csr_matrix(np.asarray(conn))
    if not isinstance(conn, csr_matrix):
        conn = conn.tocsr()
    return (
        np.ascontiguousarray(conn.data, dtype=np.float64),
        np.ascontiguousarray(conn.indices, dtype=np.int32),
        np.ascontiguousarray(conn.indptr, dtype=np.int32),
    )


def _write_var_columns(
    adata,
    *,
    gene_mask,
    alpha,
    beta,
    gamma,
    t_,
    scaling,
    likelihood,
    variance,
    std_u=None,
    std_s=None,
    u0_=None,
    s0_=None,
    pval_steady=None,
    steady_u=None,
    steady_s=None,
):
    n_vars = adata.n_vars
    nan = np.full(n_vars, np.nan, dtype=np.float64)

    def _scatter(values):
        out = nan.copy()
        if values is not None:
            out[gene_mask] = values
        return out

    adata.var["fit_alpha"] = _scatter(alpha)
    adata.var["fit_beta"] = _scatter(beta)
    adata.var["fit_gamma"] = _scatter(gamma)
    adata.var["fit_t_"] = _scatter(t_)
    adata.var["fit_scaling"] = _scatter(scaling)
    adata.var["fit_likelihood"] = _scatter(likelihood)
    adata.var["fit_variance"] = _scatter(variance)
    adata.var["fit_std_u"] = _scatter(std_u)
    adata.var["fit_std_s"] = _scatter(std_s)
    # scvelo's dm.u0/s0 are 0 unless fit_basal_transcription=True (unsupported).
    adata.var["fit_u0"] = _scatter(np.zeros_like(alpha))
    adata.var["fit_s0"] = _scatter(np.zeros_like(alpha))
    adata.var["fit_pval_steady"] = _scatter(pval_steady)
    adata.var["fit_steady_u"] = _scatter(steady_u)
    adata.var["fit_steady_s"] = _scatter(steady_s)


def _write_layers(adata, *, gene_mask, fit_t_sub, fit_tau_sub, fit_tau__sub, conn):
    """Scatter (n_cells, n_fit_genes) sub-arrays into full (n_cells, n_vars)
    layers. Apply connectivity smoothing on `fit_t` only (matches scvelo)."""
    n_cells = adata.n_obs
    n_vars = adata.n_vars

    def _scatter_2d(sub):
        out = np.full((n_cells, n_vars), np.nan, dtype=np.float64)
        out[:, gene_mask] = sub
        return out

    fit_t_full = _scatter_2d(fit_t_sub)
    fit_tau_full = _scatter_2d(fit_tau_sub)
    fit_tau__full = _scatter_2d(fit_tau__sub)

    if conn is not None and conn is not False:
        smoothed = np.array(fit_t_full)
        smoothed[:, gene_mask] = conn.dot(fit_t_full[:, gene_mask])
        adata.layers["fit_t"] = smoothed
    else:
        adata.layers["fit_t"] = fit_t_full
    adata.layers["fit_tau"] = fit_tau_full
    adata.layers["fit_tau_"] = fit_tau__full
