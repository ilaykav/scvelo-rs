//! Differential-kinetics test (LRT) for per-gene per-cluster splicing kinetics.
//!
//! Per gene (Rayon-parallel):
//! 1. Call `assign_timepoints` to get per-cell `t`, `tau`, `tau_`, `o` with
//!    connectivity smoothing — matching scvelo's `compute_divergence(mode='assign_timepoints')`.
//! 2. Eval per-cell `ut/st` at the BEST per-cell `tau` (induction branch) and
//!    `ut_/st_` at the BEST per-cell `tau_` (repression branch). These are
//!    independently optimized per branch — NOT derived from a single t.
//! 3. Compute per-cell distu/distu_/dists/dists_ scaled by `(* scaling / std_u)`
//!    and `/std_s` respectively.
//! 4. Build the `outside_of_trajectory` mask via `sign(distu) * sign(distu_) == 1`.
//! 5. weights_outer = base_weights & outside_of_trajectory.
//! 6. Per-cell distx = distu²+dists² if o==1 (induction), else distu_²+dists_²
//!    (matches scvelo's `get_dists` after vectorize).
//! 7. Per cluster: aggregate distx_sum, n_cells under weights_outer.
//! 8. Pick worst cluster (highest MSE), compute closed-form orth_beta on
//!    `base_weights & cluster_mask` (NOT weights_outer — matches scvelo's
//!    `get_orth_fit` which overrides weighted=True internally).
//! 9. Per cluster: compute orth_distx_sum on weights_outer cells, compute
//!    LRT p-value via `norm_sf((distx_sum/2 - orth_distx_sum) / (varx * sqrt(8 * cluster_n)))`.
//!
//! The redundancy in scvelo's slow path (recomputing assign_timepoints per
//! cluster despite identical result modulo masking) is removed here — we
//! compute once per gene and slice per cluster. This is mathematically
//! identical to scvelo's loop, just without the wasted work.

use rayon::prelude::*;
use std::f64::consts::SQRT_2;

use crate::csr::CsrView;
use crate::divergence::{assign_timepoints, AssignmentMode};
use crate::dynamics::splicing_solution_scalar;

/// Standard normal survival function via libm::erfc.
#[inline]
fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / SQRT_2)
}

/// Per-gene LRT result.
#[derive(Clone, Debug)]
pub struct DiffKineticResult {
    pub pvals: Vec<f64>,
    pub worst_cluster_idx: i32,
    pub orth_beta: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn diff_kinetic_test_one_gene(
    u_scaled: &[f64],
    s: &[f64],
    weights: &[bool],
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    std_u_orig: f64,
    std_s: f64,
    varx: f64,
    cluster_assign: &[i32],
    n_clusters: usize,
    min_cells: usize,
    fit_steady_states: bool,
    connectivities: Option<CsrView<'_>>,
) -> DiffKineticResult {
    let n = u_scaled.len();
    debug_assert_eq!(s.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(cluster_assign.len(), n);

    // 1. Per-cell t, tau, tau_, o via assign_timepoints (bit-exact match
    //    to scvelo's compute_divergence(mode='assign_timepoints')).
    let assign = assign_timepoints(
        u_scaled,
        s,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u_orig,
        std_s,
        fit_steady_states,
        true, // constraint_time_increments
        connectivities,
        AssignmentMode::None,
    );
    let tau = &assign.tau;
    let tau_ = &assign.tau_;
    let o = &assign.o;

    // scvelo normalizes std_u by scaling inside compute_divergence.
    let std_u = std_u_orig / scaling;

    // 2 + 3. Per-cell ut/st (induction at tau), ut_/st_ (repression at tau_),
    //        plus per-cell distu/distu_/dists/dists_.
    let mut distx = vec![0.0f64; n];
    let mut outside = vec![false; n];

    for i in 0..n {
        // Induction branch at PER-CELL tau (best induction time for cell i)
        let (ut_ind, st_ind) = splicing_solution_scalar(tau[i], alpha, beta, gamma, 0.0, 0.0);
        // Repression branch at PER-CELL tau_ (best repression time for cell i)
        let (ut_rep, st_rep) = splicing_solution_scalar(tau_[i], 0.0, beta, gamma, u0_, s0_);

        let u_i = u_scaled[i];
        let s_i = s[i];
        // scvelo's distu formula: (u - ut) / std_u — after std_u /= scaling
        let distu = (u_i - ut_ind) / std_u;
        let distu_ = (u_i - ut_rep) / std_u;
        let dists = (s_i - st_ind) / std_s;
        let dists_ = (s_i - st_rep) / std_s;

        outside[i] = (distu.signum() * distu_.signum()) == 1.0;

        // distx via branch selection: o[i] == 1 → induction; else repression
        // (this matches what get_dists / vectorize produces in scvelo)
        distx[i] = if o[i] == 1 {
            distu * distu + dists * dists
        } else {
            distu_ * distu_ + dists_ * dists_
        };
    }

    // 5. weights_outer = base_weights & outside_of_trajectory
    let mut weights_outer = vec![false; n];
    for i in 0..n {
        weights_outer[i] = weights[i] && outside[i];
    }

    // 7. Per cluster: distx_sum + n_cells
    let mut cluster_distx_sum = vec![0.0f64; n_clusters];
    let mut cluster_n_cells = vec![0usize; n_clusters];
    for i in 0..n {
        if !weights_outer[i] {
            continue;
        }
        let c = cluster_assign[i];
        if c < 0 || (c as usize) >= n_clusters {
            continue;
        }
        let cu = c as usize;
        cluster_distx_sum[cu] += distx[i];
        cluster_n_cells[cu] += 1;
    }

    // 8a. Pick worst cluster (highest MSE) among clusters with >= min_cells
    let mut worst_idx: i32 = -1;
    let mut worst_mse = f64::NEG_INFINITY;
    for c in 0..n_clusters {
        if cluster_n_cells[c] < min_cells {
            continue;
        }
        let mse = cluster_distx_sum[c] / (cluster_n_cells[c] as f64);
        if mse > worst_mse {
            worst_mse = mse;
            worst_idx = c as i32;
        }
    }

    // 8b. orth_beta on worst-cluster cells under BASE weights (not weights_outer).
    //     scvelo's get_orth_fit overrides kwargs['weighted'] = True (== base
    //     self.weights) regardless of the calling kwargs.
    let mut orth_beta = 1.0f64;
    if worst_idx >= 0 {
        let wc = worst_idx as usize;
        let mut sum_su = 0.0f64;
        let mut sum_uu_minus_ss = 0.0f64;
        for i in 0..n {
            if weights[i] && cluster_assign[i] as usize == wc {
                let u_i = u_scaled[i];
                let s_i = s[i];
                sum_su += s_i * u_i;
                sum_uu_minus_ss += u_i * u_i - s_i * s_i;
            }
        }
        if sum_su.abs() > 0.0 {
            let a = sum_su;
            let b = sum_uu_minus_ss;
            orth_beta = (b + (b * b + 4.0 * a * a).sqrt()) / (2.0 * a);
        }
    }

    // 9. Per cluster: LRT p-value
    //    distx_sum_c   = sum(distx[cell_in_cluster_and_weights_outer]) / 2
    //    orth_distx_sum_c = sum((orth_beta*s_real - u)/std_u_orig * scaling)² + ((s_real - s)/std_s)²
    //                       where s_real = (s + orth_beta*u) / (1 + orth_beta²)
    //    denom_c       = varx * sqrt(8 * cluster_n_cells[c])
    //    pval_c        = norm_sf((distx_sum_c - orth_distx_sum_c) / denom_c)
    let mut pvals = vec![1.0f64; n_clusters];
    let denom_one_plus_beta_sq = 1.0 + orth_beta * orth_beta;
    for c in 0..n_clusters {
        if cluster_n_cells[c] < min_cells {
            pvals[c] = 1.0;
            continue;
        }
        let mut orth_distx_sum_c = 0.0f64;
        let mut distx_sum_c = 0.0f64;
        for i in 0..n {
            if weights_outer[i] && cluster_assign[i] as usize == c {
                let u_i = u_scaled[i];
                let s_i = s[i];
                let s_real = (s_i + orth_beta * u_i) / denom_one_plus_beta_sq;
                let sdiff = (s_real - s_i) / std_s;
                let udiff = (orth_beta * s_real - u_i) / std_u_orig * scaling;
                orth_distx_sum_c += udiff * udiff + sdiff * sdiff;
                distx_sum_c += distx[i];
            }
        }
        let denom_c = varx * (8.0 * (cluster_n_cells[c] as f64)).sqrt();
        let stat = (distx_sum_c / 2.0 - orth_distx_sum_c) / denom_c;
        pvals[c] = norm_sf(stat);
    }

    DiffKineticResult {
        pvals,
        worst_cluster_idx: worst_idx,
        orth_beta,
    }
}

/// Batched per-gene LRT. Rayon-parallel over genes. The connectivity CSR
/// (if provided) is shared by reference across all gene tasks — no copies.
#[allow(clippy::too_many_arguments)]
pub fn diff_kinetic_test_kernel(
    n_cells: usize,
    n_genes: usize,
    u_scaled: &[f64], // (n_cells * n_genes) C-order
    s: &[f64],
    weights: &[bool],
    alpha: &[f64],
    beta: &[f64],
    gamma: &[f64],
    scaling: &[f64],
    t_: &[f64],
    u0_: &[f64],
    s0_: &[f64],
    std_u: &[f64],
    std_s: &[f64],
    varx: &[f64],
    cluster_assign: &[i32],
    n_clusters: usize,
    min_cells: usize,
    fit_steady_states: bool,
    connectivities: Option<CsrView<'_>>,
) -> Vec<f64> {
    let mut pvals_out = vec![1.0f64; n_genes * n_clusters];

    // Materialize per-gene column views first (saves the strided indexing
    // inside the hot per-cell loop).
    let gene_cols: Vec<(Vec<f64>, Vec<f64>, Vec<bool>)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut u_col = vec![0.0f64; n_cells];
            let mut s_col = vec![0.0f64; n_cells];
            let mut w_col = vec![false; n_cells];
            for i in 0..n_cells {
                let idx = i * n_genes + g;
                u_col[i] = u_scaled[idx];
                s_col[i] = s[idx];
                w_col[i] = weights[idx];
            }
            (u_col, s_col, w_col)
        })
        .collect();

    let results: Vec<DiffKineticResult> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let (u_col, s_col, w_col) = &gene_cols[g];
            diff_kinetic_test_one_gene(
                u_col,
                s_col,
                w_col,
                alpha[g],
                beta[g],
                gamma[g],
                scaling[g],
                t_[g],
                u0_[g],
                s0_[g],
                std_u[g],
                std_s[g],
                varx[g],
                cluster_assign,
                n_clusters,
                min_cells,
                fit_steady_states,
                connectivities,
            )
        })
        .collect();

    for g in 0..n_genes {
        for c in 0..n_clusters {
            pvals_out[g * n_clusters + c] = results[g].pvals[c];
        }
    }

    pvals_out
}
