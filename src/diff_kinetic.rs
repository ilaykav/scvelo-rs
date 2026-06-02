//! Bit-exact port of scvelo's `DynamicsRecovery.differential_kinetic_test`.
//!
//! Per fitted gene (Rayon-parallel) the kernel mirrors scvelo's
//! `_em_model_core.differential_kinetic_test` outer loop:
//!
//! 1. Caller provides per-gene `(alpha, beta_internal = fit_beta * fit_scaling,
//!    gamma, scaling, t_)` and raw `Mu`, `Ms` (no scaling applied) plus the
//!    connectivities CSR. `u0_, s0_` are reconstructed from the ODE so they
//!    match scvelo's `load_pars` (`SplicingDynamics(alpha, beta_internal,
//!    gamma).get_solution(t_)`).
//! 2. `initialize_weights(weighted=False)`: weights = (u>0)&(s>0),
//!    `std_u/std_s` recomputed on raw u, s; `weights_upper` from max u_w/3,
//!    max s_w/3. Matches scvelo line 793-818 (`weighted=False` branch).
//! 3. Single call to `assign_timepoints(...)` gives the post-adjust-increments
//!    `t, tau, tau_, o, tau_unmasked, tau__unmasked` matching
//!    `compute_divergence(mode='assign_timepoints')`. `dm.get_dists` (with
//!    `refit_time=True`, the default after load_pars) goes through this
//!    re-assignment, not the cached load_pars t.
//! 4. `outside_of_trajectory` mask: recompute `ut, ut_` at
//!    `tau_unmasked, tau__unmasked` (pre-final-mask), then `sign(distu) *
//!    sign(distu_) == 1`. Matches scvelo's `compute_divergence(mode=
//!    'outside_of_trajectory')` early-return at line 324.
//! 5. Per-cell `distx[i]`: branch-select via the post-assign `o[i]`. When
//!    `o==1` use `tau[i]` with `(alpha, [0,0])`; when `o==0` use `tau_[i]`
//!    with `(0, [u0_, s0_])`. `udiff = (ut - u_scaled) / std_u_raw * scaling`,
//!    `sdiff = (st - s) / std_s`. (Matches scvelo's `get_dists` line 1050-1056.)
//! 6. `varx = get_variance(weighted='upper')`: `mean(distx[upper]) -
//!    mean(sign(sdiff[upper]) * sqrt(distx[upper]))^2`.
//! 7. LRT outer mode: per-cluster MSE under `weights_outer` with min_cells
//!    trim. Worst cluster → `orth_beta = (b + sqrt(b^2 + 4a^2)) / (2a)` on
//!    `weights & cluster_mask` (weighted=False override; see scvelo
//!    `get_orth_fit` line 1652-1658) → initial pval.
//! 8. Fallback to `'upper'` mode if pval > 1e-2: recompute worst on
//!    `weights_upper`, orth_beta on `weights_upper & cluster_mask` (the
//!    default `weighted=True` path).
//! 9. Per-cluster pvals: `(distx_sum/2 - orth_distx_sum) / (varx * sqrt(8*n_c))`
//!    then `norm.sf`.

use rayon::prelude::*;
use std::f64::consts::SQRT_2;

use crate::csr::CsrView;
use crate::divergence::{assign_timepoints, AssignmentMode};
use crate::dynamics::{splicing_solution_array, splicing_solution_scalar};
use crate::numpy_compat::{pairwise_sum, percentile_sorted};

/// Default percentile used by scvelo's `initialize_weights(weighted=True)` for
/// clipping high u/s values (set by `BaseDynamics.__init__` perc=99).
const INIT_WEIGHTS_PERC: f64 = 99.0;

#[inline]
fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / SQRT_2)
}

/// numpy's `np.sign`: returns 0.0 for x == 0.0 (and signed-zero variants).
/// `f64::signum` returns 1.0 / -1.0 for ±0.0 - DIFFERENT semantics, which
/// flips `sign(0) * sign(x)` from 0 to ±1 and skews the outside_of_trajectory
/// mask plus the varx `mean(sign * sqrt(distx))` term.
#[inline]
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `np.std(arr)` (ddof=0).
fn std_pop(arr: &[f64]) -> f64 {
    let n = arr.len();
    if n == 0 {
        return 0.0;
    }
    let m = pairwise_sum(arr) / n as f64;
    let mut sq: Vec<f64> = Vec::with_capacity(n);
    for &x in arr {
        let d = x - m;
        sq.push(d * d);
    }
    (pairwise_sum(&sq) / n as f64).sqrt()
}

fn argmax(arr: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in arr.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Closed-form orth_beta on a masked subset (matches scvelo `get_orth_fit`).
///
/// When the masked subset is empty, `a = b = 0` and the formula evaluates to
/// `(0 + sqrt(0))/0 = NaN` - scvelo lets numpy propagate this NaN through the
/// downstream pval compute (which is what we want for genes/clusters with no
/// nonzero u/s cells). Do NOT short-circuit to a finite default here.
fn orth_beta_for(u_scaled: &[f64], s: &[f64], mask: &[bool]) -> f64 {
    let n = u_scaled.len();
    let mut su_buf: Vec<f64> = Vec::new();
    let mut uu_ss_buf: Vec<f64> = Vec::new();
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        su_buf.push(s[i] * u_scaled[i]);
        uu_ss_buf.push(u_scaled[i] * u_scaled[i] - s[i] * s[i]);
    }
    let a = pairwise_sum(&su_buf);
    let b = pairwise_sum(&uu_ss_buf);
    (b + (b * b + 4.0 * a * a).sqrt()) / (2.0 * a)
}

#[allow(clippy::too_many_arguments)]
fn per_cluster_pvals(
    distx: &[f64],
    u_scaled: &[f64],
    s: &[f64],
    mask: &[bool],
    cluster_assign: &[i32],
    n_clusters: usize,
    min_cells: usize,
    orth_beta: f64,
    std_u_raw: f64,
    std_s: f64,
    scaling: f64,
    varx: f64,
) -> Vec<f64> {
    let n = distx.len();
    let one_plus_b2 = 1.0 + orth_beta * orth_beta;
    // Per-cluster total size (used by scvelo's `get_pval_diff_kinetics` min_cells
    // early-out at line 1710 - it tests `cluster_mask.sum() < min_cells`,
    // NOT the (mode_mask & cluster_mask) subset size).
    let mut cluster_total = vec![0usize; n_clusters];
    for &c in cluster_assign {
        if c >= 0 && (c as usize) < n_clusters {
            cluster_total[c as usize] += 1;
        }
    }
    let mut pvals = vec![1.0f64; n_clusters];
    for c in 0..n_clusters {
        if cluster_total[c] < min_cells {
            // scvelo line 1710: small clusters return 1, not NaN.
            pvals[c] = 1.0;
            continue;
        }
        let mut dx_buf: Vec<f64> = Vec::new();
        let mut ox_buf: Vec<f64> = Vec::new();
        for i in 0..n {
            if !mask[i] {
                continue;
            }
            if cluster_assign[i] as usize != c {
                continue;
            }
            dx_buf.push(distx[i] * 0.5);
            let s_real = (s[i] + orth_beta * u_scaled[i]) / one_plus_b2;
            let sdiff_o = (s_real - s[i]) / std_s;
            let udiff_o = (orth_beta * s_real - u_scaled[i]) / std_u_raw * scaling;
            ox_buf.push(udiff_o * udiff_o + sdiff_o * sdiff_o);
        }
        // scvelo lets the formula produce NaN when the masked subset is empty
        // (denom = 0 from varx * sqrt(0)); replicate that 0/0 → NaN path.
        let n_masked = dx_buf.len();
        let distx_sum = pairwise_sum(&dx_buf);
        let orth_distx_sum = pairwise_sum(&ox_buf);
        let denom = varx * (8.0 * n_masked as f64).sqrt();
        let stat = (distx_sum - orth_distx_sum) / denom;
        pvals[c] = norm_sf(stat);
    }
    pvals
}

#[allow(clippy::too_many_arguments)]
fn single_cluster_pval(
    distx: &[f64],
    u_scaled: &[f64],
    s: &[f64],
    mode_mask: &[bool],
    cluster_mask: &[bool],
    orth_beta: f64,
    std_u_raw: f64,
    std_s: f64,
    scaling: f64,
    varx: f64,
    min_cells: usize,
) -> f64 {
    let n = distx.len();
    // scvelo's `get_pval_diff_kinetics` line 1710 early-out: cluster total
    // size (the cluster mask itself), NOT the (mode_mask & cluster_mask)
    // subset size.
    let cluster_total: usize = cluster_mask.iter().filter(|&&b| b).count();
    if cluster_total < min_cells {
        return 1.0;
    }
    let one_plus_b2 = 1.0 + orth_beta * orth_beta;
    let mut dx_buf: Vec<f64> = Vec::new();
    let mut ox_buf: Vec<f64> = Vec::new();
    for i in 0..n {
        if !(mode_mask[i] && cluster_mask[i]) {
            continue;
        }
        dx_buf.push(distx[i] * 0.5);
        let s_real = (s[i] + orth_beta * u_scaled[i]) / one_plus_b2;
        let sdiff_o = (s_real - s[i]) / std_s;
        let udiff_o = (orth_beta * s_real - u_scaled[i]) / std_u_raw * scaling;
        ox_buf.push(udiff_o * udiff_o + sdiff_o * sdiff_o);
    }
    let n_masked = dx_buf.len();
    let distx_sum = pairwise_sum(&dx_buf);
    let orth_distx_sum = pairwise_sum(&ox_buf);
    let denom = varx * (8.0 * n_masked as f64).sqrt();
    let stat = (distx_sum - orth_distx_sum) / denom;
    norm_sf(stat)
}

#[allow(clippy::too_many_arguments)]
fn diff_kinetic_test_one_gene(
    u_raw: &[f64],
    s: &[f64],
    alpha: f64,
    beta: f64, // internal beta = fit_beta * fit_scaling
    gamma: f64,
    scaling: f64,
    t_: f64,
    cluster_assign: &[i32],
    n_clusters: usize,
    min_cells: usize,
    connectivities: Option<CsrView<'_>>,
) -> Vec<f64> {
    let n = u_raw.len();

    // (a) u0_, s0_ from ODE at t_ (matches scvelo load_pars).
    let (u0_, s0_) = splicing_solution_scalar(t_, alpha, beta, gamma, 0.0, 0.0);

    // (b) u_scaled = u_raw / scaling.
    let mut u_scaled = vec![0.0f64; n];
    for i in 0..n {
        u_scaled[i] = u_raw[i] / scaling;
    }

    // (c) `initialize_weights(weighted=True, perc=99)` - percentile-clipped.
    //     scvelo computes varx using THIS state, BEFORE
    //     initialize_diff_kinetics overwrites it with weighted=False.
    let mut nonzero = vec![false; n];
    let mut u_nz: Vec<f64> = Vec::new();
    let mut s_nz: Vec<f64> = Vec::new();
    for i in 0..n {
        let nz = u_raw[i] > 0.0 && s[i] > 0.0;
        nonzero[i] = nz;
        if nz {
            u_nz.push(u_raw[i]);
            s_nz.push(s[i]);
        }
    }
    let mut u_nz_sorted = u_nz.clone();
    u_nz_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut s_nz_sorted = s_nz.clone();
    s_nz_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ub_u = percentile_sorted(&u_nz_sorted, INIT_WEIGHTS_PERC);
    let ub_s = percentile_sorted(&s_nz_sorted, INIT_WEIGHTS_PERC);
    let mut weights_clipped = nonzero.clone();
    if ub_s > 0.0 {
        for i in 0..n {
            weights_clipped[i] &= s[i] <= ub_s;
        }
    }
    if ub_u > 0.0 {
        for i in 0..n {
            weights_clipped[i] &= u_raw[i] <= ub_u;
        }
    }
    let mut u_w_clipped: Vec<f64> = Vec::new();
    let mut s_w_clipped: Vec<f64> = Vec::new();
    for i in 0..n {
        if weights_clipped[i] {
            u_w_clipped.push(u_raw[i]);
            s_w_clipped.push(s[i]);
        }
    }
    let std_u_clipped = std_pop(&u_w_clipped);
    let std_s_clipped = std_pop(&s_w_clipped);
    let max_u_clipped = u_w_clipped
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_s_clipped = s_w_clipped
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights_upper_clipped = vec![false; n];
    for i in 0..n {
        weights_upper_clipped[i] =
            weights_clipped[i] && u_raw[i] > max_u_clipped / 3.0 && s[i] > max_s_clipped / 3.0;
    }

    // (d) Pass 1: varx via assign_timepoints with CLIPPED std (matches
    //     `dm.get_variance()` called BEFORE `initialize_weights(weighted=False)`
    //     overwrites the weighted=True state).
    let assign_clipped = assign_timepoints(
        &u_scaled,
        s,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u_clipped,
        std_s_clipped,
        true,
        true,
        connectivities,
        AssignmentMode::None,
    );
    let varx = {
        let mut dx_buf: Vec<f64> = Vec::new();
        let mut sgn_sqrt_buf: Vec<f64> = Vec::new();
        for i in 0..n {
            if !weights_upper_clipped[i] {
                continue;
            }
            let (ut, st) = if assign_clipped.o[i] == 1 {
                splicing_solution_scalar(assign_clipped.tau[i], alpha, beta, gamma, 0.0, 0.0)
            } else {
                splicing_solution_scalar(assign_clipped.tau_[i], 0.0, beta, gamma, u0_, s0_)
            };
            let udiff = (ut - u_scaled[i]) / std_u_clipped * scaling;
            let sdi = (st - s[i]) / std_s_clipped;
            let dxi = udiff * udiff + sdi * sdi;
            dx_buf.push(dxi);
            sgn_sqrt_buf.push(np_sign(sdi) * dxi.sqrt());
        }
        if dx_buf.is_empty() {
            f64::NAN
        } else {
            let n_up = dx_buf.len() as f64;
            let mean_dx = pairwise_sum(&dx_buf) / n_up;
            let mean_sgn = pairwise_sum(&sgn_sqrt_buf) / n_up;
            mean_dx - mean_sgn * mean_sgn
        }
    };

    // (e) `initialize_weights(weighted=False)` - un-clipped state for LRT.
    let mut weights = vec![false; n];
    let mut u_w_buf: Vec<f64> = Vec::new();
    let mut s_w_buf: Vec<f64> = Vec::new();
    for i in 0..n {
        weights[i] = nonzero[i];
        if weights[i] {
            u_w_buf.push(u_raw[i]);
            s_w_buf.push(s[i]);
        }
    }
    let std_u_raw = std_pop(&u_w_buf);
    let std_s = std_pop(&s_w_buf);
    let max_u_w = u_w_buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_s_w = s_w_buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut weights_upper = vec![false; n];
    for i in 0..n {
        weights_upper[i] = weights[i] && u_raw[i] > max_u_w / 3.0 && s[i] > max_s_w / 3.0;
    }

    // (f) Pass 2: assign_timepoints with UN-CLIPPED std.
    let assign = assign_timepoints(
        &u_scaled,
        s,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u_raw,
        std_s,
        true,
        true,
        connectivities,
        AssignmentMode::None,
    );

    // (g) outside_of_trajectory using PASS-2 tau_unmasked / tau__unmasked.
    let std_u_div = std_u_raw / scaling;
    let mut ut_out = vec![0.0f64; n];
    let mut st_out = vec![0.0f64; n];
    let mut ut__out = vec![0.0f64; n];
    let mut st__out = vec![0.0f64; n];
    splicing_solution_array(
        &assign.tau_unmasked,
        alpha,
        beta,
        gamma,
        0.0,
        0.0,
        &mut ut_out,
        &mut st_out,
    );
    splicing_solution_array(
        &assign.tau__unmasked,
        0.0,
        beta,
        gamma,
        u0_,
        s0_,
        &mut ut__out,
        &mut st__out,
    );
    let mut weights_outer = vec![false; n];
    for i in 0..n {
        let distu = (u_scaled[i] - ut_out[i]) / std_u_div;
        let distu_ = (u_scaled[i] - ut__out[i]) / std_u_div;
        let outside = np_sign(distu) * np_sign(distu_) == 1.0;
        weights_outer[i] = weights[i] && outside;
    }

    // (h) Per-cell distx via branch-selected ODE at PASS-2 tau/tau_/o.
    let mut distx = vec![0.0f64; n];
    for i in 0..n {
        let (ut, st) = if assign.o[i] == 1 {
            splicing_solution_scalar(assign.tau[i], alpha, beta, gamma, 0.0, 0.0)
        } else {
            splicing_solution_scalar(assign.tau_[i], 0.0, beta, gamma, u0_, s0_)
        };
        let udiff = (ut - u_scaled[i]) / std_u_raw * scaling;
        let sdi = (st - s[i]) / std_s;
        distx[i] = udiff * udiff + sdi * sdi;
    }

    // (i) Cluster MSE under weights_outer.
    let mut cluster_dx_sum_outer = vec![0.0f64; n_clusters];
    let mut cluster_n_outer = vec![0usize; n_clusters];
    for i in 0..n {
        if !weights_outer[i] {
            continue;
        }
        let c = cluster_assign[i];
        if c < 0 || c as usize >= n_clusters {
            continue;
        }
        cluster_dx_sum_outer[c as usize] += distx[i];
        cluster_n_outer[c as usize] += 1;
    }
    let mut mse_outer = vec![0.0f64; n_clusters];
    for c in 0..n_clusters {
        if cluster_n_outer[c] >= min_cells {
            mse_outer[c] = cluster_dx_sum_outer[c] / cluster_n_outer[c] as f64;
        }
    }

    // (j) Worst cluster + orth_beta (weighted=False override) + initial pval.
    let worst_outer = argmax(&mse_outer);
    let mut cluster_mask = vec![false; n];
    for i in 0..n {
        cluster_mask[i] = cluster_assign[i] >= 0 && (cluster_assign[i] as usize) == worst_outer;
    }
    let mut weights_and_cluster = vec![false; n];
    for i in 0..n {
        weights_and_cluster[i] = weights[i] && cluster_mask[i];
    }
    let mut orth_beta = orth_beta_for(&u_scaled, s, &weights_and_cluster);
    let pval_worst = single_cluster_pval(
        &distx,
        &u_scaled,
        s,
        &weights_outer,
        &cluster_mask,
        orth_beta,
        std_u_raw,
        std_s,
        scaling,
        varx,
        min_cells,
    );

    // (k) Fallback to 'upper' if not significant.
    let mode_mask: Vec<bool>;
    if pval_worst > 1e-2 {
        let mut cluster_dx_sum_upper = vec![0.0f64; n_clusters];
        let mut cluster_n_upper = vec![0usize; n_clusters];
        for i in 0..n {
            if !weights_upper[i] {
                continue;
            }
            let c = cluster_assign[i];
            if c < 0 || c as usize >= n_clusters {
                continue;
            }
            cluster_dx_sum_upper[c as usize] += distx[i];
            cluster_n_upper[c as usize] += 1;
        }
        let mut mse_upper = vec![0.0f64; n_clusters];
        for c in 0..n_clusters {
            if cluster_n_upper[c] >= min_cells {
                mse_upper[c] = cluster_dx_sum_upper[c] / cluster_n_upper[c] as f64;
            }
        }
        let worst_upper = argmax(&mse_upper);
        let mut cluster_mask_upper = vec![false; n];
        let mut weights_and_cluster_upper = vec![false; n];
        for i in 0..n {
            cluster_mask_upper[i] =
                cluster_assign[i] >= 0 && (cluster_assign[i] as usize) == worst_upper;
            // scvelo's `get_orth_fit` hard-overrides `weighted=True`
            // (line 1654: `kwargs["weighted"] = True`). Inside `get_weights`,
            // `weighted=True` (boolean) returns `self.weights` (NOT
            // `self.weights_upper`). So orth_beta is computed on the
            // base `weights & cluster_mask`, regardless of which LRT mode
            // we ended up in.
            weights_and_cluster_upper[i] = weights[i] && cluster_mask_upper[i];
        }
        orth_beta = orth_beta_for(&u_scaled, s, &weights_and_cluster_upper);
        mode_mask = weights_upper;
    } else {
        mode_mask = weights_outer;
    }

    // (l) Per-cluster pvals under chosen mode.
    per_cluster_pvals(
        &distx,
        &u_scaled,
        s,
        &mode_mask,
        cluster_assign,
        n_clusters,
        min_cells,
        orth_beta,
        std_u_raw,
        std_s,
        scaling,
        varx,
    )
}

/// Batched per-gene LRT. Rayon-parallel over genes.
#[allow(clippy::too_many_arguments)]
pub fn diff_kinetic_test_kernel(
    n_cells: usize,
    n_genes: usize,
    u_raw: &[f64],
    s: &[f64],
    alpha: &[f64],
    beta: &[f64],
    gamma: &[f64],
    scaling: &[f64],
    t_: &[f64],
    cluster_assign: &[i32],
    n_clusters: usize,
    min_cells: usize,
    connectivities: Option<CsrView<'_>>,
) -> Vec<f64> {
    let mut out = vec![1.0f64; n_genes * n_clusters];

    let gene_cols: Vec<(Vec<f64>, Vec<f64>)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut u_col = vec![0.0f64; n_cells];
            let mut s_col = vec![0.0f64; n_cells];
            for i in 0..n_cells {
                let idx = i * n_genes + g;
                u_col[i] = u_raw[idx];
                s_col[i] = s[idx];
            }
            (u_col, s_col)
        })
        .collect();

    let results: Vec<Vec<f64>> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let (u_col, s_col) = &gene_cols[g];
            diff_kinetic_test_one_gene(
                u_col,
                s_col,
                alpha[g],
                beta[g],
                gamma[g],
                scaling[g],
                t_[g],
                cluster_assign,
                n_clusters,
                min_cells,
                connectivities,
            )
        })
        .collect();

    for g in 0..n_genes {
        for c in 0..n_clusters {
            out[g * n_clusters + c] = results[g][c];
        }
    }

    out
}
