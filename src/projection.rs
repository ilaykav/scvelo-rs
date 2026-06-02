use ndarray::ArrayView1;
use rayon::prelude::*;

/// Closed-form unspliced solution: u(tau) under (alpha, beta) starting from u0.
#[inline]
pub fn unspliced(tau: f64, u0: f64, alpha: f64, beta: f64) -> f64 {
    let expu = (-beta * tau).exp();
    u0 * expu + (alpha / beta) * (1.0 - expu)
}

/// Closed-form spliced solution: s(tau) under (alpha, beta, gamma) starting from (u0, s0).
#[inline]
pub fn spliced(tau: f64, s0: f64, u0: f64, alpha: f64, beta: f64, gamma: f64) -> f64 {
    let expu = (-beta * tau).exp();
    let exps = (-gamma * tau).exp();
    let c = (alpha - u0 * beta) / (gamma - beta);
    s0 * exps + (alpha / gamma) * (1.0 - exps) + c * (exps - expu)
}

/// Sample the (u, s) curve at `num` evenly spaced time points in [0, t_end].
pub fn sample_curve(
    t_end: f64,
    num: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
    out_u: &mut [f64],
    out_s: &mut [f64],
    out_t: &mut [f64],
) {
    debug_assert_eq!(out_u.len(), num);
    debug_assert_eq!(out_s.len(), num);
    debug_assert_eq!(out_t.len(), num);

    let dt = if num > 1 {
        t_end / (num - 1) as f64
    } else {
        0.0
    };
    for k in 0..num {
        let t = (k as f64) * dt;
        out_t[k] = t;
        out_u[k] = unspliced(t, u0, alpha, beta);
        out_s[k] = spliced(t, s0, u0, alpha, beta, gamma);
    }
    // numpy's `linspace(0, t_end, num)` explicitly sets `y[-1] = stop` to make
    // the endpoint bit-exact equal to t_end. Without this, `(num-1) * dt` can
    // round 1 ULP above t_end for some t_end values, and cells projecting to
    // that endpoint then get classified `tau = t_i - t_ = 1 ULP` (off-state)
    // instead of `tau = 0`, which makes `exp(-beta*tau) ≠ 1.0` (it's 1 - 2e-16),
    // propagating a 1-ULP difference into ut/distx and breaking NM tie-breaks
    // on the boundary. Match numpy here.
    if num > 1 {
        out_t[num - 1] = t_end;
        out_u[num - 1] = unspliced(t_end, u0, alpha, beta);
        out_s[num - 1] = spliced(t_end, s0, u0, alpha, beta, gamma);
    }
}

/// Find tau per cell: nearest sampled curve point in (u, s)-space.
pub fn project_to_curve_serial(
    u: ArrayView1<f64>,
    s: ArrayView1<f64>,
    xt_u: &[f64],
    xt_s: &[f64],
    tpoints: &[f64],
    tau: &mut [f64],
) {
    let n_cells = u.len();
    debug_assert_eq!(s.len(), n_cells);
    debug_assert_eq!(tau.len(), n_cells);
    let num = xt_u.len();

    for i in 0..n_cells {
        let u_i = u[i];
        let s_i = s[i];
        let mut best_d = f64::INFINITY;
        let mut best_k = 0usize;
        for k in 0..num {
            let du = u_i - xt_u[k];
            let ds = s_i - xt_s[k];
            let d = du * du + ds * ds;
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        tau[i] = tpoints[best_k];
    }
}

/// Thread-parallel version. Cells are independent; Rayon splits the cell range
/// across the global thread pool with shared `&[f64]` views over the curve.
pub fn project_to_curve_parallel(
    u: ArrayView1<f64>,
    s: ArrayView1<f64>,
    xt_u: &[f64],
    xt_s: &[f64],
    tpoints: &[f64],
    tau: &mut [f64],
) {
    let n_cells = u.len();
    debug_assert_eq!(s.len(), n_cells);
    debug_assert_eq!(tau.len(), n_cells);
    let num = xt_u.len();

    tau.par_iter_mut().enumerate().for_each(|(i, out)| {
        let u_i = u[i];
        let s_i = s[i];
        let mut best_d = f64::INFINITY;
        let mut best_k = 0usize;
        for k in 0..num {
            let du = u_i - xt_u[k];
            let ds = s_i - xt_s[k];
            let d = du * du + ds * ds;
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        *out = tpoints[best_k];
    });
}

/// Vectorised forward ODE eval over an array of time points.
pub fn splicing_dynamics_eval(
    t: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
    out_u: &mut [f64],
    out_s: &mut [f64],
    parallel: bool,
) {
    debug_assert_eq!(t.len(), out_u.len());
    debug_assert_eq!(t.len(), out_s.len());

    // gamma == beta is degenerate; bump by tiny epsilon to keep `c` finite.
    let g_minus_b = gamma - beta;
    let safe_gmb = if g_minus_b.abs() < 1e-300 {
        1e-300
    } else {
        g_minus_b
    };
    let c = (alpha - u0 * beta) / safe_gmb;
    let alpha_over_beta = alpha / beta;
    let alpha_over_gamma = alpha / gamma;

    if parallel && t.len() >= 4096 {
        out_u
            .par_iter_mut()
            .zip(out_s.par_iter_mut())
            .zip(t.par_iter())
            .for_each(|((ou, os), &ti)| {
                let expu = (-beta * ti).exp();
                let exps = (-gamma * ti).exp();
                *ou = u0 * expu + alpha_over_beta * (1.0 - expu);
                *os = s0 * exps + alpha_over_gamma * (1.0 - exps) + c * (exps - expu);
            });
    } else {
        for i in 0..t.len() {
            let ti = t[i];
            let expu = (-beta * ti).exp();
            let exps = (-gamma * ti).exp();
            out_u[i] = u0 * expu + alpha_over_beta * (1.0 - expu);
            out_s[i] = s0 * exps + alpha_over_gamma * (1.0 - exps) + c * (exps - expu);
        }
    }
}

/// Per-cell evaluation of the splicing dynamics for a single gene with
/// fitted (alpha, beta, gamma, t_, scaling) and additive offsets
/// (u0_offset, s0_offset). Mirrors `scvelo.utils.compute_dynamics`:
/// for each `t_i`, switches between induction (`t_i < t_`) and repression
/// (`t_i >= t_`) phases via the standard scvelo vectorize logic, then
/// evaluates the closed-form solution.
///
/// `t` is taken as-is. Sorting (when requested by the Python caller) must
/// happen before invoking this function.
#[allow(clippy::too_many_arguments)]
pub fn compute_dynamics_eval(
    t: &[f64],
    t_: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    u0_offset: f64,
    s0_offset: f64,
    out_alpha: &mut [f64],
    out_u: &mut [f64],
    out_s: &mut [f64],
    parallel: bool,
) {
    debug_assert_eq!(t.len(), out_alpha.len());
    debug_assert_eq!(t.len(), out_u.len());
    debug_assert_eq!(t.len(), out_s.len());

    let g_minus_b = gamma - beta;
    // Sign-preserving guard: gamma == beta is degenerate; bump by tiny epsilon
    // keeping the sign so the resulting `c = (alpha - u0*beta) / safe_gmb`
    // does not silently flip when gamma < beta + |eps|.
    let safe_gmb = if g_minus_b.abs() < 1e-300 {
        if g_minus_b < 0.0 {
            -1e-300
        } else {
            1e-300
        }
    } else {
        g_minus_b
    };
    let alpha_over_beta = alpha / beta;
    let alpha_over_gamma = alpha / gamma;

    // State at the switching point t_ (induction with u0=s0=0):
    //   u0_switch = unspliced(t_, 0, alpha, beta)
    //   s0_switch = spliced(t_, 0, 0, alpha, beta, gamma)
    let expu_t_ = (-beta * t_).exp();
    let exps_t_ = (-gamma * t_).exp();
    let u0_switch = alpha_over_beta * (1.0 - expu_t_);
    let c_switch = alpha / safe_gmb; // (alpha - 0*beta) / (gamma-beta) when u0=0
    let s0_switch = alpha_over_gamma * (1.0 - exps_t_) + c_switch * (exps_t_ - expu_t_);

    let body = |ti: f64| -> (f64, f64, f64) {
        let induction = ti < t_;
        let tau = if induction { ti } else { ti - t_ };
        let (u0_eff, s0_eff, alpha_eff) = if induction {
            (0.0, 0.0, alpha)
        } else {
            (u0_switch, s0_switch, 0.0)
        };

        let expu = (-beta * tau).exp();
        let exps = (-gamma * tau).exp();
        let alpha_eff_over_beta = if induction { alpha_over_beta } else { 0.0 };
        let alpha_eff_over_gamma = if induction { alpha_over_gamma } else { 0.0 };
        let c = (alpha_eff - u0_eff * beta) / safe_gmb;

        let u_raw = u0_eff * expu + alpha_eff_over_beta * (1.0 - expu);
        let s_raw = s0_eff * exps + alpha_eff_over_gamma * (1.0 - exps) + c * (exps - expu);
        (alpha_eff, u_raw * scaling + u0_offset, s_raw + s0_offset)
    };

    if parallel && t.len() >= 4096 {
        out_alpha
            .par_iter_mut()
            .zip(out_u.par_iter_mut())
            .zip(out_s.par_iter_mut())
            .zip(t.par_iter())
            .for_each(|(((oa, ou), os), &ti)| {
                let (a, u, s) = body(ti);
                *oa = a;
                *ou = u;
                *os = s;
            });
    } else {
        for i in 0..t.len() {
            let (a, u, s) = body(t[i]);
            out_alpha[i] = a;
            out_u[i] = u;
            out_s[i] = s;
        }
    }
}

/// Closed-form `tau` from `u` only (no spliced data).
#[inline]
pub fn tau_inv_u_scalar(u: f64, u0: f64, alpha: f64, beta: f64) -> f64 {
    let uinf = alpha / beta;
    let num = u - uinf;
    let den = u0 - uinf;
    let den_inv = if den != 0.0 { 1.0 / den } else { 0.0 };
    let ratio = num * den_inv;
    let lb_eps = 1e-6_f64;
    let ub_eps = 1.0 - 1e-6_f64;
    let clipped = ratio.clamp(lb_eps, ub_eps);
    -1.0 / beta * clipped.ln()
}

/// Full per-cell tau assignment via curve projection (induction + repression).
#[allow(clippy::too_many_arguments)]
pub fn assign_tau_full(
    u: &[f64],
    s: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    num: usize,
    tau: &mut [f64],
    tau_: &mut [f64],
    parallel: bool,
) {
    let n_cells = u.len();
    debug_assert_eq!(s.len(), n_cells);
    debug_assert_eq!(tau.len(), n_cells);
    debug_assert_eq!(tau_.len(), n_cells);

    // Repression-phase end: t0 = tau_inv at min(u[s > 0]) under (u0_, alpha=0, beta).
    let mut min_u_s_pos = f64::INFINITY;
    for i in 0..n_cells {
        if s[i] > 0.0 && u[i] < min_u_s_pos {
            min_u_s_pos = u[i];
        }
    }
    let t0 = if min_u_s_pos.is_finite() {
        tau_inv_u_scalar(min_u_s_pos, u0_, 0.0, beta)
    } else {
        t_
    };

    // Induction curve: alpha, beta, gamma, u0=0, s0=0, sampled in [0, t_].
    let mut xt_u = vec![0.0f64; num];
    let mut xt_s = vec![0.0f64; num];
    let mut tpoints = vec![0.0f64; num];
    sample_curve(
        t_,
        num,
        alpha,
        beta,
        gamma,
        0.0,
        0.0,
        &mut xt_u,
        &mut xt_s,
        &mut tpoints,
    );

    // Repression curve: alpha=0, beta, gamma, u0=u0_, s0=s0_, sampled in (0, t0]
    // (first grid point dropped → effective num-1 samples).
    let num_ = num.saturating_sub(1).max(1);
    let mut xt_u_ = vec![0.0f64; num_];
    let mut xt_s_ = vec![0.0f64; num_];
    let mut tpoints_ = vec![0.0f64; num_];
    // linspace(0, t0, num=num)[1:] - first grid point dropped.
    let dt = if num > 1 { t0 / (num - 1) as f64 } else { 0.0 };
    for k in 0..num_ {
        let t = ((k + 1) as f64) * dt;
        tpoints_[k] = t;
        xt_u_[k] = unspliced(t, u0_, 0.0, beta);
        xt_s_[k] = spliced(t, s0_, u0_, 0.0, beta, gamma);
    }
    // numpy's linspace endpoint fix - same bit-exact reasoning as `sample_curve`.
    // tpoints_'s LAST sample (k = num_ - 1, originally index num-1 of full linspace)
    // should equal t0 exactly. Without this, projections to the endpoint get a
    // 1-ULP-off tau and propagate the same NM-trajectory drift.
    if num_ >= 1 {
        tpoints_[num_ - 1] = t0;
        xt_u_[num_ - 1] = unspliced(t0, u0_, 0.0, beta);
        xt_s_[num_ - 1] = spliced(t0, s0_, u0_, 0.0, beta, gamma);
    }

    // Project both curves.
    let u_view = ndarray::ArrayView1::from(u);
    let s_view = ndarray::ArrayView1::from(s);

    if parallel {
        project_to_curve_parallel(u_view, s_view, &xt_u, &xt_s, &tpoints, tau);
        project_to_curve_parallel(u_view, s_view, &xt_u_, &xt_s_, &tpoints_, tau_);
    } else {
        project_to_curve_serial(u_view, s_view, &xt_u, &xt_s, &tpoints, tau);
        project_to_curve_serial(u_view, s_view, &xt_u_, &xt_s_, &tpoints_, tau_);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn closed_form_at_zero() {
        // At tau=0 the closed forms should return (u0, s0).
        let u = unspliced(0.0, 1.5, 2.0, 1.0);
        let s = spliced(0.0, 0.7, 1.5, 2.0, 1.0, 0.5);
        assert!((u - 1.5).abs() < 1e-12);
        assert!((s - 0.7).abs() < 1e-12);
    }

    #[test]
    fn project_picks_nearest_point() {
        let xt_u = vec![0.0, 1.0, 2.0, 3.0];
        let xt_s = vec![0.0, 1.0, 2.0, 3.0];
        let tpoints = vec![0.0, 0.5, 1.0, 1.5];
        let u = Array1::from_vec(vec![0.05, 1.95, 2.95]);
        let s = Array1::from_vec(vec![0.05, 1.95, 2.95]);
        let mut tau = vec![0.0f64; 3];

        project_to_curve_serial(u.view(), s.view(), &xt_u, &xt_s, &tpoints, &mut tau);

        assert_eq!(tau[0], 0.0); // nearest (0, 0)
        assert_eq!(tau[1], 1.0); // nearest (2, 2)
        assert_eq!(tau[2], 1.5); // nearest (3, 3)
    }
}
