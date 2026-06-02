#[inline]
pub fn invert_scalar(x: f64) -> f64 {
    // 1/x with the convention that 0 maps to 0 (no-op rather than infinity).
    if x == 0.0 {
        0.0
    } else {
        1.0 / x
    }
}

/// Element-wise log clipped to `[eps, 1-eps]` with `eps = 1e-6`.
#[inline]
pub fn clipped_log_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let lb = 1e-6_f64;
    let ub = 1.0 - 1e-6_f64;
    let clipped = if x < lb {
        lb
    } else if x > ub {
        ub
    } else {
        x
    };
    clipped.ln()
}

/// Closed-form unspliced solution `u(tau, u0, alpha, beta)`.
#[inline]
pub fn unspliced(tau: f64, u0: f64, alpha: f64, beta: f64) -> f64 {
    let expu = (-beta * tau).exp();
    u0 * expu + alpha / beta * (1.0 - expu)
}

/// Closed-form spliced solution `s(tau, s0, u0, alpha, beta, gamma)`.
#[inline]
pub fn spliced(tau: f64, s0: f64, u0: f64, alpha: f64, beta: f64, gamma: f64) -> f64 {
    // c uses reciprocal-then-multiply (two roundings) - required for ULP-level
    // match with the reference Python `(alpha - u0*beta) * invert(gamma - beta)`.
    let c = (alpha - u0 * beta) * invert_scalar(gamma - beta);
    let expu = (-beta * tau).exp();
    let exps = (-gamma * tau).exp();
    s0 * exps + alpha / gamma * (1.0 - exps) + c * (exps - expu)
}

/// Closed-form ODE solution at a single tau. Returns `(u, s)`.
#[inline]
pub fn splicing_solution_scalar(
    tau: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
) -> (f64, f64) {
    let expu = (-beta * tau).exp();
    let exps = (-gamma * tau).exp();
    let u = u0 * expu + alpha / beta * (1.0 - expu);
    let c = (alpha - u0 * beta) * invert_scalar(gamma - beta);
    let s = s0 * exps + alpha / gamma * (1.0 - exps) + c * (exps - expu);
    (u, s)
}

/// Vectorized `SplicingDynamics.get_solution` for an array of tau with
/// scalar parameters. Writes into out_u, out_s.
#[allow(clippy::too_many_arguments)]
pub fn splicing_solution_array(
    tau: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
    out_u: &mut [f64],
    out_s: &mut [f64],
) {
    debug_assert_eq!(tau.len(), out_u.len());
    debug_assert_eq!(tau.len(), out_s.len());
    let aob = alpha / beta;
    let aog = alpha / gamma;
    let c = (alpha - u0 * beta) * invert_scalar(gamma - beta);
    for i in 0..tau.len() {
        let t = tau[i];
        let expu = (-beta * t).exp();
        let exps = (-gamma * t).exp();
        out_u[i] = u0 * expu + aob * (1.0 - expu);
        out_s[i] = s0 * exps + aog * (1.0 - exps) + c * (exps - expu);
    }
}

/// Inverse mapping `tau(u, s, u0, s0, alpha, beta, gamma)` per cell. When
/// `s_provided` is false, only the u-only branch is used.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn tau_inv_scalar(
    u: f64,
    s: f64,
    u0: f64,
    s0: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    s_provided: bool,
) -> f64 {
    tau_inv_scalar_dtyped(u, s, u0, s0, alpha, beta, gamma, s_provided, false)
}

/// Full f32-emulating tau_inv: round-trips every per-cell intermediate through f32
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn tau_inv_scalar_f32_emu(
    u: f64,
    s: f64,
    u0: f64,
    s0: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    s_provided: bool,
) -> f64 {
    let u_f = u as f32;
    let s_f = s as f32;
    let u0_f = u0 as f32;
    let s0_f = s0 as f32;
    let alpha_f = alpha as f32;
    let beta_f = beta as f32;
    let gamma_f = gamma as f32;
    let lb_f = 1e-6_f32;
    let ub_f = 1.0_f32 - 1e-6_f32;

    let res_f = if gamma_f >= beta_f || !s_provided {
        // u-only branch: -1/beta * clipped_log((u - uinf)/(u0 - uinf))
        let uinf = alpha_f / beta_f;
        let num = u_f - uinf;
        let den = u0_f - uinf;
        let ratio = num / den;
        let clipped = if ratio.is_nan() {
            f32::NAN
        } else if ratio < lb_f {
            lb_f
        } else if ratio > ub_f {
            ub_f
        } else {
            ratio
        };
        let log_c = clipped.ln();
        -1.0_f32 / beta_f * log_c
    } else {
        // (u, s) branch: beta_ = beta * invert(gmb), all in f32.
        let gmb = gamma_f - beta_f;
        let inv_gmb = if gmb == 0.0 { 0.0_f32 } else { 1.0_f32 / gmb };
        let beta__f = beta_f * inv_gmb;
        let xinf = alpha_f / gamma_f - beta__f * (alpha_f / beta_f);
        let num = s_f - beta__f * u_f - xinf;
        let den = s0_f - beta__f * u0_f - xinf;
        let ratio = num / den;
        let clipped = if ratio.is_nan() {
            f32::NAN
        } else if ratio < lb_f {
            lb_f
        } else if ratio > ub_f {
            ub_f
        } else {
            ratio
        };
        let log_c = clipped.ln();
        -1.0_f32 / gamma_f * log_c
    };
    res_f as f64
}

/// f32-precision splicing solution: every intermediate truncated to f32.
#[inline]
pub fn splicing_solution_scalar_f32_emu(
    tau: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
) -> (f64, f64) {
    let tau_f = tau as f32;
    let alpha_f = alpha as f32;
    let beta_f = beta as f32;
    let gamma_f = gamma as f32;
    let u0_f = u0 as f32;
    let s0_f = s0 as f32;

    let expu = (-beta_f * tau_f).exp();
    let exps = (-gamma_f * tau_f).exp();
    let u = u0_f * expu + alpha_f / beta_f * (1.0_f32 - expu);
    let gmb = gamma_f - beta_f;
    let inv_gmb = if gmb == 0.0 { 0.0_f32 } else { 1.0_f32 / gmb };
    let c = (alpha_f - u0_f * beta_f) * inv_gmb;
    let s = s0_f * exps + alpha_f / gamma_f * (1.0_f32 - exps) + c * (exps - expu);

    (u as f64, s as f64)
}

/// f32-emulating splicing_solution_array - vectorised over tau with scalar params.
#[allow(clippy::too_many_arguments)]
pub fn splicing_solution_array_f32_emu(
    tau: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
    out_u: &mut [f64],
    out_s: &mut [f64],
) {
    debug_assert_eq!(tau.len(), out_u.len());
    debug_assert_eq!(tau.len(), out_s.len());
    let alpha_f = alpha as f32;
    let beta_f = beta as f32;
    let gamma_f = gamma as f32;
    let u0_f = u0 as f32;
    let s0_f = s0 as f32;
    let aob = alpha_f / beta_f;
    let aog = alpha_f / gamma_f;
    let gmb = gamma_f - beta_f;
    let inv_gmb = if gmb == 0.0 { 0.0_f32 } else { 1.0_f32 / gmb };
    let c = (alpha_f - u0_f * beta_f) * inv_gmb;
    for i in 0..tau.len() {
        let t_f = tau[i] as f32;
        let expu = (-beta_f * t_f).exp();
        let exps = (-gamma_f * t_f).exp();
        out_u[i] = (u0_f * expu + aob * (1.0_f32 - expu)) as f64;
        out_s[i] = (s0_f * exps + aog * (1.0_f32 - exps) + c * (exps - expu)) as f64;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn tau_inv_scalar_dtyped(
    u: f64,
    s: f64,
    u0: f64,
    s0: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    s_provided: bool,
    param_dtype_f32: bool,
) -> f64 {
    if gamma >= beta || !s_provided {
        let uinf = alpha / beta;
        let num = u - uinf;
        let den = u0 - uinf;
        let ratio = num / den;
        let log_clipped = clipped_log_scalar(ratio);
        if param_dtype_f32 {
            let coef_f32 = -1.0_f32 / (beta as f32);
            (coef_f32 as f64) * log_clipped
        } else {
            -1.0 / beta * log_clipped
        }
    } else {
        let gmb = gamma - beta;
        let beta_ = beta * invert_scalar(gmb);
        let xinf = alpha / gamma - beta_ * (alpha / beta);
        let num = s - beta_ * u - xinf;
        let den = s0 - beta_ * u0 - xinf;
        let ratio = num / den;
        let log_clipped = clipped_log_scalar(ratio);
        if param_dtype_f32 {
            let coef_f32 = -1.0_f32 / (gamma as f32);
            (coef_f32 as f64) * log_clipped
        } else {
            -1.0 / gamma * log_clipped
        }
    }
}

/// f32-precision `vectorize_per_cell`: every intermediate truncated to f32.
#[inline]
pub fn vectorize_per_cell_f32_emu(
    t: f64,
    t_: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
) -> (f64, f64, f64, f64) {
    let t_f = t as f32;
    let t__f = t_ as f32;
    let alpha_f = alpha as f32;
    let beta_f = beta as f32;
    let gamma_f = gamma as f32;
    let u0_f = u0 as f32;
    let s0_f = s0 as f32;

    let on: f32 = if t_f < t__f { 1.0 } else { 0.0 };
    let off = 1.0_f32 - on;
    let tau = t_f * on + (t_f - t__f) * off;

    // unspliced(t_, u0, alpha, beta) in f32
    let expu = (-beta_f * t__f).exp();
    let u0_switch = u0_f * expu + alpha_f / beta_f * (1.0_f32 - expu);

    // spliced(t_, s0, u0, alpha, beta, gamma) in f32
    let exps = (-gamma_f * t__f).exp();
    let gmb = gamma_f - beta_f;
    let inv_gmb = if gmb == 0.0 { 0.0_f32 } else { 1.0_f32 / gmb };
    let c = (alpha_f - u0_f * beta_f) * inv_gmb;
    let s0_switch = s0_f * exps + alpha_f / gamma_f * (1.0_f32 - exps) + c * (exps - expu);

    let u0_eff = u0_f * on + u0_switch * off;
    let s0_eff = s0_f * on + s0_switch * off;
    let alpha_eff = alpha_f * on;

    (tau as f64, alpha_eff as f64, u0_eff as f64, s0_eff as f64)
}

/// Per-cell `vectorize`: returns (tau, alpha_eff, u0_eff, s0_eff) for a cell
/// at time `t` given the switch time `t_` and ODE parameters.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn vectorize_per_cell(
    t: f64,
    t_: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
) -> (f64, f64, f64, f64) {
    // o = (t < t_) -> 1 (induction), else 0 (repression)
    let o = if t < t_ { 1.0 } else { 0.0 };
    let tau = t * o + (t - t_) * (1.0 - o);

    // Switch-point state: u0_, s0_ at t_ from u0, s0 with full alpha
    let u0_switch = unspliced(t_, u0, alpha, beta);
    let s0_switch = spliced(t_, s0, u0, alpha, beta, gamma);

    let u0_eff = u0 * o + u0_switch * (1.0 - o);
    let s0_eff = s0 * o + s0_switch * (1.0 - o);
    let alpha_eff = alpha * o; // alpha_ defaults to 0
    (tau, alpha_eff, u0_eff, s0_eff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unspliced_at_zero_returns_u0() {
        let u = unspliced(0.0, 1.5, 2.0, 1.0);
        assert!((u - 1.5).abs() < 1e-12);
    }

    #[test]
    fn spliced_at_zero_returns_s0() {
        let s = spliced(0.0, 0.7, 1.5, 2.0, 1.0, 0.5);
        assert!((s - 0.7).abs() < 1e-12);
    }

    #[test]
    fn clipped_log_inside_interval() {
        // Inside (eps, 1-eps) it's just ln(x).
        let x = 0.5_f64;
        assert!((clipped_log_scalar(x) - x.ln()).abs() < 1e-15);
    }

    #[test]
    fn clipped_log_clips_both_bounds() {
        // Below lower bound clips to eps.
        let lo = clipped_log_scalar(-1.0);
        assert!((lo - 1e-6_f64.ln()).abs() < 1e-15);
        // Above upper bound clips to 1 - eps.
        let hi = clipped_log_scalar(2.0);
        assert!((hi - (1.0 - 1e-6_f64).ln()).abs() < 1e-15);
    }

    #[test]
    fn invert_zero_returns_zero() {
        assert_eq!(invert_scalar(0.0), 0.0);
        assert!((invert_scalar(2.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn tau_inv_u_only_branch_when_gamma_ge_beta() {
        // u-only branch: tau = -1/beta * ln_clip((u - uinf)/(u0 - uinf))
        let u = 0.5;
        let u0 = 1.0;
        let alpha = 2.0;
        let beta = 1.0;
        let gamma = 1.5; // gamma >= beta -> u-only
        let tau = tau_inv_scalar(u, 0.3, u0, 0.0, alpha, beta, gamma, true);
        let uinf = alpha / beta;
        let expected = -1.0 / beta * ((u - uinf) / (u0 - uinf)).clamp(1e-6, 1.0 - 1e-6).ln();
        assert!((tau - expected).abs() < 1e-15);
    }

    #[test]
    fn vectorize_induction_matches_t() {
        let (tau, a, u0, s0) = vectorize_per_cell(0.5, 1.0, 2.0, 1.0, 1.5, 0.0, 0.0);
        assert!((tau - 0.5).abs() < 1e-15);
        assert!((a - 2.0).abs() < 1e-15);
        assert_eq!(u0, 0.0);
        assert_eq!(s0, 0.0);
    }

    #[test]
    fn vectorize_repression_uses_switch_point() {
        let t = 1.5;
        let t_ = 1.0;
        let alpha = 2.0;
        let beta = 1.0;
        let gamma = 1.5;
        let (tau, a, u0_eff, s0_eff) = vectorize_per_cell(t, t_, alpha, beta, gamma, 0.0, 0.0);
        assert!((tau - (t - t_)).abs() < 1e-15);
        assert_eq!(a, 0.0);
        // Repression starts from (u0, s0) at t_
        let u_t = unspliced(t_, 0.0, alpha, beta);
        let s_t = spliced(t_, 0.0, 0.0, alpha, beta, gamma);
        assert!((u0_eff - u_t).abs() < 1e-15);
        assert!((s0_eff - s_t).abs() < 1e-15);
    }
}
