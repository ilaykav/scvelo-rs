use crate::csr::CsrView;
use crate::divergence::{assign_timepoints, AssignmentMode};
use crate::dynamics::{splicing_solution_scalar, vectorize_per_cell};

/// Per-cell `distx = udiff^2 + sdiff^2 + reg^2` over the weighted subset.
#[allow(clippy::too_many_arguments)]
pub fn get_distx_full(
    u_scaled: &[f64], // self.u / scaling (full n_cells)
    s_full: &[f64],   // self.s (full n_cells)
    weights: &[bool], // self.weights (full n_cells)
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    std_u: f64, // self.std_u (un-divided)
    std_s: f64,
    fit_steady_states: bool,
    connectivities: Option<CsrView<'_>>,
    assignment_mode: AssignmentMode,
    steady_state_ratio: Option<f64>,
    out_distx: &mut Vec<f64>,
) {
    let n = u_scaled.len();
    debug_assert_eq!(s_full.len(), n);
    debug_assert_eq!(weights.len(), n);

    // (1) Full assign_timepoints - yields t, tau, o for ALL n cells.
    //     Mirrors get_time_assignment(refit_time=True) before the weights mask.
    let assign = assign_timepoints(
        u_scaled,
        s_full,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u,
        std_s,
        fit_steady_states,
        /*constraint_time_increments=*/ true,
        connectivities,
        assignment_mode,
    );

    // (2) Weight-mask the per-cell arrays.
    out_distx.clear();
    out_distx.reserve_exact(weights.iter().filter(|&&b| b).count());

    // (3) For each weighted cell: vectorize → get_solution → udiff, sdiff, reg → distx.
    //     udiff matches scvelo: `(ut - u_scaled) / std_u * scaling` where u_scaled
    //     is `self.u / scaling` and std_u is the un-divided self.std_u.
    let reg_coef = match steady_state_ratio {
        Some(ssr) => gamma / beta - ssr,
        None => 0.0,
    };
    let has_reg = steady_state_ratio.is_some();

    for i in 0..n {
        if !weights[i] {
            continue;
        }
        let t_i = assign.t[i];
        let (tau_i, alpha_i, u0_i, s0_i) =
            vectorize_per_cell(t_i, t_, alpha, beta, gamma, 0.0, 0.0);
        let (ut_i, st_i) = splicing_solution_scalar(tau_i, alpha_i, beta, gamma, u0_i, s0_i);

        let udiff = (ut_i - u_scaled[i]) / std_u * scaling;
        let sdiff = (st_i - s_full[i]) / std_s;
        let mut distx = udiff * udiff + sdiff * sdiff;
        if has_reg {
            let reg = reg_coef * s_full[i] / std_s;
            distx += reg * reg;
        }
        out_distx.push(distx);
    }
}

/// Cached-time variant: skips compute_divergence entirely, uses caller-supplied
#[allow(clippy::too_many_arguments)]
pub fn get_distx_cached_time(
    u_scaled: &[f64],
    s_full: &[f64],
    weights: &[bool],
    t_full: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    std_u: f64,
    std_s: f64,
    steady_state_ratio: Option<f64>,
    out_distx: &mut Vec<f64>,
) {
    let n = u_scaled.len();
    debug_assert_eq!(s_full.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(t_full.len(), n);

    out_distx.clear();
    out_distx.reserve_exact(weights.iter().filter(|&&b| b).count());

    let reg_coef = match steady_state_ratio {
        Some(ssr) => gamma / beta - ssr,
        None => 0.0,
    };
    let has_reg = steady_state_ratio.is_some();

    for i in 0..n {
        if !weights[i] {
            continue;
        }
        let t_i = t_full[i];
        let (tau_i, alpha_i, u0_i, s0_i) =
            vectorize_per_cell(t_i, t_, alpha, beta, gamma, 0.0, 0.0);
        let (ut_i, st_i) = splicing_solution_scalar(tau_i, alpha_i, beta, gamma, u0_i, s0_i);

        let udiff = (ut_i - u_scaled[i]) / std_u * scaling;
        let sdiff = (st_i - s_full[i]) / std_s;
        let mut distx = udiff * udiff + sdiff * sdiff;
        if has_reg {
            let reg = reg_coef * s_full[i] / std_s;
            distx += reg * reg;
        }
        out_distx.push(distx);
    }
}
