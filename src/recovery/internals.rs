use crate::csr::CsrView;
use crate::divergence::AssignmentMode;
use crate::dynamics::{
    splicing_solution_scalar, splicing_solution_scalar_f32_emu, vectorize_per_cell,
    vectorize_per_cell_f32_emu,
};
use crate::nelder_mead::{self, Settings as NMSettings};
use crate::numpy_compat::pairwise_sum;

use super::types::{f32q, State};
use super::TRACE_ALT_T_ENABLED;

#[inline]
pub(super) fn compute_mse(
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    state: &State,
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    u_scaled_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
) -> f64 {
    let mut t: Vec<f64> = Vec::new();
    let mut tau: Vec<f64> = Vec::new();
    let mut tau_: Vec<f64> = Vec::new();
    let mut o: Vec<u8> = Vec::new();
    compute_mse_with_assign(
        u_orig,
        s_full,
        weights,
        state,
        conn,
        fit_steady_states,
        assignment_mode,
        u_scaled_scratch,
        distx_scratch,
        &mut t,
        &mut tau,
        &mut tau_,
        &mut o,
    )
}

#[inline]
pub(super) fn compute_mse_cached_t(
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    state: &State,
    u_scaled_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
) -> f64 {
    let n = u_orig.len();
    if state.cached_t.len() != n {
        return f64::INFINITY;
    }
    if u_scaled_scratch.len() != n {
        u_scaled_scratch.clear();
        u_scaled_scratch.resize(n, 0.0);
    }
    let scaling = state.scaling;
    for i in 0..n {
        u_scaled_scratch[i] = u_orig[i] / scaling;
    }
    crate::mse::get_distx_cached_time(
        u_scaled_scratch,
        s_full,
        weights,
        &state.cached_t,
        state.alpha,
        state.beta,
        state.gamma,
        state.scaling,
        state.t_,
        state.std_u,
        state.std_s,
        state.steady_state_ratio,
        distx_scratch,
    );
    if distx_scratch.is_empty() {
        return f64::INFINITY;
    }
    pairwise_sum(distx_scratch) / distx_scratch.len() as f64
}

// u0_/s0_ are the ODE switch-point values at t=t_ from origin (0,0), recomputed
// from current (alpha, beta, gamma, t_) - NOT the data-derived asymptote stored
// at init time. Using the asymptote here was the source of ~25% direct-call
// drift before the fix.
#[inline]
pub(super) fn compute_mse_with_assign(
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    state: &State,
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    u_scaled_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
    out_t: &mut Vec<f64>,
    out_tau: &mut Vec<f64>,
    out_tau_: &mut Vec<f64>,
    out_o: &mut Vec<u8>,
) -> f64 {
    let n = u_orig.len();
    if u_scaled_scratch.len() != n {
        u_scaled_scratch.clear();
        u_scaled_scratch.resize(n, 0.0);
    }
    if state.f32_mode {
        let inv_scaling = 1.0_f32 / (state.scaling as f32);
        for i in 0..n {
            u_scaled_scratch[i] = ((u_orig[i] as f32) * inv_scaling) as f64;
        }
    } else {
        // Direct division (`u / scaling`). `u * (1/scaling)` differs at the last
        // ULP and that ULP propagates through `ut`, `distx`, `alt_loss`, flipping
        // the strict `<` comparison in alt_t_.
        let scaling = state.scaling;
        for i in 0..n {
            u_scaled_scratch[i] = u_orig[i] / scaling;
        }
    }

    let (u0_eff, s0_eff) =
        splicing_solution_scalar(state.t_, state.alpha, state.beta, state.gamma, 0.0, 0.0);

    let assign = crate::divergence::assign_timepoints_dtyped(
        u_scaled_scratch,
        s_full,
        state.alpha,
        state.beta,
        state.gamma,
        state.scaling,
        state.t_,
        u0_eff,
        s0_eff,
        state.std_u,
        state.std_s,
        fit_steady_states,
        true,
        conn,
        assignment_mode,
        state.f32_mode,
    );

    distx_scratch.clear();
    let reg_coef = match state.steady_state_ratio {
        Some(ssr) => state.gamma / state.beta - ssr,
        None => 0.0,
    };
    let has_reg = state.steady_state_ratio.is_some();

    if state.f32_mode {
        let std_u_f = state.std_u as f32;
        let std_s_f = state.std_s as f32;
        let scaling_f = state.scaling as f32;
        let reg_coef_f = reg_coef as f32;
        for i in 0..n {
            if !weights[i] {
                continue;
            }
            let t_i = assign.t[i];
            let (tau_i, alpha_i, u0_i, s0_i) = vectorize_per_cell_f32_emu(
                t_i,
                state.t_,
                state.alpha,
                state.beta,
                state.gamma,
                0.0,
                0.0,
            );
            let (ut_i, st_i) = splicing_solution_scalar_f32_emu(
                tau_i,
                alpha_i,
                state.beta,
                state.gamma,
                u0_i,
                s0_i,
            );
            let ut_f = ut_i as f32;
            let st_f = st_i as f32;
            let u_f = u_scaled_scratch[i] as f32;
            let s_f = s_full[i] as f32;

            let udiff = (ut_f - u_f) / std_u_f * scaling_f;
            let sdiff = (st_f - s_f) / std_s_f;
            let mut distx_f = udiff * udiff + sdiff * sdiff;
            if has_reg {
                let reg = reg_coef_f * s_f / std_s_f;
                distx_f += reg * reg;
            }
            distx_scratch.push(distx_f as f64);
        }
    } else {
        for i in 0..n {
            if !weights[i] {
                continue;
            }
            let t_i = assign.t[i];
            let (tau_i, alpha_i, u0_i, s0_i) = vectorize_per_cell(
                t_i,
                state.t_,
                state.alpha,
                state.beta,
                state.gamma,
                0.0,
                0.0,
            );
            let (ut_i, st_i) =
                splicing_solution_scalar(tau_i, alpha_i, state.beta, state.gamma, u0_i, s0_i);

            let udiff = (ut_i - u_scaled_scratch[i]) / state.std_u * state.scaling;
            let sdiff = (st_i - s_full[i]) / state.std_s;
            let mut distx = udiff * udiff + sdiff * sdiff;
            if has_reg {
                let reg = reg_coef * s_full[i] / state.std_s;
                distx += reg * reg;
            }
            distx_scratch.push(distx);
        }
    }

    *out_t = assign.t;
    *out_tau = assign.tau;
    *out_tau_ = assign.tau_;
    *out_o = assign.o;

    if distx_scratch.is_empty() {
        return f64::INFINITY;
    }
    let mean = pairwise_sum(distx_scratch) / distx_scratch.len() as f64;
    if state.f32_mode {
        f32q(mean)
    } else {
        mean
    }
}

pub(super) fn run_stage(
    state: &mut State,
    x0: Vec<f64>,
    apply_x_to_state: impl Fn(&mut State, &[f64]),
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    nm_cfg: &NMSettings,
    u_scaled_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
    adjust_t_: bool,
    refit_time: bool,
) {
    let initial_loss = compute_mse(
        u_orig,
        s_full,
        weights,
        state,
        conn,
        fit_steady_states,
        assignment_mode,
        u_scaled_scratch,
        distx_scratch,
    );
    if state.last_loss.is_nan() || initial_loss < state.last_loss {
        state.last_loss = initial_loss;
    }

    let state_template = state.clone();
    let res_x: Vec<f64>;
    {
        let mut f = |x: &[f64]| -> f64 {
            let mut s = state_template.clone();
            apply_x_to_state(&mut s, x);
            if refit_time {
                compute_mse(
                    u_orig,
                    s_full,
                    weights,
                    &s,
                    conn,
                    fit_steady_states,
                    assignment_mode,
                    u_scaled_scratch,
                    distx_scratch,
                )
            } else {
                compute_mse_cached_t(u_orig, s_full, weights, &s, u_scaled_scratch, distx_scratch)
            }
        };
        let cb = |_x: &[f64], _fx: f64| {};
        let res = nelder_mead::minimize(&mut f, &x0, nm_cfg, cb);
        res_x = res.x;
    }

    let _ = try_update(
        state,
        |c| apply_x_to_state(c, &res_x),
        u_orig,
        s_full,
        weights,
        conn,
        fit_steady_states,
        assignment_mode,
        u_scaled_scratch,
        distx_scratch,
        adjust_t_,
    );
}

// Try to commit a candidate (built by `apply_x_to_cand`) onto `state`. Loss
// must improve over the previous accepted loss. When `adjust_t_`, the alt_t_
// saddle escape can rescue a non-improving candidate by moving t_ to the
// max-on-cell time.
pub(super) fn try_update(
    state: &mut State,
    apply_x_to_cand: impl FnOnce(&mut State),
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    u_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
    adjust_t_: bool,
) -> bool {
    let n = u_orig.len();
    let loss_prev = if state.last_loss.is_nan() {
        1e6
    } else {
        state.last_loss
    };
    let trace_alt_t = TRACE_ALT_T_ENABLED.with(|c| c.get());

    let mut cand = state.clone();
    apply_x_to_cand(&mut cand);

    let mut cand_t: Vec<f64> = Vec::new();
    let mut cand_tau: Vec<f64> = Vec::new();
    let mut cand_tau_: Vec<f64> = Vec::new();
    let mut cand_o: Vec<u8> = Vec::new();
    let cand_loss = compute_mse_with_assign(
        u_orig,
        s_full,
        weights,
        &cand,
        conn,
        fit_steady_states,
        assignment_mode,
        u_scratch,
        distx_scratch,
        &mut cand_t,
        &mut cand_tau,
        &mut cand_tau_,
        &mut cand_o,
    );
    let mut perform_update = cand_loss < loss_prev;
    if trace_alt_t {
        let cached_on = state.cached_o.iter().filter(|&&x| x == 1).count();
        eprintln!(
            "[alt_t entry] alpha={:.10e} beta={:.10e} gamma={:.10e} scaling={:.10e} t_={:.10e} | cand alpha={:.10e} beta={:.10e} gamma={:.10e} scaling={:.10e} t_={:.10e} | cand_loss={:.10e} loss_prev={:.10e} perform_update={} cached_on={} adjust_t_={}",
            state.alpha, state.beta, state.gamma, state.scaling, state.t_,
            cand.alpha, cand.beta, cand.gamma, cand.scaling, cand.t_,
            cand_loss, loss_prev, perform_update, cached_on, adjust_t_,
        );
    }

    let (work_alpha, work_beta, work_gamma, work_scaling);
    let mut work_t_;
    let mut work_t: Vec<f64>;
    let mut work_tau: Vec<f64>;
    let mut work_tau_: Vec<f64>;
    let mut work_o: Vec<u8>;
    let mut work_loss: f64;
    if perform_update {
        work_alpha = cand.alpha;
        work_beta = cand.beta;
        work_gamma = cand.gamma;
        work_scaling = cand.scaling;
        work_t_ = cand.t_;
        work_t = cand_t;
        work_tau = cand_tau;
        work_tau_ = cand_tau_;
        work_o = cand_o;
        work_loss = cand_loss;
    } else {
        work_alpha = state.alpha;
        work_beta = state.beta;
        work_gamma = state.gamma;
        work_scaling = state.scaling;
        work_t_ = state.t_;
        work_t = Vec::new();
        work_tau = Vec::new();
        work_tau_ = Vec::new();
        work_o = Vec::new();
        work_loss = compute_mse_with_assign(
            u_orig,
            s_full,
            weights,
            state,
            conn,
            fit_steady_states,
            assignment_mode,
            u_scratch,
            distx_scratch,
            &mut work_t,
            &mut work_tau,
            &mut work_tau_,
            &mut work_o,
        );
    }

    if adjust_t_ {
        let any_on = state.cached_o.contains(&1);
        if any_on && !state.cached_o.is_empty() && state.cached_o.len() == work_t.len() {
            let mut alt_t_cand = f64::NEG_INFINITY;
            for i in 0..n {
                if state.cached_o[i] == 1 && work_t[i] > alt_t_cand {
                    alt_t_cand = work_t[i];
                }
            }
            if trace_alt_t {
                let cur_on = work_o.iter().filter(|&&x| x == 1).count();
                let work_t_max = work_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "[alt_t check] work_alpha={:.10e} work_beta={:.10e} work_gamma={:.10e} work_scaling={:.10e} work_t_={:.10e} | alt_t_raw={:.10e} work_t.max={:.10e} cached_on={} cur_work_on={} gate_pass={}",
                    work_alpha, work_beta, work_gamma, work_scaling, work_t_,
                    alt_t_cand, work_t_max,
                    state.cached_o.iter().filter(|&&x| x == 1).count(),
                    cur_on,
                    alt_t_cand > 0.0 && alt_t_cand < work_t_,
                );
            }
            if alt_t_cand > 0.0 && alt_t_cand < work_t_ {
                let max_t = work_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let n_at_switch = work_t.iter().filter(|&&v| v == work_t_).count();
                alt_t_cand += max_t / (n as f64) * (n_at_switch as f64);

                let alt_state = State {
                    alpha: work_alpha,
                    beta: work_beta,
                    gamma: work_gamma,
                    scaling: work_scaling,
                    t_: alt_t_cand,
                    u0_: state.u0_,
                    s0_: state.s0_,
                    std_u: state.std_u,
                    std_s: state.std_s,
                    steady_state_ratio: state.steady_state_ratio,
                    last_loss: state.last_loss,
                    cached_t: vec![],
                    cached_tau: vec![],
                    cached_tau_: vec![],
                    cached_o: vec![],
                    f32_mode: state.f32_mode,
                };
                let mut alt_t_v: Vec<f64> = Vec::new();
                let mut alt_tau_v: Vec<f64> = Vec::new();
                let mut alt_tau__v: Vec<f64> = Vec::new();
                let mut alt_o_v: Vec<u8> = Vec::new();
                let alt_loss = compute_mse_with_assign(
                    u_orig,
                    s_full,
                    weights,
                    &alt_state,
                    conn,
                    fit_steady_states,
                    assignment_mode,
                    u_scratch,
                    distx_scratch,
                    &mut alt_t_v,
                    &mut alt_tau_v,
                    &mut alt_tau__v,
                    &mut alt_o_v,
                );

                let ut_cur = crate::dynamics::unspliced(work_t_, 0.0, work_alpha, work_beta);
                let ut_alt = crate::dynamics::unspliced(alt_t_cand, 0.0, work_alpha, work_beta);
                let min_loss = work_loss.min(loss_prev);

                if alt_loss * 0.99 <= min_loss || ut_cur * 0.99 < ut_alt {
                    work_t_ = alt_t_cand;
                    work_loss = alt_loss;
                    work_t = alt_t_v;
                    work_tau = alt_tau_v;
                    work_tau_ = alt_tau__v;
                    work_o = alt_o_v;
                    perform_update = true;
                }
            }
        }
    }

    if perform_update {
        // Rescale u0_ in proportion to the scaling change to keep it consistent
        // with the new scaling factor.
        if work_scaling != state.scaling && state.scaling != 0.0 {
            let factor = state.scaling / work_scaling;
            state.u0_ *= factor;
        }
        state.alpha = work_alpha;
        state.beta = work_beta;
        state.gamma = work_gamma;
        state.scaling = work_scaling;
        state.t_ = work_t_;
        state.last_loss = work_loss;
        state.cached_t = work_t;
        state.cached_tau = work_tau;
        state.cached_tau_ = work_tau_;
        state.cached_o = work_o;
    }
    perform_update
}

// Best-of-5 perturbation around the initial alpha before the first NM stage:
// alpha + linspace(-1, 1, 5) * alpha / 10, each tested via try_update.
pub(super) fn pre_perturb_alpha(
    state: &mut State,
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    u_scaled_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
) {
    let val = state.alpha;
    if !val.is_finite() || val == 0.0 {
        return;
    }
    let fracs: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

    if state.last_loss.is_nan() {
        let mut new_t: Vec<f64> = Vec::new();
        let mut new_tau: Vec<f64> = Vec::new();
        let mut new_tau_: Vec<f64> = Vec::new();
        let mut new_o: Vec<u8> = Vec::new();
        let l0 = compute_mse_with_assign(
            u_orig,
            s_full,
            weights,
            state,
            conn,
            fit_steady_states,
            assignment_mode,
            u_scaled_scratch,
            distx_scratch,
            &mut new_t,
            &mut new_tau,
            &mut new_tau_,
            &mut new_o,
        );
        state.last_loss = l0;
        state.cached_t = new_t;
        state.cached_tau = new_tau;
        state.cached_tau_ = new_tau_;
        state.cached_o = new_o;
    }

    for f in fracs.iter() {
        let candidate = val + f * val / 10.0;
        try_update(
            state,
            |c| {
                c.alpha = candidate;
            },
            u_orig,
            s_full,
            weights,
            conn,
            fit_steady_states,
            assignment_mode,
            u_scaled_scratch,
            distx_scratch,
            true,
        );
    }
}

pub(super) fn initialize_scaling(
    state: &mut State,
    sight: f64,
    u_orig: &[f64],
    s_full: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    fit_steady_states: bool,
    assignment_mode: AssignmentMode,
    u_scratch: &mut Vec<f64>,
    distx_scratch: &mut Vec<f64>,
) {
    if state.scaling == 0.0 || !state.scaling.is_finite() {
        return;
    }
    let scaling_snapshot = state.scaling;
    let fracs: [f64; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];

    for f in fracs.iter() {
        let z = scaling_snapshot + f * scaling_snapshot * sight;
        let beta_new = state.beta / state.scaling * z;
        try_update(
            state,
            |c| {
                c.scaling = z;
                c.beta = beta_new;
            },
            u_orig,
            s_full,
            weights,
            conn,
            fit_steady_states,
            assignment_mode,
            u_scratch,
            distx_scratch,
            true,
        );
    }
}
