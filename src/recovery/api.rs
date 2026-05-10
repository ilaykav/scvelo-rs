use crate::csr::CsrView;
use crate::divergence::AssignmentMode;
use crate::dynamics::{splicing_solution_scalar, vectorize_per_cell};
use crate::mse::get_distx_full;
use crate::nelder_mead::Settings as NMSettings;
use crate::numpy_compat::pairwise_sum;

use super::internals::{
    compute_mse, compute_mse_with_assign, pre_perturb_alpha, run_stage, try_update,
};
use super::types::{mean_masked, mean_masked_or, std_pop, GeneFitFull, Initial, State};
use super::TRACE_ALT_T_ENABLED;

// Public entry: per-gene init phase. KDE bimodality is computed in Python and
// passed in as `pval_steady`/`steady_u`/`steady_s` (NaN to skip).
pub fn initialize_one_gene(
    u: &[f64],
    s: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    fit_scaling: bool,
    fit_steady_states: bool,
    f32_mode: bool,
    pval_steady: f64,
    steady_u: f64,
    _steady_s: f64,
) -> Initial {
    let n = u.len();
    debug_assert_eq!(s.len(), n);
    debug_assert_eq!(weights.len(), n);

    let n_w = weights.iter().filter(|&&b| b).count();
    if n_w <= 2 {
        return Initial {
            alpha: f64::NAN,
            beta: f64::NAN,
            gamma: f64::NAN,
            scaling: f64::NAN,
            t_: f64::NAN,
            u0_: f64::NAN,
            s0_: f64::NAN,
            std_u: f64::NAN,
            std_s: f64::NAN,
            steady_state_ratio: None,
            f32_mode,
        };
    }

    let mut u_w: Vec<f64> = Vec::with_capacity(n_w);
    let mut s_w: Vec<f64> = Vec::with_capacity(n_w);
    for i in 0..n {
        if weights[i] {
            u_w.push(u[i]);
            s_w.push(s[i]);
        }
    }

    let q32 = |x: f64| -> f64 { (x as f32) as f64 };
    let std_u = if f32_mode {
        q32(std_pop(&u_w))
    } else {
        std_pop(&u_w)
    };
    let std_s = if f32_mode {
        q32(std_pop(&s_w))
    } else {
        std_pop(&s_w)
    };
    let (std_u, std_s) = if std_u == 0.0 || std_s == 0.0 {
        (1.0, 1.0)
    } else {
        (std_u, std_s)
    };

    let scaling = if fit_scaling {
        if f32_mode {
            q32(std_u / std_s)
        } else {
            std_u / std_s
        }
    } else {
        1.0
    };

    let u_w_scaled: Vec<f64> = if f32_mode {
        u_w.iter().map(|x| q32(x / scaling)).collect()
    } else {
        u_w.iter().map(|x| x / scaling).collect()
    };

    let mut s_w_sorted = s_w.clone();
    s_w_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut u_w_sorted = u_w_scaled.clone();
    u_w_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let perc_s = crate::numpy_compat::percentile_sorted(&s_w_sorted, 98.0);
    let perc_u = crate::numpy_compat::percentile_sorted(&u_w_sorted, 98.0);
    let weights_s_mask: Vec<bool> = s_w.iter().map(|&v| v >= perc_s).collect();
    let weights_u_mask: Vec<bool> = u_w_scaled.iter().map(|&v| v >= perc_u).collect();

    // linreg via convolved products + pairwise sum (matches np.sum bit-exactly).
    let weights_g = &weights_s_mask;
    let mut us_arr: Vec<f64> = Vec::with_capacity(n_w);
    let mut ss_arr: Vec<f64> = Vec::with_capacity(n_w);
    for i in 0..n_w {
        if weights_g[i] {
            us_arr.push(u_w_scaled[i] * s_w[i]);
            ss_arr.push(s_w[i] * s_w[i]);
        } else {
            us_arr.push(0.0);
            ss_arr.push(0.0);
        }
    }
    let us_sum = crate::numpy_compat::pairwise_sum(&us_arr);
    let ss_sum = crate::numpy_compat::pairwise_sum(&ss_arr);
    let mut gamma = if f32_mode {
        q32(q32(us_sum / ss_sum) + 1e-6_f64)
    } else {
        us_sum / ss_sum + 1e-6
    };
    if f32_mode {
        if gamma < q32(0.05 / scaling) {
            gamma = q32(gamma * 1.2);
        } else if gamma > q32(1.5 / scaling) {
            gamma = q32(gamma / 1.2);
        }
    } else if gamma < 0.05 / scaling {
        gamma *= 1.2;
    } else if gamma > 1.5 / scaling {
        gamma /= 1.2;
    }
    let mut beta: f64 = 1.0;

    let mut u_inf = if f32_mode {
        q32(mean_masked_or(
            &u_w_scaled,
            &weights_u_mask,
            &weights_s_mask,
        ))
    } else {
        mean_masked_or(&u_w_scaled, &weights_u_mask, &weights_s_mask)
    };
    let s_inf = if f32_mode {
        q32(mean_masked(&s_w, &weights_s_mask))
    } else {
        mean_masked(&s_w, &weights_s_mask)
    };
    let mut u0_ = u_inf;
    let mut s0_ = s_inf;
    let mut alpha = u_inf * beta;

    // KDE bimodality override: when steady-state mode is detected, push alpha/beta
    // toward a different kinetic regime where steady_u/steady_s have non-trivial means.
    if pval_steady.is_finite() && pval_steady < 1e-3 {
        u_inf = (u_inf + steady_u) * 0.5;
        alpha = gamma * s_inf;
        beta = alpha / u_inf;
        u0_ = u_inf;
        s0_ = s_inf;
    }

    let t_ = if f32_mode {
        crate::dynamics::tau_inv_scalar_f32_emu(u0_, s0_, 0.0, 0.0, alpha, beta, gamma, true)
    } else {
        crate::dynamics::tau_inv_scalar(u0_, s0_, 0.0, 0.0, alpha, beta, gamma, true)
    };
    if std::env::var("SCVELORS_TRACE_INIT_T").is_ok() {
        eprintln!(
            "[init-t-trace] line96 t_={t_:.10e} alpha={alpha:.10e} beta={beta:.10e} \
             gamma={gamma:.10e} u0_={u0_:.10e} s0_={s0_:.10e}"
        );
    }

    // Init values are computed in (optionally) f32 to match scvelo's default-layer
    // behaviour, but the EM compute path runs in f64 (`f32_mode: false` below).
    let mut state = State {
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u,
        std_s,
        steady_state_ratio: None,
        last_loss: f64::NAN,
        cached_t: vec![0.0; n],
        cached_tau: vec![0.0; n],
        cached_tau_: vec![0.0; n],
        cached_o: vec![0u8; n],
        f32_mode: false,
    };

    let mut u_scaled_scratch: Vec<f64> = Vec::with_capacity(n);
    let mut distx_scratch: Vec<f64> = Vec::with_capacity(n);

    // Cache t/tau/o + loss before initialize_scaling so its alt_t_ branch can fire.
    {
        let mut new_t: Vec<f64> = Vec::new();
        let mut new_tau: Vec<f64> = Vec::new();
        let mut new_tau_: Vec<f64> = Vec::new();
        let mut new_o: Vec<u8> = Vec::new();
        let l0 = compute_mse_with_assign(
            u,
            s,
            weights,
            &state,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &mut u_scaled_scratch,
            &mut distx_scratch,
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

    if fit_scaling {
        for sight in [0.5_f64, 0.1] {
            let scaling_snap = state.scaling;
            for f_frac in [-1.0_f64, -1.0 / 3.0, 1.0 / 3.0, 1.0] {
                let z = scaling_snap + f_frac * scaling_snap * sight;
                let beta_z = state.beta / state.scaling * z;
                try_update(
                    &mut state,
                    |c| {
                        c.scaling = z;
                        c.beta = beta_z;
                    },
                    u,
                    s,
                    weights,
                    conn,
                    fit_steady_states,
                    AssignmentMode::None,
                    &mut u_scaled_scratch,
                    &mut distx_scratch,
                    true,
                );
            }
        }
    }

    let ssr = state.gamma / state.beta;

    Initial {
        alpha: state.alpha,
        beta: state.beta,
        gamma: state.gamma,
        scaling: state.scaling,
        t_: state.t_,
        u0_: state.u0_,
        s0_: state.s0_,
        std_u: state.std_u,
        std_s: state.std_s,
        steady_state_ratio: Some(ssr),
        f32_mode,
    }
}

// Public entry: full per-gene EM driver. 5 NM stages, then a projection-mode refit,
// then a final no-arg update that gives alt_t_ one last chance to fire.
pub fn fit_one_gene(
    u: &[f64],
    s: &[f64],
    weights: &[bool],
    conn: Option<CsrView<'_>>,
    init: Initial,
    max_iter: usize,
    fit_scaling: bool,
    fit_steady_states: bool,
    gene_idx: usize,
) -> GeneFitFull {
    let n = u.len();
    debug_assert_eq!(s.len(), n);
    debug_assert_eq!(weights.len(), n);

    let dbg = match std::env::var("SCVELORS_DEBUG_GENE") {
        Ok(v) => v
            .parse::<usize>()
            .ok()
            .map(|t| t == gene_idx)
            .unwrap_or(false),
        Err(_) => false,
    };
    let trace_alt_t = dbg && std::env::var("SCVELORS_TRACE_ALT_T").is_ok();
    TRACE_ALT_T_ENABLED.with(|c| c.set(trace_alt_t));
    struct TraceGuard;
    impl Drop for TraceGuard {
        fn drop(&mut self) {
            TRACE_ALT_T_ENABLED.with(|c| c.set(false));
        }
    }
    let _trace_guard = TraceGuard;
    let dbg_print = |stage: &str, st: &State| {
        if dbg {
            eprintln!(
                "[trace gene={gene_idx} stage={stage}] alpha={:.6e} beta={:.6e} gamma={:.6e} t_={:.6e} scaling={:.6e} u0_={:.6e} s0_={:.6e} loss={:.6e}",
                st.alpha, st.beta, st.gamma, st.t_, st.scaling, st.u0_, st.s0_, st.last_loss,
            );
        }
    };

    if max_iter == 0 || init.alpha.is_nan() || init.beta.is_nan() || init.gamma.is_nan() {
        return GeneFitFull::nan(n);
    }

    let mut state = State::from_initial(&init, n);
    let mut u_scaled_scratch: Vec<f64> = Vec::with_capacity(n);
    let mut distx_scratch: Vec<f64> = Vec::with_capacity(n);
    if dbg {
        let l0 = compute_mse(
            u,
            s,
            weights,
            &state,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &mut u_scaled_scratch,
            &mut distx_scratch,
        );
        eprintln!("[trace gene={gene_idx} stage=initial] alpha={:.6e} beta={:.6e} gamma={:.6e} t_={:.6e} scaling={:.6e} u0_={:.6e} s0_={:.6e} std_u={:.6e} std_s={:.6e} ssr={:?} f32={} initial_loss={:.6e}",
            state.alpha, state.beta, state.gamma, state.t_, state.scaling,
            state.u0_, state.s0_, state.std_u, state.std_s, state.steady_state_ratio, state.f32_mode, l0);
    }

    // Stages 1, 2, 4 use tol=1e-4; stages 3, 5, 6 use tol=1e-2.
    let nm_default = NMSettings::scvelo_default(4, max_iter);
    let nm_loose = NMSettings::scvelo_with_tol(4, max_iter, 1e-2);

    pre_perturb_alpha(
        &mut state,
        u,
        s,
        weights,
        conn,
        fit_steady_states,
        AssignmentMode::None,
        &mut u_scaled_scratch,
        &mut distx_scratch,
    );
    dbg_print("post_pre_perturb", &state);

    // Stage 1: (t_, alpha)
    {
        let x0 = vec![state.t_, state.alpha];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.t_ = x[0];
                s.alpha = x[1];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &nm_default,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            true,
        );
    }
    dbg_print("post_stage_1", &state);

    // Stage 2: (t_, beta, scaling)
    if fit_scaling {
        let x0 = vec![state.t_, state.beta, state.scaling];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.t_ = x[0];
                s.beta = x[1];
                s.scaling = x[2];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &nm_default,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            true,
        );
    }
    dbg_print("post_stage_2", &state);

    // Stage 3: (alpha, gamma)
    {
        let x0 = vec![state.alpha, state.gamma];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.alpha = x[0];
                s.gamma = x[1];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &nm_loose,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            true,
        );
    }
    dbg_print("post_stage_3", &state);

    // Stage 4: (t_,)
    {
        let x0 = vec![state.t_];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.t_ = x[0];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &nm_default,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            true,
        );
    }
    dbg_print("post_stage_4", &state);

    // Stage 5: (t_, alpha, beta, gamma)
    {
        let x0 = vec![state.t_, state.alpha, state.beta, state.gamma];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.t_ = x[0];
                s.alpha = x[1];
                s.beta = x[2];
                s.gamma = x[3];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            AssignmentMode::None,
            &nm_loose,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            true,
        );
    }
    dbg_print("post_stage_5", &state);

    // Stage 6: switch to projection mode + refit with cached time grid.
    let assignment_mode = AssignmentMode::Projection;
    {
        // Re-anchor last_loss under the new mode without changing params.
        let _ = try_update(
            &mut state,
            |_c| {},
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            assignment_mode,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            false,
        );

        // refit_time=False: NM optimises rates against the cached time grid.
        let x0 = vec![state.t_, state.alpha, state.beta, state.gamma];
        run_stage(
            &mut state,
            x0,
            |s, x| {
                s.t_ = x[0];
                s.alpha = x[1];
                s.beta = x[2];
                s.gamma = x[3];
            },
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            assignment_mode,
            &nm_loose,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
            false,
        );

        // Final no-arg update: gives alt_t_ one last chance to fire.
        let _ = try_update(
            &mut state,
            |_c| {},
            u,
            s,
            weights,
            conn,
            fit_steady_states,
            assignment_mode,
            &mut u_scaled_scratch,
            &mut distx_scratch,
            true,
        );
    }
    dbg_print("post_stage_6", &state);

    // Final mse → variance / likelihood approximation.
    if u_scaled_scratch.len() != n {
        u_scaled_scratch.clear();
        u_scaled_scratch.resize(n, 0.0);
    }
    let scaling = state.scaling;
    for i in 0..n {
        u_scaled_scratch[i] = u[i] / scaling;
    }

    let (u0_eff, s0_eff) =
        splicing_solution_scalar(state.t_, state.alpha, state.beta, state.gamma, 0.0, 0.0);
    state.u0_ = u0_eff;
    state.s0_ = s0_eff;

    get_distx_full(
        &u_scaled_scratch,
        s,
        weights,
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
        conn,
        assignment_mode,
        state.steady_state_ratio,
        &mut distx_scratch,
    );
    let n_w = distx_scratch.len();
    let varx = if n_w > 0 {
        pairwise_sum(&distx_scratch) / n_w as f64
    } else {
        f64::NAN
    };

    // Capture the per-cell layer outputs. fit_t comes from the cached state
    // (last committed update); fit_tau/fit_tau_ come from a fresh assignment.
    let assign_unmasked = crate::divergence::assign_timepoints(
        &u_scaled_scratch,
        s,
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
    );
    let assign = crate::divergence::AssignTimepoints {
        t: state.cached_t.clone(),
        tau: assign_unmasked.tau,
        tau_: assign_unmasked.tau_,
        tau_unmasked: assign_unmasked.tau_unmasked,
        tau__unmasked: assign_unmasked.tau__unmasked,
        o: assign_unmasked.o,
    };
    let _ = vectorize_per_cell;

    let likelihood = (-0.5 * varx).exp();

    GeneFitFull {
        alpha: state.alpha,
        beta: state.beta,
        gamma: state.gamma,
        t_: state.t_,
        scaling: state.scaling,
        likelihood,
        variance: varx,
        fit_t: assign.t,
        fit_tau: assign.tau_unmasked,
        fit_tau_: assign.tau__unmasked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_synthetic_gene() {
        let alpha_true = 1.5;
        let beta_true = 0.8;
        let gamma_true = 0.4;
        let n_cells = 200;
        let mut u = vec![0.0f64; n_cells];
        let mut s = vec![0.0f64; n_cells];
        for i in 0..n_cells {
            let t = (i as f64) * 0.05;
            let expu = (-beta_true * t).exp();
            let exps = (-gamma_true * t).exp();
            u[i] = alpha_true / beta_true * (1.0 - expu);
            let c = alpha_true / (gamma_true - beta_true);
            s[i] = alpha_true / gamma_true * (1.0 - exps) + c * (exps - expu);
        }
        let weights = vec![true; n_cells];

        let init = Initial {
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.5,
            scaling: 1.0,
            t_: 5.0,
            u0_: 0.5,
            s0_: 0.5,
            std_u: u.iter().fold(0.0f64, |a, &x| a + x * x).sqrt() / (n_cells as f64).sqrt(),
            std_s: s.iter().fold(0.0f64, |a, &x| a + x * x).sqrt() / (n_cells as f64).sqrt(),
            steady_state_ratio: Some(0.5),
            f32_mode: false,
        };

        let fit = fit_one_gene(&u, &s, &weights, None, init, 10, true, true, 0);
        assert!(!fit.alpha.is_nan(), "should converge to a real value");
        assert!(fit.alpha > 0.0 && fit.alpha < 5.0, "alpha={}", fit.alpha);
        assert!(fit.beta > 0.0 && fit.beta < 3.0, "beta={}", fit.beta);
        assert!(fit.gamma > 0.0 && fit.gamma < 3.0, "gamma={}", fit.gamma);
    }
}
