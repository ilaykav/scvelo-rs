use crate::csr::{matvec_multi, CsrView};
use crate::dynamics::{
    splicing_solution_array, splicing_solution_array_f32_emu, tau_inv_scalar,
    tau_inv_scalar_f32_emu,
};
use crate::projection;

use super::increments::{adjust_increments_paired, adjust_increments_single};
use super::{f32q, AssignTimepoints, AssignmentMode};

pub fn assign_timepoints(
    u: &[f64],
    s: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    std_u_in: f64,
    std_s: f64,
    fit_steady_states: bool,
    constraint_time_increments: bool,
    connectivities: Option<CsrView<'_>>,
    assignment_mode: AssignmentMode,
) -> AssignTimepoints {
    assign_timepoints_dtyped(
        u,
        s,
        alpha,
        beta,
        gamma,
        scaling,
        t_,
        u0_,
        s0_,
        std_u_in,
        std_s,
        fit_steady_states,
        constraint_time_increments,
        connectivities,
        assignment_mode,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn assign_timepoints_dtyped(
    u: &[f64],
    s: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    std_u_in: f64,
    std_s: f64,
    fit_steady_states: bool,
    constraint_time_increments: bool,
    connectivities: Option<CsrView<'_>>,
    assignment_mode: AssignmentMode,
    f32_mode: bool,
) -> AssignTimepoints {
    let n = u.len();
    debug_assert_eq!(s.len(), n);

    let std_u = if f32_mode {
        f32q(std_u_in / scaling)
    } else {
        std_u_in / scaling
    };

    let mut tau = vec![0.0f64; n];
    let mut tau_ = vec![0.0f64; n];
    assign_tau_full_match(
        u,
        s,
        alpha,
        beta,
        gamma,
        t_,
        u0_,
        s0_,
        assignment_mode,
        &mut tau,
        &mut tau_,
        f32_mode,
    );

    if constraint_time_increments {
        let mut ut = vec![0.0f64; n];
        let mut st = vec![0.0f64; n];
        let mut ut_ = vec![0.0f64; n];
        let mut st_ = vec![0.0f64; n];
        if f32_mode {
            splicing_solution_array_f32_emu(&tau, alpha, beta, gamma, 0.0, 0.0, &mut ut, &mut st);
            splicing_solution_array_f32_emu(&tau_, 0.0, beta, gamma, u0_, s0_, &mut ut_, &mut st_);
        } else {
            splicing_solution_array(&tau, alpha, beta, gamma, 0.0, 0.0, &mut ut, &mut st);
            splicing_solution_array(&tau_, 0.0, beta, gamma, u0_, s0_, &mut ut_, &mut st_);
        }

        let mut row_off = vec![0.0f64; n];
        let mut row_on = vec![0.0f64; n];
        if f32_mode {
            for i in 0..n {
                let du = f32q((u[i] - ut[i]) / std_u);
                let du_ = f32q((u[i] - ut_[i]) / std_u);
                let ds = f32q((s[i] - st[i]) / std_s);
                let ds_ = f32q((s[i] - st_[i]) / std_s);
                row_off[i] = f32q(f32q(du_ * du_) + f32q(ds_ * ds_));
                row_on[i] = f32q(f32q(du * du) + f32q(ds * ds));
            }
        } else {
            for i in 0..n {
                let du = (u[i] - ut[i]) / std_u;
                let du_ = (u[i] - ut_[i]) / std_u;
                let ds = (s[i] - st[i]) / std_s;
                let ds_ = (s[i] - st_[i]) / std_s;
                row_off[i] = du_ * du_ + ds_ * ds_;
                row_on[i] = du * du + ds * ds;
            }
        }

        let (smoothed_off, smoothed_on);
        let mut so = vec![0.0f64; n];
        let mut sn = vec![0.0f64; n];
        let (off_view, on_view): (&[f64], &[f64]) = match connectivities {
            Some(conn) => {
                let cols: [&[f64]; 2] = [&row_off, &row_on];
                {
                    let (lo, hi) = (&mut so, &mut sn);
                    let mut outs: [&mut [f64]; 2] = [lo, hi];
                    matvec_multi(conn, &cols, &mut outs);
                }
                smoothed_off = &so[..];
                smoothed_on = &sn[..];
                (smoothed_off, smoothed_on)
            }
            None => (&row_off[..], &row_on[..]),
        };

        let mut o = vec![0u8; n];
        for i in 0..n {
            o[i] = if off_view[i] <= on_view[i] { 0 } else { 1 };
        }

        let any_on = o.contains(&1);
        let any_off = o.contains(&0);
        if any_on && any_off {
            adjust_increments_paired(&mut tau, &mut tau_, &o);
        } else if any_on {
            adjust_increments_single(&mut tau, &o, true);
        } else if any_off {
            adjust_increments_single(&mut tau_, &o, false);
        }
    }

    let mut ut = vec![0.0f64; n];
    let mut st = vec![0.0f64; n];
    let mut ut_ = vec![0.0f64; n];
    let mut st_ = vec![0.0f64; n];
    if f32_mode {
        splicing_solution_array_f32_emu(&tau, alpha, beta, gamma, 0.0, 0.0, &mut ut, &mut st);
        splicing_solution_array_f32_emu(&tau_, 0.0, beta, gamma, u0_, s0_, &mut ut_, &mut st_);
    } else {
        splicing_solution_array(&tau, alpha, beta, gamma, 0.0, 0.0, &mut ut, &mut st);
        splicing_solution_array(&tau_, 0.0, beta, gamma, u0_, s0_, &mut ut_, &mut st_);
    }

    let mut row0 = vec![0.0f64; n];
    let mut row1 = vec![0.0f64; n];
    let mut row2 = vec![0.0f64; n];
    let mut row3 = vec![0.0f64; n];

    let u_inf = if f32_mode {
        f32q(alpha / beta)
    } else {
        alpha / beta
    };
    let s_inf = if f32_mode {
        f32q(alpha / gamma)
    } else {
        alpha / gamma
    };
    if f32_mode {
        for i in 0..n {
            let du = f32q((u[i] - ut[i]) / std_u);
            let du_ = f32q((u[i] - ut_[i]) / std_u);
            let ds = f32q((s[i] - st[i]) / std_s);
            let ds_ = f32q((s[i] - st_[i]) / std_s);
            row0[i] = f32q(f32q(du_ * du_) + f32q(ds_ * ds_));
            row1[i] = f32q(f32q(du * du) + f32q(ds * ds));
            if fit_steady_states {
                let usu = f32q(u[i] / std_u);
                let ssu = f32q(s[i] / std_s);
                row2[i] = f32q(f32q(usu * usu) + f32q(ssu * ssu));
                let usi = f32q((u[i] - u_inf) / std_u);
                let ssi = f32q((s[i] - s_inf) / std_s);
                row3[i] = f32q(f32q(usi * usi) + f32q(ssi * ssi));
            }
        }
    } else {
        for i in 0..n {
            let du = (u[i] - ut[i]) / std_u;
            let du_ = (u[i] - ut_[i]) / std_u;
            let ds = (s[i] - st[i]) / std_s;
            let ds_ = (s[i] - st_[i]) / std_s;
            row0[i] = du_ * du_ + ds_ * ds_;
            row1[i] = du * du + ds * ds;
            if fit_steady_states {
                let usu = u[i] / std_u;
                let ssu = s[i] / std_s;
                row2[i] = usu * usu + ssu * ssu;
                let usi = (u[i] - u_inf) / std_u;
                let ssi = (s[i] - s_inf) / std_s;
                row3[i] = usi * usi + ssi * ssi;
            }
        }
    }

    if let Some(conn) = connectivities {
        if fit_steady_states {
            let cols: [&[f64]; 4] = [&row0, &row1, &row2, &row3];
            let mut s0v = vec![0.0f64; n];
            let mut s1v = vec![0.0f64; n];
            let mut s2v = vec![0.0f64; n];
            let mut s3v = vec![0.0f64; n];
            {
                let mut outs: [&mut [f64]; 4] = [&mut s0v, &mut s1v, &mut s2v, &mut s3v];
                matvec_multi(conn, &cols, &mut outs);
            }
            row0 = s0v;
            row1 = s1v;
            row2 = s2v;
            row3 = s3v;
        } else {
            let cols: [&[f64]; 2] = [&row0, &row1];
            let mut s0v = vec![0.0f64; n];
            let mut s1v = vec![0.0f64; n];
            {
                let mut outs: [&mut [f64]; 2] = [&mut s0v, &mut s1v];
                matvec_multi(conn, &cols, &mut outs);
            }
            row0 = s0v;
            row1 = s1v;
        }
    }

    let mut o = vec![0u8; n];
    if fit_steady_states {
        let rows: [&[f64]; 4] = [&row0, &row1, &row2, &row3];
        argmin_axis0_u8(&rows, &mut o);
    } else {
        let rows: [&[f64]; 2] = [&row0, &row1];
        argmin_axis0_u8(&rows, &mut o);
    }

    let tau_unmasked = tau.clone();
    let tau__unmasked = tau_.clone();

    let mut t = vec![0.0f64; n];
    for i in 0..n {
        let raw = o[i];
        let mul_off_raw = (raw == 0) as i32 as f64;
        let mul_on_raw = (raw == 1) as i32 as f64;
        tau_[i] *= mul_off_raw;
        tau[i] *= mul_on_raw;

        let collapsed = match raw {
            2 => 1u8,
            3 => 0u8,
            x => x,
        };
        o[i] = collapsed;

        let on = (collapsed == 1) as i32 as f64;
        let off = (collapsed == 0) as i32 as f64;
        t[i] = tau[i] * on + (tau_[i] + t_) * off;
    }

    AssignTimepoints {
        t,
        tau,
        tau_,
        tau_unmasked,
        tau__unmasked,
        o,
    }
}

#[allow(clippy::too_many_arguments)]
fn assign_tau_full_match(
    u: &[f64],
    s: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    mode: AssignmentMode,
    tau: &mut [f64],
    tau_: &mut [f64],
    f32_mode: bool,
) {
    let n = u.len();
    let use_projection = matches!(
        mode,
        AssignmentMode::FullProjection | AssignmentMode::PartialProjection
    ) || (matches!(mode, AssignmentMode::Projection) && beta < gamma);

    if use_projection {
        let num = (n / 5).clamp(200, 500);
        projection::assign_tau_full(
            u, s, alpha, beta, gamma, t_, u0_, s0_, num, tau, tau_, false,
        );
    } else {
        if f32_mode {
            let t_f = t_ as f32;
            for i in 0..n {
                let t = tau_inv_scalar_f32_emu(u[i], s[i], 0.0, 0.0, alpha, beta, gamma, true);
                let t_clamped = (t as f32).max(0.0).min(t_f) as f64;
                tau[i] = t_clamped;
            }
            for i in 0..n {
                tau_[i] = tau_inv_scalar_f32_emu(u[i], s[i], u0_, s0_, 0.0, beta, gamma, true);
            }
        } else {
            for i in 0..n {
                let t = tau_inv_scalar(u[i], s[i], 0.0, 0.0, alpha, beta, gamma, true);
                tau[i] = t.clamp(0.0, t_);
            }
            for i in 0..n {
                tau_[i] = tau_inv_scalar(u[i], s[i], u0_, s0_, 0.0, beta, gamma, true);
            }
        }
        let mut ub = f64::NEG_INFINITY;
        for i in 0..n {
            if s[i] > 0.0 && tau_[i] > ub {
                ub = tau_[i];
            }
        }
        if !ub.is_finite() {
            for v in tau_.iter_mut() {
                *v = 0.0;
            }
        } else {
            for v in tau_.iter_mut() {
                *v = v.clamp(0.0, ub);
            }
        }
    }
}

fn argmin_axis0_u8(rows: &[&[f64]], out: &mut [u8]) {
    let k = rows.len();
    let n = rows[0].len();
    debug_assert!(k <= 255);
    debug_assert_eq!(out.len(), n);
    for c in 0..n {
        let mut best_v = rows[0][c];
        let mut best_i = 0u8;
        for r in 1..k {
            let v = rows[r][c];
            if v < best_v {
                best_v = v;
                best_i = r as u8;
            }
        }
        out[c] = best_i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_timepoints_smoke() {
        let alpha = 1.5;
        let beta = 1.0;
        let gamma = 1.5;
        let t_ = 2.0;
        let u0_ = crate::dynamics::unspliced(t_, 0.0, alpha, beta);
        let s0_ = crate::dynamics::spliced(t_, 0.0, 0.0, alpha, beta, gamma);
        let n = 5;
        let mut u = vec![0.0; n];
        let mut s = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) * 0.5;
            u[i] = crate::dynamics::unspliced(t, 0.0, alpha, beta);
            s[i] = crate::dynamics::spliced(t, 0.0, 0.0, alpha, beta, gamma);
        }
        let res = assign_timepoints(
            &u,
            &s,
            alpha,
            beta,
            gamma,
            1.0,
            t_,
            u0_,
            s0_,
            1.0,
            1.0,
            true,
            true,
            None,
            AssignmentMode::None,
        );
        assert_eq!(res.t.len(), n);
        for &oi in &res.o {
            assert!(oi == 0 || oi == 1);
        }
    }
}
