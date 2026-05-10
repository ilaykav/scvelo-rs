use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{csr, divergence, projection};

#[pyfunction]
#[pyo3(signature = (
    t, t_, alpha, beta, gamma,
    scaling=1.0, u0_offset=0.0, s0_offset=0.0,
    parallel=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn compute_dynamics_kernel<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    t_: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    u0_offset: f64,
    s0_offset: f64,
    parallel: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let t_arr = t.as_array();
    let t_slice = t_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("t must be C-contiguous"))?;
    let n = t_slice.len();
    let mut a_out = vec![0.0f64; n];
    let mut u_out = vec![0.0f64; n];
    let mut s_out = vec![0.0f64; n];
    py.allow_threads(|| {
        projection::compute_dynamics_eval(
            t_slice, t_, alpha, beta, gamma, scaling, u0_offset, s0_offset,
            &mut a_out, &mut u_out, &mut s_out, parallel,
        );
    });
    Ok((
        a_out.into_pyarray_bound(py),
        u_out.into_pyarray_bound(py),
        s_out.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (t, alpha, beta, gamma, u0=0.0, s0=0.0, parallel=false))]
pub fn splicing_dynamics_eval_kernel<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let t_arr = t.as_array();
    let t_slice = t_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("t must be C-contiguous"))?;
    let n = t_slice.len();
    let mut u = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];
    py.allow_threads(|| {
        projection::splicing_dynamics_eval(
            t_slice, alpha, beta, gamma, u0, s0, &mut u, &mut s, parallel,
        );
    });
    Ok((u.into_pyarray_bound(py), s.into_pyarray_bound(py)))
}

#[pyfunction]
#[pyo3(signature = (u, s, alpha, beta, gamma, t_end, num, u0=0.0, s0=0.0, parallel=true))]
#[allow(clippy::too_many_arguments)]
pub fn project_to_curve_kernel<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    t_end: f64,
    num: usize,
    u0: f64,
    s0: f64,
    parallel: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let u_arr = u.as_array();
    let s_arr = s.as_array();
    if u_arr.len() != s_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u and s must have the same length",
        ));
    }
    if num < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("num must be >= 2"));
    }

    let mut xt_u = vec![0.0f64; num];
    let mut xt_s = vec![0.0f64; num];
    let mut tpoints = vec![0.0f64; num];
    projection::sample_curve(
        t_end,
        num,
        alpha,
        beta,
        gamma,
        u0,
        s0,
        &mut xt_u,
        &mut xt_s,
        &mut tpoints,
    );

    let mut tau = vec![0.0f64; u_arr.len()];
    py.allow_threads(|| {
        if parallel {
            projection::project_to_curve_parallel(u_arr, s_arr, &xt_u, &xt_s, &tpoints, &mut tau);
        } else {
            projection::project_to_curve_serial(u_arr, s_arr, &xt_u, &xt_s, &tpoints, &mut tau);
        }
    });
    Ok((
        tau.into_pyarray_bound(py),
        xt_u.into_pyarray_bound(py),
        xt_s.into_pyarray_bound(py),
        tpoints.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    u_scaled, s_full,
    alpha, beta, gamma, scaling, t_, u0_, s0_,
    std_u, std_s, fit_steady_states, assignment_mode_str,
    constraint_time_increments=true,
    conn_data=None, conn_indices=None, conn_indptr=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn assign_timepoints_kernel<'py>(
    py: Python<'py>,
    u_scaled: PyReadonlyArray1<'py, f64>,
    s_full: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    scaling: f64,
    t_: f64,
    u0_: f64,
    s0_: f64,
    std_u: f64,
    std_s: f64,
    fit_steady_states: bool,
    assignment_mode_str: Option<&str>,
    constraint_time_increments: bool,
    conn_data: Option<PyReadonlyArray1<'py, f64>>,
    conn_indices: Option<PyReadonlyArray1<'py, i32>>,
    conn_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let u_arr = u_scaled.as_array();
    let s_arr = s_full.as_array();
    if u_arr.len() != s_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    let u_slice = u_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("u_scaled must be C-contiguous"))?;
    let s_slice = s_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("s_full must be C-contiguous"))?;

    let mode = match assignment_mode_str {
        Some("projection") => divergence::AssignmentMode::Projection,
        Some("full_projection") => divergence::AssignmentMode::FullProjection,
        Some("partial_projection") => divergence::AssignmentMode::PartialProjection,
        _ => divergence::AssignmentMode::None,
    };

    let conn_data_view = conn_data.as_ref().map(|a| a.as_array());
    let conn_indices_view = conn_indices.as_ref().map(|a| a.as_array());
    let conn_indptr_view = conn_indptr.as_ref().map(|a| a.as_array());
    let conn_view = match (
        conn_data_view.as_ref(),
        conn_indices_view.as_ref(),
        conn_indptr_view.as_ref(),
    ) {
        (Some(d), Some(i), Some(p)) => Some(csr::CsrView::new(
            d.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("conn_data must be C-contiguous")
            })?,
            i.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("conn_indices must be C-contiguous")
            })?,
            p.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("conn_indptr must be C-contiguous")
            })?,
        )),
        _ => None,
    };

    let assign = py.allow_threads(|| {
        divergence::assign_timepoints(
            u_slice,
            s_slice,
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
            constraint_time_increments,
            conn_view,
            mode,
        )
    });
    Ok((
        assign.t.into_pyarray_bound(py),
        assign.tau.into_pyarray_bound(py),
        assign.o.into_pyarray_bound(py),
    ))
}
