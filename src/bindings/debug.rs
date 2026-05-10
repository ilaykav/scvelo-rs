use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{dynamics, numpy_compat};

#[pyfunction]
#[pyo3(signature = (u, s, u0, s0, alpha, beta, gamma))]
pub fn _debug_tau_inv<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    u0: f64,
    s0: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let u_arr = u.as_array();
    let s_arr = s.as_array();
    let u_slice = u_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    let s_slice = s_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    let n = u_slice.len();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[i] = dynamics::tau_inv_scalar(u_slice[i], s_slice[i], u0, s0, alpha, beta, gamma, true);
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (u, s, u0, s0, alpha, beta, gamma))]
pub fn _debug_tau_inv_intermediates(
    u: f64,
    s: f64,
    u0: f64,
    s0: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let gmb: f64 = gamma - beta;
    let inv: f64 = if gmb == 0.0 { 0.0 } else { 1.0 / gmb };
    let beta_: f64 = beta * inv;
    let xinf: f64 = alpha / gamma - beta_ * (alpha / beta);
    let num: f64 = s - beta_ * u - xinf;
    let den: f64 = s0 - beta_ * u0 - xinf;
    let ratio: f64 = num / den;
    let lb: f64 = 1e-6_f64;
    let ub: f64 = 1.0 - 1e-6_f64;
    let clipped: f64 = if ratio.is_nan() {
        f64::NAN
    } else if ratio < lb {
        lb
    } else if ratio > ub {
        ub
    } else {
        ratio
    };
    let log_clipped: f64 = clipped.ln();
    let tau: f64 = -1.0 / gamma * log_clipped;
    (gmb, beta_, xinf, num, den, ratio, log_clipped, tau)
}

#[pyfunction]
pub fn _debug_splicing_solution<'py>(
    py: Python<'py>,
    tau: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    u0: f64,
    s0: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let t_arr = tau.as_array();
    let t_slice = t_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    let n = t_slice.len();
    let mut u = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];
    py.allow_threads(|| {
        dynamics::splicing_solution_array(t_slice, alpha, beta, gamma, u0, s0, &mut u, &mut s)
    });
    Ok((u.into_pyarray_bound(py), s.into_pyarray_bound(py)))
}

#[pyfunction]
pub fn _debug_exp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_array();
    let xs_slice = xs
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    let mut out = vec![0.0f64; xs_slice.len()];
    for i in 0..xs_slice.len() {
        out[i] = xs_slice[i].exp();
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn _debug_ln<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_array();
    let xs_slice = xs
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    let mut out = vec![0.0f64; xs_slice.len()];
    for i in 0..xs_slice.len() {
        out[i] = xs_slice[i].ln();
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
pub fn _debug_pairwise_sum<'py>(_py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
    let xs = x.as_array();
    let xs_slice = xs
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("not C-contiguous"))?;
    Ok(numpy_compat::pairwise_sum(xs_slice))
}
