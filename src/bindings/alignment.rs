use pyo3::prelude::*;

use crate::align_dynamics;

#[pyfunction]
#[pyo3(signature = (alpha, beta, gamma, t_, big_t, tau, tau_, t_max=20.0))]
#[allow(clippy::too_many_arguments)]
pub fn align_dynamics_kernel<'py>(
    _py: Python<'py>,
    alpha: numpy::PyReadwriteArray1<'py, f64>,
    beta: numpy::PyReadwriteArray1<'py, f64>,
    gamma: numpy::PyReadwriteArray1<'py, f64>,
    t_: numpy::PyReadwriteArray1<'py, f64>,
    big_t: numpy::PyReadwriteArray2<'py, f64>,
    tau: numpy::PyReadwriteArray2<'py, f64>,
    tau_: numpy::PyReadwriteArray2<'py, f64>,
    t_max: f64,
) -> PyResult<()> {
    let mut alpha = alpha;
    let mut beta = beta;
    let mut gamma = gamma;
    let mut t_v = t_;
    let mut big_t = big_t;
    let mut tau = tau;
    let mut tau_ = tau_;

    let alpha_slice = alpha
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("alpha must be C-contiguous"))?;
    let beta_slice = beta
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("beta must be C-contiguous"))?;
    let gamma_slice = gamma
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gamma must be C-contiguous"))?;
    let t__slice = t_v
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("t_ must be C-contiguous"))?;

    let n_genes = alpha_slice.len();
    let big_t_view = big_t.as_array_mut();
    let n_cells = big_t_view.shape()[0];
    if big_t_view.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "T must be (n_cells, n_genes)",
        ));
    }
    let big_t_slice = big_t
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("T must be C-contiguous"))?;
    let tau_slice = tau
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Tau must be C-contiguous"))?;
    let tau__slice = tau_
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Tau_ must be C-contiguous"))?;

    let mut idx = vec![false; n_genes];
    align_dynamics::compute_idx(big_t_slice, n_cells, n_genes, &mut idx);
    align_dynamics::align_total_time(
        alpha_slice,
        beta_slice,
        gamma_slice,
        t__slice,
        big_t_slice,
        tau_slice,
        tau__slice,
        &idx,
        n_cells,
        n_genes,
        t_max,
    );
    Ok(())
}
