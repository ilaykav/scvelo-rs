use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{csr, diff_kinetic};

/// Bit-exact differential-kinetics LRT, Rayon-parallel over genes.
///
/// Inputs (all f64 unless noted):
/// - `u_raw`, `s`, `t_cell`: (n_cells, n_genes) C-contiguous. `u_raw`/`s` are
///   the RAW Mu/Ms layers (not divided by scaling). `t_cell` is
///   `adata.layers["fit_t"][:, var_idx]`, the per-cell time `load_pars` reads.
/// - per-gene 1-D: `alpha`, `beta` (= `fit_beta * fit_scaling`, the internal
///   beta scvelo uses after load_pars), `gamma`, `scaling`, `t_`.
/// - `cluster_assign`: (n_cells,) i32 - per-cell cluster id; -1 to skip.
/// - `n_clusters`, `min_cells`.
/// - optional CSR connectivities (conn_data, conn_indices, conn_indptr).
///
/// Returns: (n_genes, n_clusters) f64 array of per-cluster p-values.
#[pyfunction]
#[pyo3(signature = (
    u_raw, s,
    alpha, beta, gamma, scaling, t_,
    cluster_assign, n_clusters, min_cells=10,
    conn_data=None, conn_indices=None, conn_indptr=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn diff_kinetic_test_kernel<'py>(
    py: Python<'py>,
    u_raw: PyReadonlyArray2<'py, f64>,
    s: PyReadonlyArray2<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    gamma: PyReadonlyArray1<'py, f64>,
    scaling: PyReadonlyArray1<'py, f64>,
    t_: PyReadonlyArray1<'py, f64>,
    cluster_assign: PyReadonlyArray1<'py, i32>,
    n_clusters: usize,
    min_cells: usize,
    conn_data: Option<PyReadonlyArray1<'py, f64>>,
    conn_indices: Option<PyReadonlyArray1<'py, i32>>,
    conn_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let u_view = u_raw.as_array();
    let s_view = s.as_array();

    let (n_cells, n_genes) = (u_view.shape()[0], u_view.shape()[1]);
    if s_view.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u_raw and s must both be (n_cells, n_genes)",
        ));
    }
    for (name, len) in [
        ("alpha", alpha.len()?),
        ("beta", beta.len()?),
        ("gamma", gamma.len()?),
        ("scaling", scaling.len()?),
        ("t_", t_.len()?),
    ] {
        if len != n_genes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "per-gene array {name} has length {len}, expected {n_genes}"
            )));
        }
    }
    if cluster_assign.len()? != n_cells {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_assign must have length n_cells",
        ));
    }

    let u_slice = u_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("u_raw must be C-contiguous"))?;
    let s_slice = s_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("s must be C-contiguous"))?;

    let alpha_arr = alpha.as_array();
    let alpha_s = alpha_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("alpha must be C-contiguous"))?;
    let beta_arr = beta.as_array();
    let beta_s = beta_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("beta must be C-contiguous"))?;
    let gamma_arr = gamma.as_array();
    let gamma_s = gamma_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("gamma must be C-contiguous"))?;
    let scaling_arr = scaling.as_array();
    let scaling_s = scaling_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("scaling must be C-contiguous"))?;
    let t__arr = t_.as_array();
    let t__s = t__arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("t_ must be C-contiguous"))?;
    let ca_arr = cluster_assign.as_array();
    let ca_s = ca_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("cluster_assign must be C-contiguous")
    })?;

    // Optional CSR connectivities (shared by reference across gene tasks).
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

    let pvals_flat = py.allow_threads(|| {
        diff_kinetic::diff_kinetic_test_kernel(
            n_cells, n_genes, u_slice, s_slice, alpha_s, beta_s, gamma_s, scaling_s, t__s, ca_s,
            n_clusters, min_cells, conn_view,
        )
    });

    let pvals_arr = ndarray::Array2::from_shape_vec((n_genes, n_clusters), pvals_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("pvals shape error: {e}")))?;
    Ok(pvals_arr.into_pyarray_bound(py))
}
