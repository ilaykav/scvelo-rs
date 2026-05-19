use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{csr, diff_kinetic};

/// Batched per-gene differential-kinetics LRT.
///
/// Inputs:
/// - `u_scaled`, `s`, `weights`: (n_cells, n_genes) C-contiguous arrays.
///   `u_scaled` is `Mu / scaling` per gene.
/// - per-gene 1-D arrays of length n_genes: `alpha`, `beta`, `gamma`,
///   `scaling`, `t_`, `u0_`, `s0_`, `std_u` (= std of u_scaled), `std_s`, `varx`.
/// - `cluster_assign`: (n_cells,) i32 — integer cluster id per cell.
/// - `n_clusters`, `min_cells`, `fit_steady_states`.
/// - optional CSR connectivities (conn_data, conn_indices, conn_indptr).
///
/// Returns (n_genes, n_clusters) f64 array of per-cluster p-values.
#[pyfunction]
#[pyo3(signature = (
    u_scaled, s, weights,
    alpha, beta, gamma, scaling, t_, u0_, s0_,
    std_u, std_s, varx,
    cluster_assign, n_clusters, min_cells=10, fit_steady_states=true,
    conn_data=None, conn_indices=None, conn_indptr=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn diff_kinetic_test_kernel<'py>(
    py: Python<'py>,
    u_scaled: PyReadonlyArray2<'py, f64>,
    s: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, bool>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    gamma: PyReadonlyArray1<'py, f64>,
    scaling: PyReadonlyArray1<'py, f64>,
    t_: PyReadonlyArray1<'py, f64>,
    u0_: PyReadonlyArray1<'py, f64>,
    s0_: PyReadonlyArray1<'py, f64>,
    std_u: PyReadonlyArray1<'py, f64>,
    std_s: PyReadonlyArray1<'py, f64>,
    varx: PyReadonlyArray1<'py, f64>,
    cluster_assign: PyReadonlyArray1<'py, i32>,
    n_clusters: usize,
    min_cells: usize,
    fit_steady_states: bool,
    conn_data: Option<PyReadonlyArray1<'py, f64>>,
    conn_indices: Option<PyReadonlyArray1<'py, i32>>,
    conn_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let u_view = u_scaled.as_array();
    let s_view = s.as_array();
    let w_view = weights.as_array();

    let (n_cells, n_genes) = (u_view.shape()[0], u_view.shape()[1]);
    if s_view.shape() != [n_cells, n_genes] || w_view.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u_scaled, s, weights must all be (n_cells, n_genes)",
        ));
    }
    if alpha.len()? != n_genes
        || beta.len()? != n_genes
        || gamma.len()? != n_genes
        || scaling.len()? != n_genes
        || t_.len()? != n_genes
        || u0_.len()? != n_genes
        || s0_.len()? != n_genes
        || std_u.len()? != n_genes
        || std_s.len()? != n_genes
        || varx.len()? != n_genes
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "per-gene 1-D arrays must all have length n_genes",
        ));
    }
    if cluster_assign.len()? != n_cells {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_assign must have length n_cells",
        ));
    }

    let u_slice = u_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("u_scaled must be C-contiguous")
    })?;
    let s_slice = s_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("s must be C-contiguous"))?;
    let w_slice = w_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("weights must be C-contiguous"))?;

    let alpha_arr = alpha.as_array();
    let alpha_s = alpha_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("alpha must be C-contiguous")
    })?;
    let beta_arr = beta.as_array();
    let beta_s = beta_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("beta must be C-contiguous"))?;
    let gamma_arr = gamma.as_array();
    let gamma_s = gamma_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("gamma must be C-contiguous")
    })?;
    let scaling_arr = scaling.as_array();
    let scaling_s = scaling_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("scaling must be C-contiguous")
    })?;
    let t__arr = t_.as_array();
    let t__s = t__arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("t_ must be C-contiguous"))?;
    let u0__arr = u0_.as_array();
    let u0__s = u0__arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("u0_ must be C-contiguous"))?;
    let s0__arr = s0_.as_array();
    let s0__s = s0__arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("s0_ must be C-contiguous"))?;
    let std_u_arr = std_u.as_array();
    let std_u_s = std_u_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("std_u must be C-contiguous")
    })?;
    let std_s_arr = std_s.as_array();
    let std_s_s = std_s_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("std_s must be C-contiguous")
    })?;
    let varx_arr = varx.as_array();
    let varx_s = varx_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("varx must be C-contiguous"))?;
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
            n_cells,
            n_genes,
            u_slice,
            s_slice,
            w_slice,
            alpha_s,
            beta_s,
            gamma_s,
            scaling_s,
            t__s,
            u0__s,
            s0__s,
            std_u_s,
            std_s_s,
            varx_s,
            ca_s,
            n_clusters,
            min_cells,
            fit_steady_states,
            conn_view,
        )
    });

    let pvals_arr =
        ndarray::Array2::from_shape_vec((n_genes, n_clusters), pvals_flat).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("pvals shape error: {e}"))
        })?;
    Ok(pvals_arr.into_pyarray_bound(py))
}
