use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::scanpy_replacement;

#[pyfunction]
#[pyo3(signature = (x, n_comps=50, zero_center=true))]
pub fn pca_kernel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    n_comps: usize,
    zero_center: bool,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x_arr = x.as_array();
    let (n_cells, n_genes) = (x_arr.shape()[0], x_arr.shape()[1]);
    let x_slice = x_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("x must be C-contiguous"))?;
    let n_comps_actual = n_comps.min(n_cells.min(n_genes));

    let (x_pca, pcs, var, var_ratio) = py.allow_threads(|| {
        scanpy_replacement::pca::fit(x_slice, n_cells, n_genes, n_comps, zero_center)
    });

    let x_pca_arr = ndarray::Array2::from_shape_vec((n_cells, n_comps_actual), x_pca)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let pcs_arr = ndarray::Array2::from_shape_vec((n_comps_actual, n_genes), pcs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    Ok((
        x_pca_arr.to_pyarray_bound(py),
        pcs_arr.to_pyarray_bound(py),
        var.into_pyarray_bound(py),
        var_ratio.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (x, k=30))]
pub fn knn_kernel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
    let x_arr = x.as_array();
    let (n_cells, n_genes) = (x_arr.shape()[0], x_arr.shape()[1]);
    let x_slice = x_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("x must be C-contiguous"))?;

    let (idx, dist) = py
        .allow_threads(|| scanpy_replacement::knn::fit_knn_euclidean(x_slice, n_cells, n_genes, k));
    Ok((idx.into_pyarray_bound(py), dist.into_pyarray_bound(py)))
}
