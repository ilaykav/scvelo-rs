use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::{velocity, velocity_graph};

#[pyfunction]
#[pyo3(signature = (x, v, indices, n_recurse_neighbors=2))]
pub fn velocity_graph_kernel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    v: PyReadonlyArray2<'py, f32>,
    indices: PyReadonlyArray2<'py, i32>,
    n_recurse_neighbors: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let x_arr = x.as_array();
    let v_arr = v.as_array();
    let idx_arr = indices.as_array();
    let (n_cells, n_genes) = (x_arr.shape()[0], x_arr.shape()[1]);
    if v_arr.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and v must have the same shape",
        ));
    }
    let n_knn = idx_arr.shape()[1];
    if idx_arr.shape()[0] != n_cells {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices must be (n_cells, n_knn)",
        ));
    }
    let x_slice = x_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("x must be C-contiguous"))?;
    let v_slice = v_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("v must be C-contiguous"))?;
    let idx_slice = idx_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("indices must be C-contiguous"))?;

    let triplets = py.allow_threads(|| {
        velocity_graph::compute_cosines_all(
            x_slice,
            v_slice,
            idx_slice,
            n_cells,
            n_genes,
            n_knn,
            n_recurse_neighbors,
        )
    });

    let total: usize = triplets.iter().map(|t| t.vals.len()).sum();
    let mut rows = Vec::with_capacity(total);
    let mut cols = Vec::with_capacity(total);
    let mut vals = Vec::with_capacity(total);
    for t in triplets {
        for (col, val) in t.neighs.into_iter().zip(t.vals.into_iter()) {
            rows.push(t.source);
            cols.push(col);
            vals.push(val);
        }
    }
    Ok((
        rows.into_pyarray_bound(py),
        cols.into_pyarray_bound(py),
        vals.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (ms, mu, fit_offset=false, min_r2=0.01, min_ratio=0.01,
                    constrain_lo=None, constrain_hi=None,
                    perc_lo=None, perc_hi=None))]
#[allow(clippy::too_many_arguments)]
pub fn velocity_kernel<'py>(
    py: Python<'py>,
    ms: PyReadonlyArray2<'py, f64>,
    mu: PyReadonlyArray2<'py, f64>,
    fit_offset: bool,
    min_r2: f64,
    min_ratio: f64,
    constrain_lo: Option<f64>,
    constrain_hi: Option<f64>,
    perc_lo: Option<f64>,
    perc_hi: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<bool>>,
)> {
    let ms_arr = ms.as_array();
    let mu_arr = mu.as_array();
    let (n_cells, n_genes) = (ms_arr.shape()[0], ms_arr.shape()[1]);
    if mu_arr.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Ms and Mu must have the same shape",
        ));
    }
    let ms_slice = ms_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Ms must be C-contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Mu must be C-contiguous"))?;

    let constrain = match (constrain_lo, constrain_hi) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        (None, Some(hi)) => Some((f64::NEG_INFINITY, hi)),
        (Some(lo), None) => Some((lo, f64::INFINITY)),
        (None, None) => None,
    };
    let percentile = perc_lo.map(|lo| (lo, perc_hi));

    let fit = py.allow_threads(|| {
        velocity::fit_deterministic(
            ms_slice, mu_slice, n_cells, n_genes, fit_offset, min_r2, min_ratio, constrain,
            percentile,
        )
    });

    let residual_arr = ndarray::Array2::from_shape_vec((n_cells, n_genes), fit.residual)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok((
        fit.gamma.into_pyarray_bound(py),
        fit.offset.into_pyarray_bound(py),
        fit.r2.into_pyarray_bound(py),
        residual_arr.to_pyarray_bound(py),
        numpy::PyArray1::from_vec_bound(py, fit.velocity_genes),
    ))
}
