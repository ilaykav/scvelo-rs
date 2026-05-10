use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::{bimodality, csr, em, numpy_compat, recovery};

#[pyfunction]
#[pyo3(signature = (
    mu, ms, weights,
    alpha0, beta0, gamma0, scaling0, t_0, u0_0, s0_0,
    std_u0, std_s0, ssr0,
    max_iter, fit_scaling, fit_steady_states, f32_mode,
    conn_data=None, conn_indices=None, conn_indptr=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn recover_dynamics_kernel<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    ms: PyReadonlyArray2<'py, f64>,
    weights: numpy::PyReadonlyArray2<'py, bool>,
    alpha0: PyReadonlyArray1<'py, f64>,
    beta0: PyReadonlyArray1<'py, f64>,
    gamma0: PyReadonlyArray1<'py, f64>,
    scaling0: PyReadonlyArray1<'py, f64>,
    t_0: PyReadonlyArray1<'py, f64>,
    u0_0: PyReadonlyArray1<'py, f64>,
    s0_0: PyReadonlyArray1<'py, f64>,
    std_u0: PyReadonlyArray1<'py, f64>,
    std_s0: PyReadonlyArray1<'py, f64>,
    ssr0: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    fit_scaling: bool,
    fit_steady_states: bool,
    f32_mode: bool,
    conn_data: Option<PyReadonlyArray1<'py, f64>>,
    conn_indices: Option<PyReadonlyArray1<'py, i32>>,
    conn_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
)> {
    let mu_arr = mu.as_array();
    let ms_arr = ms.as_array();
    let w_arr = weights.as_array();
    if mu_arr.shape() != ms_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mu and ms must have the same shape",
        ));
    }
    if mu_arr.shape() != w_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must have the same shape as mu/ms",
        ));
    }
    if !mu_arr.is_standard_layout() || !ms_arr.is_standard_layout() || !w_arr.is_standard_layout() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mu/ms/weights must be C-contiguous",
        ));
    }
    let (n_cells, n_genes) = mu_arr.dim();

    let alpha0_v = alpha0.as_array();
    let beta0_v = beta0.as_array();
    let gamma0_v = gamma0.as_array();
    let scaling0_v = scaling0.as_array();
    let t_0_v = t_0.as_array();
    let u0_0_v = u0_0.as_array();
    let s0_0_v = s0_0.as_array();
    let std_u0_v = std_u0.as_array();
    let std_s0_v = std_s0.as_array();
    let ssr0_v = ssr0.as_array();
    for (name, len) in [
        ("alpha0", alpha0_v.len()),
        ("beta0", beta0_v.len()),
        ("gamma0", gamma0_v.len()),
        ("scaling0", scaling0_v.len()),
        ("t_0", t_0_v.len()),
        ("u0_0", u0_0_v.len()),
        ("s0_0", s0_0_v.len()),
        ("std_u0", std_u0_v.len()),
        ("std_s0", std_s0_v.len()),
        ("ssr0", ssr0_v.len()),
    ] {
        if len != n_genes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} length {len} != n_genes {n_genes}"
            )));
        }
    }

    let mu_owned: Vec<Vec<f64>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| mu_arr[[i, g]]).collect())
        .collect();
    let ms_owned: Vec<Vec<f64>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| ms_arr[[i, g]]).collect())
        .collect();
    let w_owned: Vec<Vec<bool>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| w_arr[[i, g]]).collect())
        .collect();

    let inits: Vec<recovery::Initial> = (0..n_genes)
        .map(|g| {
            let ssr = ssr0_v[g];
            recovery::Initial {
                alpha: alpha0_v[g],
                beta: beta0_v[g],
                gamma: gamma0_v[g],
                scaling: scaling0_v[g],
                t_: t_0_v[g],
                u0_: u0_0_v[g],
                s0_: s0_0_v[g],
                std_u: std_u0_v[g],
                std_s: std_s0_v[g],
                steady_state_ratio: if ssr.is_nan() { None } else { Some(ssr) },
                f32_mode,
            }
        })
        .collect();

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

    let mu_refs: Vec<&[f64]> = mu_owned.iter().map(|v| v.as_slice()).collect();
    let ms_refs: Vec<&[f64]> = ms_owned.iter().map(|v| v.as_slice()).collect();
    let w_refs: Vec<&[bool]> = w_owned.iter().map(|v| v.as_slice()).collect();

    let cfg = em::DriverConfig {
        max_iter,
        fit_scaling,
        fit_steady_states,
    };
    let fits =
        py.allow_threads(|| em::fit_all_genes(&mu_refs, &ms_refs, &w_refs, &inits, conn_view, cfg));

    let mut alpha = Vec::with_capacity(n_genes);
    let mut beta = Vec::with_capacity(n_genes);
    let mut gamma = Vec::with_capacity(n_genes);
    let mut t_ = Vec::with_capacity(n_genes);
    let mut scaling = Vec::with_capacity(n_genes);
    let mut likelihood = Vec::with_capacity(n_genes);
    let mut variance = Vec::with_capacity(n_genes);
    let mut fit_t_flat = vec![f64::NAN; n_cells * n_genes];
    let mut fit_tau_flat = vec![f64::NAN; n_cells * n_genes];
    let mut fit_tau__flat = vec![f64::NAN; n_cells * n_genes];

    for (g, f) in fits.iter().enumerate() {
        alpha.push(f.alpha);
        beta.push(f.beta);
        gamma.push(f.gamma);
        t_.push(f.t_);
        scaling.push(f.scaling);
        likelihood.push(f.likelihood);
        variance.push(f.variance);
        debug_assert_eq!(f.fit_t.len(), n_cells);
        for i in 0..n_cells {
            fit_t_flat[i * n_genes + g] = f.fit_t[i];
            fit_tau_flat[i * n_genes + g] = f.fit_tau[i];
            fit_tau__flat[i * n_genes + g] = f.fit_tau_[i];
        }
    }

    let fit_t_arr = ndarray::Array2::from_shape_vec((n_cells, n_genes), fit_t_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fit_t reshape: {e}")))?;
    let fit_tau_arr = ndarray::Array2::from_shape_vec((n_cells, n_genes), fit_tau_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fit_tau reshape: {e}")))?;
    let fit_tau__arr = ndarray::Array2::from_shape_vec((n_cells, n_genes), fit_tau__flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fit_tau_ reshape: {e}")))?;

    Ok((
        alpha.into_pyarray_bound(py),
        beta.into_pyarray_bound(py),
        gamma.into_pyarray_bound(py),
        t_.into_pyarray_bound(py),
        scaling.into_pyarray_bound(py),
        likelihood.into_pyarray_bound(py),
        variance.into_pyarray_bound(py),
        fit_t_arr.to_pyarray_bound(py),
        fit_tau_arr.to_pyarray_bound(py),
        fit_tau__arr.to_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (mu, ms, weights, fit_scaling, fit_steady_states, f32_mode,
                    pval_steady, steady_u, steady_s,
                    conn_data=None, conn_indices=None, conn_indptr=None))]
#[allow(clippy::too_many_arguments)]
pub fn initialize_all_genes_kernel<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    ms: PyReadonlyArray2<'py, f64>,
    weights: numpy::PyReadonlyArray2<'py, bool>,
    fit_scaling: bool,
    fit_steady_states: bool,
    f32_mode: bool,
    pval_steady: PyReadonlyArray1<'py, f64>,
    steady_u: PyReadonlyArray1<'py, f64>,
    steady_s: PyReadonlyArray1<'py, f64>,
    conn_data: Option<PyReadonlyArray1<'py, f64>>,
    conn_indices: Option<PyReadonlyArray1<'py, i32>>,
    conn_indptr: Option<PyReadonlyArray1<'py, i32>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let mu_arr = mu.as_array();
    let ms_arr = ms.as_array();
    let w_arr = weights.as_array();
    if mu_arr.shape() != ms_arr.shape() || mu_arr.shape() != w_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mu/ms/weights must all have the same shape",
        ));
    }
    let (n_cells, n_genes) = mu_arr.dim();

    let mu_owned: Vec<Vec<f64>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| mu_arr[[i, g]]).collect())
        .collect();
    let ms_owned: Vec<Vec<f64>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| ms_arr[[i, g]]).collect())
        .collect();
    let w_owned: Vec<Vec<bool>> = (0..n_genes)
        .map(|g| (0..n_cells).map(|i| w_arr[[i, g]]).collect())
        .collect();

    let mu_refs: Vec<&[f64]> = mu_owned.iter().map(|v| v.as_slice()).collect();
    let ms_refs: Vec<&[f64]> = ms_owned.iter().map(|v| v.as_slice()).collect();
    let w_refs: Vec<&[bool]> = w_owned.iter().map(|v| v.as_slice()).collect();

    let conn_data_view = conn_data.as_ref().map(|a| a.as_array());
    let conn_indices_view = conn_indices.as_ref().map(|a| a.as_array());
    let conn_indptr_view = conn_indptr.as_ref().map(|a| a.as_array());
    let conn_view = match (
        conn_data_view.as_ref(),
        conn_indices_view.as_ref(),
        conn_indptr_view.as_ref(),
    ) {
        (Some(d), Some(i), Some(p)) => Some(csr::CsrView::new(
            d.as_slice()
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("conn_data not C-contig"))?,
            i.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("conn_indices not C-contig")
            })?,
            p.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("conn_indptr not C-contig")
            })?,
        )),
        _ => None,
    };

    let pval_arr = pval_steady.as_array();
    let steady_u_arr = steady_u.as_array();
    let steady_s_arr = steady_s.as_array();
    let pval_slice = pval_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("pval_steady not C-contig"))?;
    let steady_u_slice = steady_u_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("steady_u not C-contig"))?;
    let steady_s_slice = steady_s_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("steady_s not C-contig"))?;
    let inits = py.allow_threads(|| {
        em::initialize_all_genes(
            &mu_refs,
            &ms_refs,
            &w_refs,
            conn_view,
            fit_scaling,
            fit_steady_states,
            f32_mode,
            pval_slice,
            steady_u_slice,
            steady_s_slice,
        )
    });

    let mut alpha = Vec::with_capacity(n_genes);
    let mut beta = Vec::with_capacity(n_genes);
    let mut gamma = Vec::with_capacity(n_genes);
    let mut scaling = Vec::with_capacity(n_genes);
    let mut t_ = Vec::with_capacity(n_genes);
    let mut u0_ = Vec::with_capacity(n_genes);
    let mut s0_ = Vec::with_capacity(n_genes);
    let mut std_u = Vec::with_capacity(n_genes);
    let mut std_s = Vec::with_capacity(n_genes);
    let mut ssr = Vec::with_capacity(n_genes);
    for init in &inits {
        alpha.push(init.alpha);
        beta.push(init.beta);
        gamma.push(init.gamma);
        scaling.push(init.scaling);
        t_.push(init.t_);
        u0_.push(init.u0_);
        s0_.push(init.s0_);
        std_u.push(init.std_u);
        std_s.push(init.std_s);
        ssr.push(init.steady_state_ratio.unwrap_or(f64::NAN));
    }
    Ok((
        alpha.into_pyarray_bound(py),
        beta.into_pyarray_bound(py),
        gamma.into_pyarray_bound(py),
        scaling.into_pyarray_bound(py),
        t_.into_pyarray_bound(py),
        u0_.into_pyarray_bound(py),
        s0_.into_pyarray_bound(py),
        std_u.into_pyarray_bound(py),
        std_s.into_pyarray_bound(py),
        ssr.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (mu, ms, perc=99.0, parallel=true))]
pub fn per_gene_weights_kernel<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    ms: PyReadonlyArray2<'py, f64>,
    perc: f64,
    parallel: bool,
) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
    let mu_arr = mu.as_array();
    let ms_arr = ms.as_array();
    let (n_cells, n_genes) = (mu_arr.shape()[0], mu_arr.shape()[1]);
    if ms_arr.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mu and Ms must have the same shape",
        ));
    }
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Mu must be C-contiguous"))?;
    let ms_slice = ms_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Ms must be C-contiguous"))?;

    let mut out_flat: Vec<bool> = vec![false; n_cells * n_genes];

    py.allow_threads(|| {
        let work = |g: usize, col_out: &mut [bool]| {
            let mut u_g_nz: Vec<f64> = Vec::with_capacity(n_cells);
            let mut s_g_nz: Vec<f64> = Vec::with_capacity(n_cells);
            for c in 0..n_cells {
                let u = mu_slice[c * n_genes + g];
                let s = ms_slice[c * n_genes + g];
                let nz = u > 0.0 && s > 0.0;
                col_out[c] = nz;
                if nz {
                    u_g_nz.push(u);
                    s_g_nz.push(s);
                }
            }
            if u_g_nz.len() <= 2 {
                return;
            }
            u_g_nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            s_g_nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let ub_u = numpy_compat::percentile_sorted(&u_g_nz, perc);
            let ub_s = numpy_compat::percentile_sorted(&s_g_nz, perc);
            for c in 0..n_cells {
                if !col_out[c] {
                    continue;
                }
                let u = mu_slice[c * n_genes + g];
                let s = ms_slice[c * n_genes + g];
                if ub_u > 0.0 && u > ub_u {
                    col_out[c] = false;
                    continue;
                }
                if ub_s > 0.0 && s > ub_s {
                    col_out[c] = false;
                    continue;
                }
            }
        };

        if parallel {
            use rayon::prelude::*;
            let mut col_outs: Vec<Vec<bool>> = (0..n_genes).map(|_| vec![false; n_cells]).collect();
            col_outs
                .par_iter_mut()
                .enumerate()
                .for_each(|(g, col)| work(g, col));
            for g in 0..n_genes {
                for c in 0..n_cells {
                    out_flat[c * n_genes + g] = col_outs[g][c];
                }
            }
        } else {
            let mut col_buf: Vec<bool> = vec![false; n_cells];
            for g in 0..n_genes {
                for c in 0..n_cells {
                    col_buf[c] = false;
                }
                work(g, &mut col_buf);
                for c in 0..n_cells {
                    out_flat[c * n_genes + g] = col_buf[c];
                }
            }
        }
    });

    let arr = ndarray::Array2::from_shape_vec((n_cells, n_genes), out_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(numpy::PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(signature = (mu, ms, weights, fit_scaling=true, parallel=true))]
pub fn per_gene_bimodality_kernel<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<'py, f64>,
    ms: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, bool>,
    fit_scaling: bool,
    parallel: bool,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    let mu_arr = mu.as_array();
    let ms_arr = ms.as_array();
    let w_arr = weights.as_array();
    let (n_cells, n_genes) = (mu_arr.shape()[0], mu_arr.shape()[1]);
    if ms_arr.shape() != [n_cells, n_genes] || w_arr.shape() != [n_cells, n_genes] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Mu, Ms, weights must have the same shape",
        ));
    }
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Mu must be C-contiguous"))?;
    let ms_slice = ms_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Ms must be C-contiguous"))?;
    let w_slice = w_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("weights must be C-contiguous"))?;

    let (pval, steady_u, steady_s) = py.allow_threads(|| {
        bimodality::per_gene_bimodality(
            mu_slice,
            ms_slice,
            w_slice,
            n_cells,
            n_genes,
            fit_scaling,
            parallel,
        )
    });
    Ok((
        numpy::PyArray1::from_vec_bound(py, pval),
        numpy::PyArray1::from_vec_bound(py, steady_u),
        numpy::PyArray1::from_vec_bound(py, steady_s),
    ))
}
