use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod alignment;
pub mod debug;
pub mod diff_kinetic;
pub mod projection;
pub mod recovery;
pub mod scanpy;
pub mod velocity;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recovery::recover_dynamics_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(recovery::initialize_all_genes_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(recovery::per_gene_weights_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(recovery::per_gene_bimodality_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(
        projection::splicing_dynamics_eval_kernel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(projection::compute_dynamics_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(projection::project_to_curve_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(projection::assign_timepoints_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(alignment::align_dynamics_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(velocity::velocity_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(velocity::velocity_graph_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(scanpy::pca_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(scanpy::knn_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(diff_kinetic::diff_kinetic_test_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_exp, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_ln, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_splicing_solution, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_tau_inv, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_pairwise_sum, m)?)?;
    m.add_function(wrap_pyfunction!(debug::_debug_tau_inv_intermediates, m)?)?;
    Ok(())
}
