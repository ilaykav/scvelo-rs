use pyo3::prelude::*;

mod align_dynamics;
mod bimodality;
mod bindings;
mod csr;
mod diff_kinetic;
mod divergence;
mod dynamics;
mod em;
mod mse;
mod nelder_mead;
mod numpy_compat;
mod projection;
mod recovery;
mod scanpy_replacement;
mod velocity;
mod velocity_graph;

#[pymodule]
fn _scvelo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
