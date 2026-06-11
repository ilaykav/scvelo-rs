use pyo3::prelude::*;

use mimalloc::MiMalloc;

/// Per-thread-heap allocator. See the rationale in Cargo.toml: the NM inner
/// loop is allocation-heavy, and the system allocator serialises that churn
/// across Rayon threads. mimalloc removes the contention.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

/// Optional override of the global Rayon pool size via `SCVELORS_NUM_THREADS`.
///
/// With mimalloc removing the allocator contention, Rayon's own default
/// (logical core count) scales well for the per-gene EM - on a
/// 16-physical/32-logical box, 32 threads beat 16 (~16s vs ~19s on the
/// `recover_dynamics` kernel), so we do NOT cap by default. The env var lets
/// power users tune it: pin a single thread for bit-exact debugging, or cap
/// it on a shared/over-subscribed node. Unset -> Rayon default (all logical).
fn init_thread_pool() {
    if let Some(n) = std::env::var("SCVELORS_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        // `build_global` errors if a pool already exists (e.g. a second import
        // in the same process). That's benign - the first call wins.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
}

#[pymodule]
fn _scvelo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_thread_pool();
    bindings::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
