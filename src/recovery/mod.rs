use std::cell::Cell;

mod api;
mod internals;
mod types;

thread_local! {
    pub(super) static TRACE_ALT_T_ENABLED: Cell<bool> = const { Cell::new(false) };
}

pub use api::{fit_one_gene, initialize_one_gene};
pub use types::{GeneFitFull, Initial};
