mod assign;
mod increments;

pub use assign::{assign_timepoints, assign_timepoints_dtyped};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssignmentMode {
    None,
    Projection,
    FullProjection,
    PartialProjection,
}

pub struct AssignTimepoints {
    pub t: Vec<f64>,
    pub tau: Vec<f64>,
    pub tau_: Vec<f64>,
    pub tau_unmasked: Vec<f64>,
    pub tau__unmasked: Vec<f64>,
    pub o: Vec<u8>,
}

#[inline]
pub(super) fn f32q(x: f64) -> f64 {
    (x as f32) as f64
}
