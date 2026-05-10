use crate::numpy_compat::pairwise_sum;

#[derive(Clone, Debug)]
pub struct Initial {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub scaling: f64,
    pub t_: f64,
    pub u0_: f64,
    pub s0_: f64,
    pub std_u: f64,
    pub std_s: f64,
    pub steady_state_ratio: Option<f64>,
    pub f32_mode: bool,
}

#[derive(Clone, Debug)]
pub struct GeneFitFull {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub t_: f64,
    pub scaling: f64,
    pub likelihood: f64,
    pub variance: f64,
    pub fit_t: Vec<f64>,
    pub fit_tau: Vec<f64>,
    pub fit_tau_: Vec<f64>,
}

impl GeneFitFull {
    pub fn nan(n_cells: usize) -> Self {
        Self {
            alpha: f64::NAN,
            beta: f64::NAN,
            gamma: f64::NAN,
            t_: f64::NAN,
            scaling: f64::NAN,
            likelihood: f64::NAN,
            variance: f64::NAN,
            fit_t: vec![f64::NAN; n_cells],
            fit_tau: vec![f64::NAN; n_cells],
            fit_tau_: vec![f64::NAN; n_cells],
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct State {
    pub(super) alpha: f64,
    pub(super) beta: f64,
    pub(super) gamma: f64,
    pub(super) scaling: f64,
    pub(super) t_: f64,
    pub(super) u0_: f64,
    pub(super) s0_: f64,
    pub(super) std_u: f64,
    pub(super) std_s: f64,
    pub(super) steady_state_ratio: Option<f64>,
    pub(super) last_loss: f64,
    pub(super) cached_t: Vec<f64>,
    pub(super) cached_tau: Vec<f64>,
    pub(super) cached_tau_: Vec<f64>,
    pub(super) cached_o: Vec<u8>,
    pub(super) f32_mode: bool,
}

impl State {
    pub(super) fn from_initial(init: &Initial, n_cells: usize) -> Self {
        Self {
            alpha: init.alpha,
            beta: init.beta,
            gamma: init.gamma,
            scaling: init.scaling,
            t_: init.t_,
            u0_: init.u0_,
            s0_: init.s0_,
            std_u: init.std_u,
            std_s: init.std_s,
            steady_state_ratio: init.steady_state_ratio,
            last_loss: f64::NAN,
            cached_t: vec![0.0; n_cells],
            cached_tau: vec![0.0; n_cells],
            cached_tau_: vec![0.0; n_cells],
            cached_o: vec![0u8; n_cells],
            f32_mode: init.f32_mode,
        }
    }
}

#[inline]
pub(super) fn f32q(x: f64) -> f64 {
    (x as f32) as f64
}

pub(super) fn std_pop(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    let n = arr.len() as f64;
    let mean = pairwise_sum(arr) / n;
    let sq: Vec<f64> = arr.iter().map(|x| (x - mean) * (x - mean)).collect();
    let var = pairwise_sum(&sq) / n;
    var.sqrt()
}

#[inline]
pub(super) fn mean_masked(arr: &[f64], mask: &[bool]) -> f64 {
    let subset: Vec<f64> = arr
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();
    if subset.is_empty() {
        return 0.0;
    }
    pairwise_sum(&subset) / subset.len() as f64
}

#[inline]
pub(super) fn mean_masked_or(arr: &[f64], a: &[bool], b: &[bool]) -> f64 {
    let subset: Vec<f64> = arr
        .iter()
        .zip(a.iter())
        .zip(b.iter())
        .filter_map(|((&v, &x), &y)| if x || y { Some(v) } else { None })
        .collect();
    if subset.is_empty() {
        return 0.0;
    }
    pairwise_sum(&subset) / subset.len() as f64
}
