use rayon::prelude::*;

use crate::csr::CsrView;
use crate::recovery::{fit_one_gene, initialize_one_gene, GeneFitFull, Initial};

/// Parallel-over-genes initialisation driver. Replaces the per-gene Python
pub fn initialize_all_genes(
    mu_cols: &[&[f64]],
    ms_cols: &[&[f64]],
    weights_cols: &[&[bool]],
    conn: Option<CsrView<'_>>,
    fit_scaling: bool,
    fit_steady_states: bool,
    f32_mode: bool,
    pval_steady: &[f64], // length n_genes; NaN if not provided
    steady_u: &[f64],    // length n_genes
    steady_s: &[f64],    // length n_genes
) -> Vec<Initial> {
    let n_genes = mu_cols.len();
    debug_assert_eq!(ms_cols.len(), n_genes);
    debug_assert_eq!(weights_cols.len(), n_genes);
    debug_assert_eq!(pval_steady.len(), n_genes);
    debug_assert_eq!(steady_u.len(), n_genes);
    debug_assert_eq!(steady_s.len(), n_genes);

    (0..n_genes)
        .into_par_iter()
        .map(|g| {
            initialize_one_gene(
                mu_cols[g],
                ms_cols[g],
                weights_cols[g],
                conn,
                fit_scaling,
                fit_steady_states,
                f32_mode,
                pval_steady[g],
                steady_u[g],
                steady_s[g],
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct DriverConfig {
    pub max_iter: usize,
    pub fit_scaling: bool,
    pub fit_steady_states: bool,
}

/// Parallel-over-genes EM driver. `mu_cols[g]`, `ms_cols[g]` are gene `g`'s
pub fn fit_all_genes(
    mu_cols: &[&[f64]],
    ms_cols: &[&[f64]],
    weights_cols: &[&[bool]],
    inits: &[Initial],
    conn: Option<CsrView<'_>>,
    cfg: DriverConfig,
) -> Vec<GeneFitFull> {
    let n_genes = mu_cols.len();
    debug_assert_eq!(ms_cols.len(), n_genes);
    debug_assert_eq!(weights_cols.len(), n_genes);
    debug_assert_eq!(inits.len(), n_genes);

    (0..n_genes)
        .into_par_iter()
        .map(|g| {
            fit_one_gene(
                mu_cols[g],
                ms_cols[g],
                weights_cols[g],
                conn,
                inits[g].clone(),
                cfg.max_iter,
                cfg.fit_scaling,
                cfg.fit_steady_states,
                g,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_all_genes_smoke() {
        let n_cells = 100;
        let n_genes = 3;
        let mut mu = vec![vec![0.0f64; n_cells]; n_genes];
        let mut ms = vec![vec![0.0f64; n_cells]; n_genes];
        let weights = vec![vec![true; n_cells]; n_genes];
        for g in 0..n_genes {
            let alpha = 1.0 + g as f64 * 0.5;
            let beta = 0.8;
            let gamma = 0.4 + g as f64 * 0.1;
            for i in 0..n_cells {
                let t = (i as f64) * 0.1;
                let expu = (-beta * t).exp();
                let exps = (-gamma * t).exp();
                mu[g][i] = alpha / beta * (1.0 - expu);
                let c = alpha / (gamma - beta);
                ms[g][i] = alpha / gamma * (1.0 - exps) + c * (exps - expu);
            }
        }
        let inits = (0..n_genes)
            .map(|_| Initial {
                alpha: 1.0,
                beta: 1.0,
                gamma: 0.5,
                scaling: 1.0,
                t_: 5.0,
                u0_: 0.5,
                s0_: 0.5,
                std_u: 1.0,
                std_s: 1.0,
                steady_state_ratio: Some(0.5),
                f32_mode: false,
            })
            .collect::<Vec<_>>();

        let mu_refs: Vec<&[f64]> = mu.iter().map(|v| v.as_slice()).collect();
        let ms_refs: Vec<&[f64]> = ms.iter().map(|v| v.as_slice()).collect();
        let w_refs: Vec<&[bool]> = weights.iter().map(|v| v.as_slice()).collect();

        let fits = fit_all_genes(
            &mu_refs,
            &ms_refs,
            &w_refs,
            &inits,
            None,
            DriverConfig {
                max_iter: 10,
                fit_scaling: true,
                fit_steady_states: true,
            },
        );
        assert_eq!(fits.len(), n_genes);
        for f in &fits {
            assert!(!f.alpha.is_nan(), "got NaN");
        }
    }
}
