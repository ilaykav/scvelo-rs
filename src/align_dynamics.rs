pub fn compute_idx(t: &[f64], n_cells: usize, n_genes: usize, out: &mut [bool]) {
    debug_assert_eq!(t.len(), n_cells * n_genes);
    debug_assert_eq!(out.len(), n_genes);
    for g in 0..n_genes {
        let mut any_nan = false;
        for c in 0..n_cells {
            if t[c * n_genes + g].is_nan() {
                any_nan = true;
                break;
            }
        }
        out[g] = !any_nan;
    }
}

/// Mutates alpha/beta/gamma/t_ (length n_genes) and T/Tau/Tau_ (n_cells×n_genes
#[allow(clippy::too_many_arguments)]
pub fn align_total_time(
    alpha: &mut [f64],
    beta: &mut [f64],
    gamma: &mut [f64],
    t_: &mut [f64],
    big_t: &mut [f64],
    tau: &mut [f64],
    tau_under: &mut [f64],
    idx: &[bool],
    n_cells: usize,
    n_genes: usize,
    t_max: f64,
) {
    debug_assert_eq!(alpha.len(), n_genes);
    debug_assert_eq!(beta.len(), n_genes);
    debug_assert_eq!(gamma.len(), n_genes);
    debug_assert_eq!(t_.len(), n_genes);
    debug_assert_eq!(big_t.len(), n_cells * n_genes);
    debug_assert_eq!(tau.len(), n_cells * n_genes);
    debug_assert_eq!(tau_under.len(), n_cells * n_genes);
    debug_assert_eq!(idx.len(), n_genes);

    let inv_n_cells = 1.0 / n_cells as f64;

    for g in 0..n_genes {
        if !idx[g] {
            continue;
        }
        let tg_ = t_[g];

        // T_max = max(T[:, g] * (T[:, g] < t_[g]), axis=0)
        //        + max((T[:, g] - t_[g]) * (T[:, g] > t_[g]), axis=0)
        let mut on_max: f64 = 0.0; // matches np.max of zero-masked positives → 0
        let mut off_max: f64 = 0.0;
        // denom = 1 - sum((T == t_) | (T == 0)) / n_cells
        let mut steady_count: usize = 0;
        for c in 0..n_cells {
            let v = big_t[c * n_genes + g];
            if v < tg_ && v > on_max {
                on_max = v;
            }
            if v > tg_ {
                let d = v - tg_;
                if d > off_max {
                    off_max = d;
                }
            }
            if v == tg_ || v == 0.0 {
                steady_count += 1;
            }
        }
        let mut t_max_g = on_max + off_max;
        let mut denom = 1.0 - (steady_count as f64) * inv_n_cells;
        if denom == 0.0 {
            denom = 1.0;
        }
        t_max_g /= denom;
        if t_max_g == 0.0 {
            t_max_g = 1.0;
        }
        let m_g = t_max / t_max_g;
        if m_g == 1.0 || !m_g.is_finite() {
            // No-op scaling, skip the column scan to save work.
            // (m=1 means the gene is already aligned.)
            continue;
        }

        // Apply per-gene scaling: alpha/beta/gamma /= m, T/t_/Tau/Tau_ *= m.
        alpha[g] /= m_g;
        beta[g] /= m_g;
        gamma[g] /= m_g;
        t_[g] *= m_g;
        for c in 0..n_cells {
            let i = c * n_genes + g;
            big_t[i] *= m_g;
            tau[i] *= m_g;
            tau_under[i] *= m_g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_skips_nan_genes() {
        let n_cells = 5;
        let n_genes = 2;
        // gene 0: clean. gene 1: has a NaN row.
        let mut alpha = vec![1.0, 1.0];
        let mut beta = vec![2.0, 2.0];
        let mut gamma = vec![3.0, 3.0];
        let mut t_ = vec![5.0, 5.0];
        let mut big_t = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, f64::NAN, 5.0, 5.0];
        let mut tau = big_t.clone();
        let mut tau_ = big_t.clone();
        let mut idx = vec![false; n_genes];
        compute_idx(&big_t, n_cells, n_genes, &mut idx);
        assert_eq!(idx, vec![true, false]);

        align_total_time(
            &mut alpha, &mut beta, &mut gamma, &mut t_, &mut big_t, &mut tau, &mut tau_, &idx,
            n_cells, n_genes, 20.0,
        );

        // Gene 0 was rescaled — alpha changed.
        assert_ne!(alpha[0], 1.0);
        // Gene 1 was skipped — alpha unchanged.
        assert_eq!(alpha[1], 1.0);
        assert_eq!(beta[1], 2.0);
    }
}
