use rayon::prelude::*;

pub struct VelocityFit {
    pub gamma: Vec<f64>,
    pub offset: Vec<f64>,
    pub r2: Vec<f64>,
    pub residual: Vec<f64>,        // (n_cells * n_genes) row-major
    pub velocity_genes: Vec<bool>, // (n_genes,)
}

/// Per-gene linear regression `Mu ~ gamma * Ms (+ offset)` with optional
/// extreme-quantile trimming on the (Mu, Ms) score `nd`.
pub fn fit_deterministic(
    ms: &[f64],
    mu: &[f64],
    n_cells: usize,
    n_genes: usize,
    fit_offset: bool,
    min_r2: f64,
    min_ratio: f64,
    constrain_ratio: Option<(f64, f64)>,
    percentile: Option<(f64, Option<f64>)>,
) -> VelocityFit {
    let mut gamma = vec![0.0f64; n_genes];
    let mut offset = vec![0.0f64; n_genes];

    // Per-gene fit. Embarrassingly parallel.
    let fits: Vec<(f64, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| fit_one_gene(ms, mu, n_cells, n_genes, g, fit_offset, percentile))
        .collect();

    for (g, (gm, off)) in fits.into_iter().enumerate() {
        let mut gm = gm;
        if let Some((lo, hi)) = constrain_ratio {
            gm = gm.clamp(lo, hi);
        }
        gamma[g] = gm;
        offset[g] = off;
    }

    // residual = Mu - gamma * Ms (- offset if fit_offset)
    let mut residual = vec![0.0f64; n_cells * n_genes];
    residual
        .par_chunks_mut(n_genes)
        .enumerate()
        .for_each(|(i, row)| {
            for g in 0..n_genes {
                let idx = i * n_genes + g;
                let mut r = mu[idx] - gamma[g] * ms[idx];
                if fit_offset {
                    r -= offset[g];
                }
                row[g] = r;
            }
        });

    // R^2 per gene: 1 - SS_res / SS_tot
    let mu_mean: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut s = 0.0;
            for i in 0..n_cells {
                s += mu[i * n_genes + g];
            }
            s / n_cells as f64
        })
        .collect();

    let r2: Vec<f64> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            let m = mu_mean[g];
            for i in 0..n_cells {
                let r = residual[i * n_genes + g];
                ss_res += r * r;
                let d = mu[i * n_genes + g] - m;
                ss_tot += d * d;
            }
            if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            }
        })
        .collect();

    // velocity_genes = (r2 > min_r2) & (gamma > min_ratio) &
    //                  (max(Ms > 0, axis=0) > 0) & (max(Mu > 0, axis=0) > 0)
    let velocity_genes: Vec<bool> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut ms_any = false;
            let mut mu_any = false;
            for i in 0..n_cells {
                if ms[i * n_genes + g] > 0.0 {
                    ms_any = true;
                }
                if mu[i * n_genes + g] > 0.0 {
                    mu_any = true;
                }
                if ms_any && mu_any {
                    break;
                }
            }
            r2[g] > min_r2 && gamma[g] > min_ratio && ms_any && mu_any
        })
        .collect();

    VelocityFit {
        gamma,
        offset,
        r2,
        residual,
        velocity_genes,
    }
}

fn fit_one_gene(
    ms: &[f64],
    mu: &[f64],
    n_cells: usize,
    n_genes: usize,
    g: usize,
    fit_offset: bool,
    percentile: Option<(f64, Option<f64>)>,
) -> (f64, f64) {
    // Build the cell mask for this gene.
    let mask = if let Some((lo_pct, hi_pct)) = percentile {
        gene_trim_mask(ms, mu, n_cells, n_genes, g, lo_pct, hi_pct)
    } else {
        vec![true; n_cells]
    };

    let mut xx = 0.0f64;
    let mut xy = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut n_kept = 0usize;
    for i in 0..n_cells {
        if !mask[i] {
            continue;
        }
        let x = ms[i * n_genes + g];
        let y = mu[i * n_genes + g];
        xx += x * x;
        xy += x * y;
        sx += x;
        sy += y;
        n_kept += 1;
    }
    let n = n_kept as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    if fit_offset {
        let xb = sx / n;
        let yb = sy / n;
        let denom = xx / n - xb * xb;
        if denom == 0.0 || !denom.is_finite() {
            return (0.0, 0.0);
        }
        let mut coef = (xy / n - xb * yb) / denom;
        let mut intercept = yb - coef * xb;
        if intercept < 0.0 {
            // positive_intercept: clip negative offset to zero and refit slope.
            intercept = 0.0;
            coef = if xx > 0.0 { xy / xx } else { 0.0 };
        }
        if !coef.is_finite() {
            return (0.0, intercept);
        }
        (coef, intercept)
    } else {
        if xx == 0.0 {
            return (0.0, 0.0);
        }
        let coef = xy / xx;
        if !coef.is_finite() {
            return (0.0, 0.0);
        }
        (coef, 0.0)
    }
}

/// Compute the per-gene extreme-quantile trim mask on `nd = Ms/max(Ms) + Mu/max(Mu)`.
fn gene_trim_mask(
    ms: &[f64],
    mu: &[f64],
    n_cells: usize,
    n_genes: usize,
    g: usize,
    lo_pct: f64,
    hi_pct: Option<f64>,
) -> Vec<bool> {
    // 1. Per-gene max of |Ms| and |Mu|, clipped to 1e-3.
    let mut ms_max = 0.0f64;
    let mut mu_max = 0.0f64;
    for i in 0..n_cells {
        let v = ms[i * n_genes + g];
        if v > ms_max {
            ms_max = v;
        }
        let v = mu[i * n_genes + g];
        if v > mu_max {
            mu_max = v;
        }
    }
    let ms_max = ms_max.max(1e-3);
    let mu_max = mu_max.max(1e-3);

    // 2. nd[c] = Ms[c]/ms_max + Mu[c]/mu_max for this gene.
    let mut nd = vec![0.0f64; n_cells];
    for i in 0..n_cells {
        nd[i] = ms[i * n_genes + g] / ms_max + mu[i * n_genes + g] / mu_max;
    }

    // 3. Per-gene percentile threshold(s).
    let mut sorted = nd.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_bound = numpy_percentile(&sorted, lo_pct);
    let hi_bound = hi_pct.map(|p| numpy_percentile(&sorted, p));

    // 4. Mask:
    //    range:   keep cells in the lo/hi extremes (nd <= lo OR nd >= hi)
    //    single:  keep cells with nd >= lo
    let mut mask = vec![false; n_cells];
    for i in 0..n_cells {
        let v = nd[i];
        let kept = if let Some(hi) = hi_bound {
            v <= lo_bound || v >= hi
        } else {
            v >= lo_bound
        };
        mask[i] = kept;
    }
    mask
}

/// numpy `np.percentile(a, q, method='linear')` with a pre-sorted input.
fn numpy_percentile(sorted: &[f64], q_pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = q_pct / 100.0 * (n - 1) as f64;
    let i = h.floor() as usize;
    let frac = h - i as f64;
    if i + 1 >= n {
        return sorted[n - 1];
    }
    sorted[i] + frac * (sorted[i + 1] - sorted[i])
}
