use rayon::prelude::*;

/// Result of the per-gene bimodality test: (pval, peak_low, peak_high).
/// `pval = NaN` indicates the test could not run (n_w < 3).
#[derive(Clone, Copy, Debug)]
pub struct Bimodality {
    pub pval: f64,
    pub mean_low: f64,
    pub mean_high: f64,
}

#[inline]
fn percentile_99_9(sorted: &[f64]) -> f64 {
    // numpy linear interpolation, q=99.9.
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = 0.999 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

#[inline]
fn sample_std(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

#[inline]
fn pop_std(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// Standard normal survival function: 1 - Phi(x) = 0.5 * erfc(x / sqrt(2)).
#[inline]
fn norm_sf(x: f64) -> f64 {
    0.5 * libm::erfc(x / std::f64::consts::SQRT_2)
}

/// 1-D bimodality test via Gaussian KDE on a 30-bin grid.
pub fn test_bimodality(x: &[f64]) -> Bimodality {
    let bins: usize = 30;
    let n = x.len();
    if n < 3 {
        return Bimodality {
            pval: f64::NAN,
            mean_low: 0.0,
            mean_high: 0.0,
        };
    }

    // grid: linspace(lb, ub_eff, bins). Upper bound is max(x) in the typical
    // case, falling back to the 99.9-percentile only when degenerate.
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lb = sorted[0];
    let ub_q999 = percentile_99_9(&sorted);
    let max_x = *sorted.last().unwrap();
    let ub_eff = if ub_q999 <= lb { ub_q999 } else { max_x };
    if !(ub_eff > lb) {
        // Degenerate (all x equal). pval=1, means=lb.
        return Bimodality {
            pval: 1.0,
            mean_low: lb,
            mean_high: lb,
        };
    }

    let mut grid = vec![0.0f64; bins];
    let step = (ub_eff - lb) / (bins as f64 - 1.0);
    for i in 0..bins {
        grid[i] = lb + step * i as f64;
    }

    // gaussian_kde with Scott's rule (1-D): factor = n^(-1/5); h = factor * sample_std.
    let factor = (n as f64).powf(-1.0 / 5.0);
    let std_x = sample_std(x);
    if std_x == 0.0 {
        return Bimodality {
            pval: 1.0,
            mean_low: lb,
            mean_high: lb,
        };
    }
    let h = factor * std_x;
    let h2 = h * h;
    let inv_norm = 1.0 / (n as f64 * h * (2.0 * std::f64::consts::PI).sqrt());

    // Evaluate pdf(grid_i) = sum_j exp(-(grid_i - x_j)^2 / (2 h^2)) / (n*h*sqrt(2pi)).
    let mut pdf = vec![0.0f64; bins];
    for (i, &g) in grid.iter().enumerate() {
        let mut sum = 0.0f64;
        for &xv in x.iter() {
            let d = g - xv;
            sum += (-(d * d) / (2.0 * h2)).exp();
        }
        pdf[i] = sum * inv_norm;
    }

    // idx = bins/2 - 2; idx += argmin(pdf[idx:idx+4])
    let mut idx = (bins / 2).saturating_sub(2);
    let end = (idx + 4).min(bins);
    let mut min_v = f64::INFINITY;
    let mut min_off = 0usize;
    for k in idx..end {
        if pdf[k] < min_v {
            min_v = pdf[k];
            min_off = k - idx;
        }
    }
    idx += min_off;

    // peak_0 = argmax(pdf[:idx])
    let mut peak_0 = 0usize;
    let mut max_v = f64::NEG_INFINITY;
    for k in 0..idx {
        if pdf[k] > max_v {
            max_v = pdf[k];
            peak_0 = k;
        }
    }

    // peak_1 = argmax(pdf[idx:])
    let mut peak_1 = 0usize; // index relative to idx
    let mut max_v1 = f64::NEG_INFINITY;
    for k in 0..(bins - idx) {
        if pdf[idx + k] > max_v1 {
            max_v1 = pdf[idx + k];
            peak_1 = k;
        }
    }

    let kde_peak = pdf[idx + peak_1];
    let kde_mid_sum: f64 = pdf[idx..].iter().sum();
    let kde_mid = kde_mid_sum / (bins - idx) as f64;

    let std_pdf = pop_std(&pdf);
    let denom = (std_pdf / (bins as f64).sqrt()).max(1.0);
    let t_stat = (kde_peak - kde_mid) / denom;
    let pval = norm_sf(t_stat);

    // means[0] = (grid[:idx][peak_0] + grid[:idx][min(peak_0+1, idx-1)]) / 2
    let p0_next = (peak_0 + 1).min(idx.saturating_sub(1).max(peak_0));
    let mean_low = (grid[peak_0] + grid[p0_next]) / 2.0;

    // means[1] = (grid[idx:][peak_1] + grid[idx:][min(peak_1+1, len-1)]) / 2
    let g1_len = bins - idx;
    let p1_next = (peak_1 + 1).min(g1_len.saturating_sub(1).max(peak_1));
    let mean_high = (grid[idx + peak_1] + grid[idx + p1_next]) / 2.0;

    Bimodality {
        pval,
        mean_low,
        mean_high,
    }
}

/// Per-gene bimodality on (Mu_sub, Ms_sub, weights). Mirrors
pub fn per_gene_bimodality(
    mu: &[f64], // C-contiguous (n_cells, n_genes)
    ms: &[f64],
    weights: &[bool], // (n_cells, n_genes)
    n_cells: usize,
    n_genes: usize,
    fit_scaling: bool,
    parallel: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pval = vec![f64::NAN; n_genes];
    let mut steady_u = vec![0.0f64; n_genes];
    let mut steady_s = vec![0.0f64; n_genes];

    let process_one = |g: usize| -> (f64, f64, f64) {
        // Gather weighted u, s.
        let mut u_w: Vec<f64> = Vec::with_capacity(n_cells / 4);
        let mut s_w: Vec<f64> = Vec::with_capacity(n_cells / 4);
        for c in 0..n_cells {
            if weights[c * n_genes + g] {
                u_w.push(mu[c * n_genes + g]);
                s_w.push(ms[c * n_genes + g]);
            }
        }
        if u_w.len() <= 2 {
            return (f64::NAN, 0.0, 0.0);
        }
        // scaling = std_u / std_s if fit_scaling else 1.
        let scaling = if fit_scaling {
            let std_u = pop_std(&u_w);
            let std_s = pop_std(&s_w);
            if std_u == 0.0 || std_s == 0.0 {
                1.0
            } else {
                std_u / std_s
            }
        } else {
            1.0
        };
        let u_w_scaled: Vec<f64> = u_w.iter().map(|x| x / scaling).collect();
        let bu = test_bimodality(&u_w_scaled);
        let bs = test_bimodality(&s_w);
        if bu.pval.is_nan() || bs.pval.is_nan() {
            return (f64::NAN, 0.0, 0.0);
        }
        (bu.pval.max(bs.pval), bu.mean_high, bs.mean_high)
    };

    if parallel {
        let triples: Vec<(f64, f64, f64)> = (0..n_genes).into_par_iter().map(process_one).collect();
        for (g, (p, u, s)) in triples.into_iter().enumerate() {
            pval[g] = p;
            steady_u[g] = u;
            steady_s[g] = s;
        }
    } else {
        for g in 0..n_genes {
            let (p, u, s) = process_one(g);
            pval[g] = p;
            steady_u[g] = u;
            steady_s[g] = s;
        }
    }
    (pval, steady_u, steady_s)
}
