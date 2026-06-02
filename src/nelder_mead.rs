#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub xatol: f64,
    pub fatol: f64,
    pub maxiter: usize,
    pub maxfev: usize,
}

impl Settings {
    pub fn scvelo_default(n_dim: usize, max_iter: usize) -> Self {
        // scvelo uses `options={'maxiter': int(max_iter/5)}` and scipy's default xatol/fatol.
        Self {
            xatol: 1e-4,
            fatol: 1e-4,
            maxiter: max_iter / 5,
            maxfev: 200 * n_dim, // scipy default
        }
    }

    /// scvelo's `fit_rates` and `fit_t_and_rates` pass `tol=1e-2`. scipy then sets
    /// `xatol = fatol = tol` for Nelder-Mead.
    pub fn scvelo_with_tol(n_dim: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            xatol: tol,
            fatol: tol,
            maxiter: max_iter / 5,
            maxfev: 200 * n_dim,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Result {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Run Nelder-Mead on `f` starting from `x0`. Mirrors scipy's
pub fn minimize<F, C>(f: &mut F, x0: &[f64], cfg: &Settings, mut callback: C) -> Result
where
    F: FnMut(&[f64]) -> f64,
    C: FnMut(&[f64], f64),
{
    let n = x0.len();
    let nonzdelt = 0.05_f64;
    let zdelt = 0.00025_f64;

    // Build initial simplex: shape (n+1, n).
    let mut sim = vec![vec![0.0f64; n]; n + 1];
    sim[0].copy_from_slice(x0);
    for k in 0..n {
        let mut xk = x0.to_vec();
        if xk[k] != 0.0 {
            xk[k] *= 1.0 + nonzdelt;
        } else {
            xk[k] = zdelt;
        }
        sim[k + 1] = xk;
    }

    // Evaluate f at every simplex vertex.
    let mut fsim = vec![0.0f64; n + 1];
    let mut nfev = 0usize;
    for k in 0..(n + 1) {
        fsim[k] = f(&sim[k]);
        nfev += 1;
    }

    // Sort simplex by f. Stable tie-break (preserves input order on ties) -
    // matches scipy NM for most patterns. The 4 outlier genes (Dapk1, Erc2,
    // Prox1, Arid4b) hit a numpy SIMD-quicksort tie pattern where reverse-index
    // would match scvelo, but reverse-index breaks 17 other genes that the
    // stable rule gets right. Per-pattern tie-matching (porting numpy's exact
    // sort) is the only path to closing all 4 from Rust without numpy callback.
    let mut order: Vec<usize> = (0..(n + 1)).collect();
    order.sort_by(|&a, &b| {
        fsim[a]
            .partial_cmp(&fsim[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sim_sorted: Vec<Vec<f64>> = order.iter().map(|&i| sim[i].clone()).collect();
    let fsim_sorted: Vec<f64> = order.iter().map(|&i| fsim[i]).collect();
    let mut sim = sim_sorted;
    let mut fsim = fsim_sorted;

    let rho = 1.0_f64;
    let chi = 2.0_f64;
    let psi = 0.5_f64;
    let sigma = 0.5_f64;

    // scipy `_minimize_neldermead` initializes `iterations = 1` BEFORE the
    // `while iterations < maxiter` loop, so with `maxiter=2` the body runs
    // exactly ONCE. Mirror that: start `nit=1` so the loop terminates at
    // exactly the same iteration count as scipy. Without this, scvelo's
    // default `max_iter=10 → maxiter=2` would run 2 NM iters in rust vs 1
    // in scipy, causing major drift on the EM trajectory.
    let mut nit = 1usize;
    let mut converged = false;

    // Per-iter trace for diff against scipy. Gated on env var so it's a no-op
    // in production but invaluable for debugging NM trajectory divergence.
    let trace_nm = std::env::var("SCVELORS_TRACE_NM").is_ok();
    if trace_nm {
        eprintln!("[nm-trace] init: n={}, x0={:?}", n, x0);
        eprintln!("[nm-trace] sim_sorted={:?}", sim);
        eprintln!("[nm-trace] fsim_sorted={:?}", fsim);
    }

    while nit < cfg.maxiter && nfev < cfg.maxfev {
        // Convergence check (scipy: AFTER the iteration body in scipy, but here we check first
        // to match scipy's `while iterations < maxiter` loop layout - scipy checks
        // tolerance at top of each iteration).
        if simplex_converged(&sim, &fsim, cfg.xatol, cfg.fatol) {
            converged = true;
            break;
        }

        // Centroid of all but the worst (last after sort).
        let mut xbar = vec![0.0f64; n];
        for k in 0..n {
            for j in 0..n {
                xbar[j] += sim[k][j];
            }
        }
        for j in 0..n {
            xbar[j] /= n as f64;
        }

        // Reflect.
        let xr: Vec<f64> = (0..n)
            .map(|j| (1.0 + rho) * xbar[j] - rho * sim[n][j])
            .collect();
        let fxr = f(&xr);
        nfev += 1;

        let mut doshrink = false;

        if fxr < fsim[0] {
            // Expand.
            let xe: Vec<f64> = (0..n)
                .map(|j| (1.0 + rho * chi) * xbar[j] - rho * chi * sim[n][j])
                .collect();
            let fxe = f(&xe);
            nfev += 1;
            if fxe < fxr {
                sim[n] = xe;
                fsim[n] = fxe;
            } else {
                sim[n] = xr;
                fsim[n] = fxr;
            }
        } else if fxr < fsim[n - 1] {
            sim[n] = xr;
            fsim[n] = fxr;
        } else {
            // Contraction.
            if fxr < fsim[n] {
                let xc: Vec<f64> = (0..n)
                    .map(|j| (1.0 + psi * rho) * xbar[j] - psi * rho * sim[n][j])
                    .collect();
                let fxc = f(&xc);
                nfev += 1;
                if fxc <= fxr {
                    sim[n] = xc;
                    fsim[n] = fxc;
                } else {
                    doshrink = true;
                }
            } else {
                let xcc: Vec<f64> = (0..n)
                    .map(|j| (1.0 - psi) * xbar[j] + psi * sim[n][j])
                    .collect();
                let fxcc = f(&xcc);
                nfev += 1;
                if fxcc < fsim[n] {
                    sim[n] = xcc;
                    fsim[n] = fxcc;
                } else {
                    doshrink = true;
                }
            }

            if doshrink {
                for k in 1..(n + 1) {
                    let new_pt: Vec<f64> = (0..n)
                        .map(|j| sim[0][j] + sigma * (sim[k][j] - sim[0][j]))
                        .collect();
                    sim[k] = new_pt;
                    fsim[k] = f(&sim[k]);
                    nfev += 1;
                }
            }
        }

        // Re-sort simplex (stable on ties).
        let mut order: Vec<usize> = (0..(n + 1)).collect();
        order.sort_by(|&a, &b| {
            fsim[a]
                .partial_cmp(&fsim[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sim = order.iter().map(|&i| sim[i].clone()).collect();
        fsim = order.iter().map(|&i| fsim[i]).collect();

        if trace_nm {
            eprintln!(
                "[nm-trace] iter {}: best_x={:?} best_f={} fsim={:?}",
                nit, sim[0], fsim[0], fsim
            );
        }

        // scipy invokes `callback(xk)` after each iteration with the best vertex.
        callback(&sim[0], fsim[0]);

        nit += 1;
    }

    if trace_nm {
        eprintln!(
            "[nm-trace] DONE nit={} converged={} final_x={:?} final_f={}",
            nit, converged, sim[0], fsim[0]
        );
    }

    Result {
        x: sim[0].clone(),
        fun: fsim[0],
        nit,
        nfev,
        converged,
    }
}

fn simplex_converged(sim: &[Vec<f64>], fsim: &[f64], xatol: f64, fatol: f64) -> bool {
    let n_plus_1 = sim.len();
    if n_plus_1 < 2 {
        return true;
    }
    // np.max(|fsim[1:] - fsim[0]|) <= fatol
    let f0 = fsim[0];
    let mut max_df = 0.0f64;
    for k in 1..n_plus_1 {
        let v = (fsim[k] - f0).abs();
        if v > max_df {
            max_df = v;
        }
    }
    if max_df > fatol {
        return false;
    }
    // np.max(np.max(|sim[1:] - sim[0]|, axis=-1)) <= xatol  - i.e. max over all vertex coords
    let n = sim[0].len();
    let mut max_dx = 0.0f64;
    for k in 1..n_plus_1 {
        for j in 0..n {
            let v = (sim[k][j] - sim[0][j]).abs();
            if v > max_dx {
                max_dx = v;
            }
        }
    }
    max_dx <= xatol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_minimum() {
        // f(x, y) = (1 - x)^2 + 100 (y - x^2)^2; min at (1, 1) with f = 0.
        let mut f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let cfg = Settings {
            xatol: 1e-6,
            fatol: 1e-6,
            maxiter: 200,
            maxfev: 1000,
        };
        let res = minimize(&mut f, &[0.0, 0.0], &cfg, |_, _| {});
        assert!(
            res.converged,
            "should converge on Rosenbrock within 200 iters"
        );
        assert!((res.x[0] - 1.0).abs() < 1e-3, "x[0]={}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn quadratic_immediate() {
        let mut f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let cfg = Settings {
            xatol: 1e-8,
            fatol: 1e-8,
            maxiter: 200,
            maxfev: 1000,
        };
        let res = minimize(&mut f, &[3.0, 4.0], &cfg, |_, _| {});
        assert!(res.converged);
        assert!(res.x[0].abs() < 1e-3);
        assert!(res.x[1].abs() < 1e-3);
    }

    #[test]
    fn callback_sees_best_per_iter() {
        // Track every callback call. Sequence should be monotonically non-increasing.
        let mut f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] + 2.0).powi(2);
        let cfg = Settings {
            xatol: 1e-6,
            fatol: 1e-6,
            maxiter: 50,
            maxfev: 1000,
        };
        let mut history: Vec<f64> = Vec::new();
        let cb = |_x: &[f64], fx: f64| {
            history.push(fx);
        };
        let _ = minimize(&mut f, &[0.0, 0.0], &cfg, cb);
        assert!(
            !history.is_empty(),
            "callback should be invoked at least once"
        );
        for w in history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "callback values should be non-increasing"
            );
        }
    }
}
