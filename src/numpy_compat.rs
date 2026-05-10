pub fn pairwise_sum(arr: &[f64]) -> f64 {
    let n = arr.len();
    if n == 0 {
        return 0.0;
    }
    // Numpy's base case: PW_BLOCKSIZE=128, sequential left-to-right within.
    if n < 8 {
        let mut acc = arr[0];
        for &x in &arr[1..] {
            acc += x;
        }
        return acc;
    }
    if n <= 128 {
        // Numpy's "small array" path: unrolled by 8 with separate accumulators
        // (see numpy/core/src/umath/loops_arithm_fp.dispatch.c.src).
        // We mirror by accumulating in 8 partial sums, then combining.
        let mut r = [0.0f64; 8];
        let mut i = 0;
        while i + 8 <= n {
            r[0] += arr[i];
            r[1] += arr[i + 1];
            r[2] += arr[i + 2];
            r[3] += arr[i + 3];
            r[4] += arr[i + 4];
            r[5] += arr[i + 5];
            r[6] += arr[i + 6];
            r[7] += arr[i + 7];
            i += 8;
        }
        let mut acc = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            acc += arr[i];
            i += 1;
        }
        return acc;
    }
    // numpy splits in halves rounded to multiples of 8 (block-aligned).
    let mid = (n / 2) - ((n / 2) % 8);
    pairwise_sum(&arr[..mid]) + pairwise_sum(&arr[mid..])
}

/// Numpy-compatible mean for an f64 slice. Matches `np.mean(arr)` for 1D arrays.
pub fn mean(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(arr) / arr.len() as f64
}

/// Numpy-compatible nansum: ignores NaNs in summation. Pairwise.
pub fn nansum(arr: &[f64]) -> f64 {
    let n = arr.len();
    if n == 0 {
        return 0.0;
    }
    if n <= 8 {
        let mut acc = 0.0;
        for &x in arr {
            if !x.is_nan() {
                acc += x;
            }
        }
        return acc;
    }
    let mid = n / 2;
    nansum(&arr[..mid]) + nansum(&arr[mid..])
}

/// Numpy-compatible nanmean.
pub fn nanmean(arr: &[f64]) -> f64 {
    let mut count = 0usize;
    for &x in arr {
        if !x.is_nan() {
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    nansum(arr) / count as f64
}

/// Numpy `np.percentile(arr, q, method="linear")` for a 1-D slice.
pub fn percentile_sorted(sorted: &[f64], q_pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
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

/// `np.argmin(arr, axis=0)` for a column-major (k, n) view supplied as an
pub fn argmin_axis0(rows: &[&[f64]], out: &mut [usize]) {
    let k = rows.len();
    debug_assert!(k > 0);
    let n = rows[0].len();
    debug_assert_eq!(out.len(), n);
    for c in 0..n {
        let mut best_v = rows[0][c];
        let mut best_i = 0usize;
        for r in 1..k {
            let v = rows[r][c];
            if v < best_v {
                best_v = v;
                best_i = r;
            }
        }
        out[c] = best_i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_small_matches_naive() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pairwise_sum(&arr), 15.0);
    }

    #[test]
    fn pairwise_sum_large_stable_for_uniform() {
        let arr = vec![1e-10; 10000];
        let s = pairwise_sum(&arr);
        assert!((s - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn mean_basic() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&arr), 2.5);
    }

    #[test]
    fn percentile_99_5_of_arange() {
        // For sorted [0..1000], q=99.5: h = 0.995 * 999 = 994.005
        // i = 994, frac = 0.005, result = 994 + 0.005 * 1 = 994.005
        let arr: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let p = percentile_sorted(&arr, 99.5);
        assert!((p - 994.005).abs() < 1e-12);
    }

    #[test]
    fn argmin_axis0_first_wins_on_tie() {
        let r0 = [1.0, 2.0, 3.0];
        let r1 = [1.0, 1.0, 4.0];
        let r2 = [3.0, 5.0, 0.0];
        let rows: [&[f64]; 3] = [&r0, &r1, &r2];
        let mut out = [0usize; 3];
        argmin_axis0(&rows, &mut out);
        assert_eq!(out, [0, 1, 2]);
    }
}
