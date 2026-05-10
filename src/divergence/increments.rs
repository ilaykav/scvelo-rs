use crate::numpy_compat::percentile_sorted;

pub fn adjust_increments_single_arr(arr: &[f64]) -> Vec<f64> {
    if arr.is_empty() {
        return Vec::new();
    }
    let mut new_arr = arr.to_vec();
    let mut sorted = arr.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau = diff_with_prepend_zero(&sorted);
    let mut dtau_sorted = dtau.clone();
    dtau_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ub = 3.0 * percentile_sorted(&dtau_sorted, 99.5);
    apply_increments_subtraction(&sorted, &dtau, ub, arr, &mut new_arr);
    new_arr
}

pub fn adjust_increments_paired_arr(tau: &[f64], tau_: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if tau.is_empty() && tau_.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut tau_new = tau.to_vec();
    let mut tau_new_ = tau_.to_vec();

    let mut tau_sorted = tau.to_vec();
    tau_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau = diff_with_prepend_zero(&tau_sorted);

    let mut tau_sorted_neg = tau_.to_vec();
    tau_sorted_neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau_ = diff_with_prepend_zero(&tau_sorted_neg);

    let mut combined: Vec<f64> = Vec::with_capacity(dtau.len() + dtau_.len());
    combined.extend_from_slice(&dtau);
    combined.extend_from_slice(&dtau_);
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ub = 3.0 * percentile_sorted(&combined, 99.5);

    apply_increments_subtraction(&tau_sorted_neg, &dtau_, ub, tau_, &mut tau_new_);
    apply_increments_subtraction(&tau_sorted, &dtau, ub, tau, &mut tau_new);

    (tau_new, tau_new_)
}

pub(super) fn adjust_increments_paired(tau: &mut [f64], tau_: &mut [f64], o: &[u8]) {
    let n = o.len();
    let mut idx_on = Vec::with_capacity(n);
    let mut idx_off = Vec::with_capacity(n);
    for i in 0..n {
        match o[i] {
            1 => idx_on.push(i),
            0 => idx_off.push(i),
            _ => {}
        }
    }
    let tau_on_orig: Vec<f64> = idx_on.iter().map(|&i| tau[i]).collect();
    let tau_off_orig: Vec<f64> = idx_off.iter().map(|&i| tau_[i]).collect();
    let mut tau_on_new = tau_on_orig.clone();
    let mut tau_off_new = tau_off_orig.clone();

    let mut tau_on_sorted = tau_on_orig.clone();
    tau_on_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau_on = diff_with_prepend_zero(&tau_on_sorted);

    let mut tau_off_sorted = tau_off_orig.clone();
    tau_off_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau_off = diff_with_prepend_zero(&tau_off_sorted);

    let mut combined: Vec<f64> = Vec::with_capacity(dtau_on.len() + dtau_off.len());
    combined.extend_from_slice(&dtau_on);
    combined.extend_from_slice(&dtau_off);
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ub = 3.0 * percentile_sorted(&combined, 99.5);

    apply_increments_subtraction(
        &tau_off_sorted,
        &dtau_off,
        ub,
        &tau_off_orig,
        &mut tau_off_new,
    );
    apply_increments_subtraction(&tau_on_sorted, &dtau_on, ub, &tau_on_orig, &mut tau_on_new);

    for (k, &i) in idx_on.iter().enumerate() {
        tau[i] = tau_on_new[k];
    }
    for (k, &i) in idx_off.iter().enumerate() {
        tau_[i] = tau_off_new[k];
    }
}

pub(super) fn adjust_increments_single(arr: &mut [f64], o: &[u8], pick_on: bool) {
    let target = if pick_on { 1u8 } else { 0u8 };
    let mut idx = Vec::with_capacity(o.len());
    for i in 0..o.len() {
        if o[i] == target {
            idx.push(i);
        }
    }
    let sub_orig: Vec<f64> = idx.iter().map(|&i| arr[i]).collect();
    let mut sub_new = sub_orig.clone();
    let mut sub_sorted = sub_orig.clone();
    sub_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtau = diff_with_prepend_zero(&sub_sorted);
    let mut dtau_sorted = dtau.clone();
    dtau_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ub = 3.0 * percentile_sorted(&dtau_sorted, 99.5);

    apply_increments_subtraction(&sub_sorted, &dtau, ub, &sub_orig, &mut sub_new);
    for (k, &i) in idx.iter().enumerate() {
        arr[i] = sub_new[k];
    }
}

fn diff_with_prepend_zero(sorted: &[f64]) -> Vec<f64> {
    if sorted.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(sorted.len());
    out.push(sorted[0]);
    for i in 1..sorted.len() {
        out.push(sorted[i] - sorted[i - 1]);
    }
    out
}

fn apply_increments_subtraction(
    arr_sorted: &[f64],
    dtau: &[f64],
    ub: f64,
    orig: &[f64],
    new_arr: &mut [f64],
) {
    debug_assert_eq!(arr_sorted.len(), dtau.len());
    debug_assert_eq!(orig.len(), new_arr.len());
    for i in 0..dtau.len() {
        if dtau[i] > ub {
            let ti = arr_sorted[i];
            let dti = dtau[i];
            for k in 0..new_arr.len() {
                if orig[k] >= ti {
                    new_arr[k] -= dti;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_with_prepend_zero_matches_numpy() {
        let arr = [1.0, 4.0, 9.0, 16.0];
        let d = diff_with_prepend_zero(&arr);
        assert_eq!(d, vec![1.0, 3.0, 5.0, 7.0]);
    }
}
