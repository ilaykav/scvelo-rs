#[derive(Clone, Copy)]
pub struct CsrView<'a> {
    pub data: &'a [f64],
    pub indices: &'a [i32],
    pub indptr: &'a [i32],
    pub nrows: usize,
}

impl<'a> CsrView<'a> {
    pub fn new(data: &'a [f64], indices: &'a [i32], indptr: &'a [i32]) -> Self {
        let nrows = indptr.len().saturating_sub(1);
        debug_assert!(!indptr.is_empty(), "indptr must have length >= 1");
        Self {
            data,
            indices,
            indptr,
            nrows,
        }
    }
}

/// `out = M * v`. Same accumulation order as scipy: row-major, summing left-to-right
/// over the CSR row's nz entries.
pub fn matvec(m: CsrView<'_>, v: &[f64], out: &mut [f64]) {
    debug_assert_eq!(out.len(), m.nrows);
    for i in 0..m.nrows {
        let start = m.indptr[i] as usize;
        let end = m.indptr[i + 1] as usize;
        let mut acc = 0.0f64;
        for k in start..end {
            let j = m.indices[k] as usize;
            // SAFETY: within bounds when input CSR is well-formed; debug-checked.
            debug_assert!(j < v.len());
            acc += m.data[k] * v[j];
        }
        out[i] = acc;
    }
}

/// `out[c, i] = (M * V[c])[i]`, where V is laid out as `n_cols` independent column
pub fn matvec_multi(m: CsrView<'_>, vcols: &[&[f64]], out_cols: &mut [&mut [f64]]) {
    let k = vcols.len();
    debug_assert_eq!(out_cols.len(), k);
    for c in 0..k {
        debug_assert_eq!(vcols[c].len(), m.nrows);
        debug_assert_eq!(out_cols[c].len(), m.nrows);
    }
    for i in 0..m.nrows {
        let start = m.indptr[i] as usize;
        let end = m.indptr[i + 1] as usize;
        // Fused-row loop: load each (idx, weight) once, splat across all k columns.
        // For typical k=2..4 and ~30 nnz/row this is the right shape.
        for c in 0..k {
            let mut acc = 0.0f64;
            for kk in start..end {
                let j = m.indices[kk] as usize;
                debug_assert!(j < vcols[c].len());
                acc += m.data[kk] * vcols[c][j];
            }
            out_cols[c][i] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_identity() {
        // 3x3 identity in CSR.
        let data = [1.0, 1.0, 1.0];
        let indices = [0i32, 1, 2];
        let indptr = [0i32, 1, 2, 3];
        let m = CsrView::new(&data, &indices, &indptr);
        let v = [10.0, 20.0, 30.0];
        let mut out = [0.0; 3];
        matvec(m, &v, &mut out);
        assert_eq!(out, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn matvec_dense_3x3() {
        // M = [[1,2,0],[0,3,4],[5,0,6]]
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [0i32, 1, 1, 2, 0, 2];
        let indptr = [0i32, 2, 4, 6];
        let m = CsrView::new(&data, &indices, &indptr);
        let v = [1.0, 1.0, 1.0];
        let mut out = [0.0; 3];
        matvec(m, &v, &mut out);
        assert_eq!(out, [3.0, 7.0, 11.0]);
    }

    #[test]
    fn matvec_multi_matches_per_column() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = [0i32, 1, 1, 2, 0, 2];
        let indptr = [0i32, 2, 4, 6];
        let m = CsrView::new(&data, &indices, &indptr);

        let v0 = [1.0, 2.0, 3.0];
        let v1 = [10.0, 20.0, 30.0];
        let mut o0 = [0.0; 3];
        let mut o1 = [0.0; 3];
        let cols: [&[f64]; 2] = [&v0, &v1];
        {
            let (lo, hi) = (&mut o0, &mut o1);
            let mut outs: [&mut [f64]; 2] = [lo, hi];
            matvec_multi(m, &cols, &mut outs);
        }

        let mut ref0 = [0.0; 3];
        let mut ref1 = [0.0; 3];
        matvec(m, &v0, &mut ref0);
        matvec(m, &v1, &mut ref1);
        assert_eq!(o0, ref0);
        assert_eq!(o1, ref1);
    }
}
