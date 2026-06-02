pub mod pca {
    //! PCA via `nalgebra::SVD`.
    //!
    //! Mirrors `scanpy.pp.pca(adata, n_comps, zero_center=True, svd_solver='arpack')`
    //! for dense input: subtract column means, compute thin SVD, return
    //! the top-`n_comps` left singular vectors weighted by singular values.
    //!
    //! Inputs are f64; we keep all internal math in f64. nalgebra's SVD is
    //! pure Rust (no LAPACK), portable to Windows/macOS/Linux without
    //! native deps.

    use nalgebra::DMatrix;

    /// Compute PCA on `(n_cells, n_genes)` f64 row-major.
    pub fn fit(
        x: &[f64],
        n_cells: usize,
        n_genes: usize,
        n_comps: usize,
        zero_center: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        debug_assert_eq!(x.len(), n_cells * n_genes);
        let n_comps = n_comps.min(n_cells.min(n_genes));

        // Build a column-major nalgebra matrix; the input is row-major f64.
        let mut m = DMatrix::<f64>::from_row_slice(n_cells, n_genes, x);

        if zero_center {
            // Subtract per-column mean - matches scanpy's `zero_center=True` default.
            let col_means: Vec<f64> = (0..n_genes)
                .map(|j| m.column(j).iter().sum::<f64>() / n_cells as f64)
                .collect();
            for i in 0..n_cells {
                for j in 0..n_genes {
                    m[(i, j)] -= col_means[j];
                }
            }
        }

        // Thin SVD: U, S, V^T.
        let svd = m.svd(true, true);
        let u = svd.u.expect("SVD U missing");
        let s = svd.singular_values;
        let vt = svd.v_t.expect("SVD V^T missing");

        // X_pca[:, k] = U[:, k] * S[k]   (n_cells, n_comps)
        let mut x_pca = vec![0.0f64; n_cells * n_comps];
        for i in 0..n_cells {
            for k in 0..n_comps {
                x_pca[i * n_comps + k] = u[(i, k)] * s[k];
            }
        }
        // PCs[k, :] = V^T[k, :]   (n_comps, n_genes)
        let mut pcs = vec![0.0f64; n_comps * n_genes];
        for k in 0..n_comps {
            for j in 0..n_genes {
                pcs[k * n_genes + j] = vt[(k, j)];
            }
        }
        // Variance per component: s[k]^2 / (n_cells - 1).
        // variance_ratio normalizes against the FULL trace (sum over all
        // singular values, not just top n_comps) - matches scanpy/sklearn.
        let denom = (n_cells.max(2) - 1) as f64;
        let total_var: f64 = s.iter().map(|&v| v * v).sum::<f64>() / denom;
        let var: Vec<f64> = (0..n_comps).map(|k| s[k] * s[k] / denom).collect();
        let var_ratio: Vec<f64> = if total_var > 0.0 {
            var.iter().map(|&v| v / total_var).collect()
        } else {
            vec![0.0; n_comps]
        };

        (x_pca, pcs, var, var_ratio)
    }
}

pub mod knn {
    //! KNN graph via `hnsw_rs::Hnsw`.
    //!
    //! Builds an HNSW index over rows of a (n_cells, n_genes) matrix and
    //! returns the per-cell top-k nearest neighbors (excluding self).
    //! Returns flat (n_cells * k) arrays for indices and distances; caller
    //! reshapes / builds CSR.

    use hnsw_rs::prelude::*;
    use rayon::prelude::*;

    /// Compute KNN graph using Euclidean distance.
    pub fn fit_knn_euclidean(
        x: &[f32],
        n_cells: usize,
        n_genes: usize,
        k: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        debug_assert_eq!(x.len(), n_cells * n_genes);

        // hnsw_rs `parallel_insert` wants `&[(&Vec<T>, usize)]`, not `&[(&[T], usize)]`,
        // so own the per-row Vecs in `rows_owned` first.
        let rows_owned: Vec<Vec<f32>> = (0..n_cells)
            .map(|i| x[i * n_genes..(i + 1) * n_genes].to_vec())
            .collect();

        // HNSW parameters: max_nb_connection=24, ef_construction=200 - typical
        // for single-cell-scale KNN; ef_search=max(2k, 50) keeps recall high.
        let max_nb_connection = 24;
        let ef_construction = 200;
        let nb_layer = 16.min((n_cells as f32).log2() as usize + 1);
        let hnsw = Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            n_cells,
            nb_layer,
            ef_construction,
            DistL2 {},
        );

        let inserts: Vec<(&Vec<f32>, usize)> =
            rows_owned.iter().enumerate().map(|(i, r)| (r, i)).collect();
        hnsw.parallel_insert(&inserts);

        let ef_search = (2 * k).max(50);

        // Search top-(k+1) per cell (skip self).
        let raw: Vec<Vec<Neighbour>> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let row = &x[i * n_genes..(i + 1) * n_genes];
                hnsw.search(row, k + 1, ef_search)
            })
            .collect();

        let mut idx = vec![0u32; n_cells * k];
        let mut dist = vec![0.0f32; n_cells * k];
        for (i, neigh) in raw.into_iter().enumerate() {
            let mut written = 0usize;
            for n in neigh {
                if n.d_id == i {
                    continue;
                } // skip self
                if written >= k {
                    break;
                }
                idx[i * k + written] = n.d_id as u32;
                dist[i * k + written] = n.distance;
                written += 1;
            }
            // pad with last value if HNSW didn't return enough neighbors.
            while written < k {
                idx[i * k + written] = idx[i * k + written.saturating_sub(1)];
                dist[i * k + written] = f32::INFINITY;
                written += 1;
            }
        }
        (idx, dist)
    }
}
