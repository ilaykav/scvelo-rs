use rayon::prelude::*;
use std::collections::BTreeSet;

/// Output triplets for one source cell.
pub struct CellTriplets {
    pub source: i32,
    pub neighs: Vec<i32>,
    pub vals: Vec<f32>,
}

/// Compute cosine similarities for all source cells, in parallel.
pub fn compute_cosines_all(
    x: &[f32],
    v: &[f32],
    indices: &[i32],
    n_cells: usize,
    n_genes: usize,
    n_knn: usize,
    n_recurse: usize,
) -> Vec<CellTriplets> {
    (0..n_cells)
        .into_par_iter()
        .map(|i| compute_cosines_one(x, v, indices, n_cells, n_genes, n_knn, n_recurse, i))
        .collect()
}

fn compute_cosines_one(
    x: &[f32],
    v: &[f32],
    indices: &[i32],
    n_cells: usize,
    n_genes: usize,
    n_knn: usize,
    n_recurse: usize,
    i: usize,
) -> CellTriplets {
    let v_i = &v[i * n_genes..(i + 1) * n_genes];

    // skip cells with all-zero velocity
    let mut nonzero = false;
    for &x in v_i.iter() {
        if x != 0.0 {
            nonzero = true;
            break;
        }
    }
    if !nonzero {
        return CellTriplets {
            source: i as i32,
            neighs: Vec::new(),
            vals: Vec::new(),
        };
    }

    let neighs = iterative_neighbors(indices, n_cells, n_knn, i, n_recurse);
    if neighs.is_empty() {
        return CellTriplets {
            source: i as i32,
            neighs: Vec::new(),
            vals: Vec::new(),
        };
    }

    // ||V_i|| - l2 norm
    let mut v_norm_sq: f32 = 0.0;
    for &x in v_i.iter() {
        v_norm_sq += x * x;
    }
    let v_norm = v_norm_sq.sqrt();

    let mut vals = Vec::with_capacity(neighs.len());
    if v_norm == 0.0 {
        for _ in &neighs {
            vals.push(0.0);
        }
        return CellTriplets {
            source: i as i32,
            neighs: neighs.iter().map(|&n| n as i32).collect(),
            vals,
        };
    }

    let x_i = &x[i * n_genes..(i + 1) * n_genes];
    let mut dx = vec![0.0f32; n_genes];

    for &k in &neighs {
        let x_k = &x[k * n_genes..(k + 1) * n_genes];
        // dx = X[k] - X[i]
        for j in 0..n_genes {
            dx[j] = x_k[j] - x_i[j];
        }

        // dx -= dx.mean()  - row centering before cosine
        let mut sum: f32 = 0.0;
        for &x in dx.iter() {
            sum += x;
        }
        let mean = sum / (n_genes as f32);
        for j in 0..n_genes {
            dx[j] -= mean;
        }

        // numerator: dot(dx, V_i)
        let mut dot: f32 = 0.0;
        for j in 0..n_genes {
            dot += dx[j] * v_i[j];
        }

        // denominator: ||dx|| * ||V_i||
        let mut dx_norm_sq: f32 = 0.0;
        for &x in dx.iter() {
            dx_norm_sq += x * x;
        }
        let dx_norm = dx_norm_sq.sqrt();

        let val = if dx_norm == 0.0 {
            0.0
        } else {
            dot / (dx_norm * v_norm)
        };
        vals.push(val);
    }

    CellTriplets {
        source: i as i32,
        neighs: neighs.iter().map(|&n| n as i32).collect(),
        vals,
    }
}

/// Recursive neighbor expansion. depth=1 returns the cell's direct KNN;
fn iterative_neighbors(
    indices: &[i32],
    n_cells: usize,
    n_knn: usize,
    i: usize,
    depth: usize,
) -> Vec<usize> {
    let mut current: BTreeSet<usize> = BTreeSet::new();
    current.insert(i);

    for _ in 0..depth {
        let mut expanded = current.clone();
        for &cell in &current {
            let row = &indices[cell * n_knn..(cell + 1) * n_knn];
            for &nb in row {
                if nb >= 0 && (nb as usize) < n_cells {
                    expanded.insert(nb as usize);
                }
            }
        }
        current = expanded;
    }

    current.into_iter().collect()
}
