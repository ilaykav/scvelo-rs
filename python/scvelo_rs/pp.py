"""`scvelo_rs.pp` — drop-in for `scvelo.pp`.

Routes through `_pp.py` for primitives that may have a Rust path
(`pca`, `neighbors`, `moments`). Other functions pass through to scvelo
unchanged.
"""

from ._pp import (
    filter_and_normalize,
    filter_genes,
    filter_genes_dispersion,
    log1p,
    moments,
    neighbors,
    normalize_per_cell,
    pca,
    remove_duplicate_cells,
    show_proportions,
)

__all__ = [
    "pca",
    "neighbors",
    "moments",
    "filter_and_normalize",
    "log1p",
    "filter_genes",
    "filter_genes_dispersion",
    "normalize_per_cell",
    "remove_duplicate_cells",
    "show_proportions",
]
