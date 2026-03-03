"""
normalize.py -- Expression normalization utilities for patch-seq data.

Provides conversions between expression formats:
  - log2(FPKM+1): stored format in the combined patch-seq h5ad
  - raw FPKM: format of the L1 dataset as loaded from RData
  - log1p(CPM/10K): scanpy-standard format used by SEA-AD reference

These functions replace inline conversions scattered across
build_combined_patchseq_h5ad.py and build_patchseq_in_reference_umap.py.
"""

import numpy as np

from patchseq_builder.config import NORMALIZE_TARGET_SUM


def fpkm_to_log2fpkm(X: np.ndarray) -> np.ndarray:
    """Convert raw FPKM values to log2(FPKM+1).

    Used for L1 data conversion before merging with LeeDalley data.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix in raw FPKM space. Any shape; operates element-wise.

    Returns
    -------
    np.ndarray
        Expression matrix in log2(FPKM+1) space, same shape and dtype as input.
    """
    return np.log2(X + 1)


def log2fpkm_to_fpkm(X: np.ndarray) -> np.ndarray:
    """Convert log2(FPKM+1) back to raw FPKM.

    Inverse of fpkm_to_log2fpkm. Clamps negative values to zero to handle
    floating-point edge cases.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix in log2(FPKM+1) space.

    Returns
    -------
    np.ndarray
        Expression matrix in raw FPKM space (non-negative).
    """
    fpkm = np.power(2, X) - 1
    return np.maximum(fpkm, 0)


def log2fpkm_to_scanpy(X: np.ndarray, target_sum: float = NORMALIZE_TARGET_SUM) -> np.ndarray:
    """Convert log2(FPKM+1) to scanpy-standard log1p(CPM/10K).

    This is the normalization pipeline used to put patch-seq data into the
    same expression space as the SEA-AD snRNA-seq reference for integration.

    Pipeline:
        1. log2(FPKM+1) -> FPKM  (reverse the log2 transform)
        2. FPKM -> normalize_total(target_sum)  (library-size normalization)
        3. normalized -> log1p  (natural log transform)

    Parameters
    ----------
    X : np.ndarray
        Expression matrix in log2(FPKM+1) space, shape (n_cells, n_genes).
    target_sum : float, optional
        Target sum for library-size normalization. Default is 1e4 (from config).

    Returns
    -------
    np.ndarray
        Expression matrix in log1p(CPM/10K) space, shape (n_cells, n_genes),
        dtype float32.
    """
    # Step 1: log2(FPKM+1) -> FPKM
    fpkm = log2fpkm_to_fpkm(X)

    # Step 2: library-size normalization to target_sum
    row_sums = fpkm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero for empty cells
    normalized = fpkm / row_sums * target_sum

    # Step 3: log1p
    return np.log1p(normalized).astype(np.float32)
