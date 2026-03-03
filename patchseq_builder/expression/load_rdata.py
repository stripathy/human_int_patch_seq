"""
load_rdata.py -- Load expression matrices from RData files and merge into AnnData.

Handles the two patch-seq datasets:
  - LeeDalley (778 cells, log2(FPKM+1), MTG multi-layer)
  - L1 (404 cells, raw FPKM, human L1 interneurons)

The merge resolves 43 overlap cells by keeping LeeDalley expression,
converts L1 to log2(FPKM+1), and produces a unified AnnData.
"""

import numpy as np
import pandas as pd
import pyreadr
import scanpy as sc

from patchseq_builder.config import LD_RDATA, L1_RDATA


def load_leedalley_expression() -> tuple[np.ndarray, list, list]:
    """Load LeeDalley expression matrix from RData.

    Reads the 'datPatch' object from complete_patchseq_data_sets.RData.
    The data is stored as genes x cells in log2(FPKM+1) format.

    Returns
    -------
    expression_matrix : np.ndarray
        Shape (n_cells, n_genes) = (778, 50281), dtype float64.
        Values are in log2(FPKM+1) space.
    gene_names : list[str]
        Gene identifiers (row index of the original DataFrame).
    cell_names : list[str]
        Cell/specimen identifiers (column names of the original DataFrame).
    """
    print("Loading LeeDalley expression from RData...")
    rdata = pyreadr.read_r(str(LD_RDATA))

    dat = rdata["datPatch"]  # 50281 genes x 778 cells (genes as rows)
    gene_names = dat.index.tolist()
    cell_names = dat.columns.tolist()

    # Transpose to cells x genes
    expression_matrix = dat.values.T.astype(np.float64)

    print(f"  LeeDalley: {len(cell_names)} cells x {len(gene_names)} genes")
    print(f"  Format: log2(FPKM+1), range [{expression_matrix.min():.2f}, "
          f"{expression_matrix.max():.2f}]")

    return expression_matrix, gene_names, cell_names


def load_l1_expression() -> tuple[np.ndarray, list, list]:
    """Load L1 expression matrix from RData.

    Reads the 'datPS' object from ps_human.RData.
    The data is stored as genes x cells in raw FPKM (NOT log-transformed).

    Returns
    -------
    expression_matrix : np.ndarray
        Shape (n_cells, n_genes) = (404, 50281), dtype float64.
        Values are in raw FPKM space (not log-transformed).
    gene_names : list[str]
        Gene identifiers (row index of the original DataFrame).
    cell_names : list[str]
        Cell/specimen identifiers (column names of the original DataFrame).
    """
    print("Loading L1 expression from RData...")
    rdata = pyreadr.read_r(str(L1_RDATA))

    dat = rdata["datPS"]  # 50281 genes x 404 cells (genes as rows)
    gene_names = dat.index.tolist()
    cell_names = dat.columns.tolist()

    # Transpose to cells x genes
    expression_matrix = dat.values.T.astype(np.float64)

    print(f"  L1: {len(cell_names)} cells x {len(gene_names)} genes")
    print(f"  Format: raw FPKM, range [{expression_matrix.min():.2f}, "
          f"{expression_matrix.max():.2f}]")

    return expression_matrix, gene_names, cell_names


def load_leedalley_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LeeDalley annotation and metadata tables from RData.

    Returns
    -------
    anno : pd.DataFrame
        The 'annoPatch' table (per-cell annotations: sample_id, spec_id_label, etc.).
    meta : pd.DataFrame
        The 'metaPatch' table (per-cell metadata: Donor, structure, etc.).
    """
    rdata = pyreadr.read_r(str(LD_RDATA))
    return rdata["annoPatch"], rdata["metaPatch"]


def load_l1_annotations() -> pd.DataFrame:
    """Load L1 annotation table from RData.

    Returns
    -------
    anno : pd.DataFrame
        The 'annoPS' table (per-cell annotations: sample_id, spec_id_label, etc.).
    """
    rdata = pyreadr.read_r(str(L1_RDATA))
    return rdata["annoPS"]


def merge_expression(
    ld_expr: np.ndarray,
    ld_genes: list,
    ld_cells: list,
    l1_expr: np.ndarray,
    l1_genes: list,
    l1_cells: list,
    metadata: pd.DataFrame,
) -> sc.AnnData:
    """Merge LeeDalley + L1 expression matrices into a single AnnData.

    The two datasets share 43 overlap cells. For these, LeeDalley expression
    is kept (they agree to high precision after conversion). L1 raw FPKM is
    converted to log2(FPKM+1) before merging.

    Parameters
    ----------
    ld_expr : np.ndarray
        LeeDalley expression, shape (778, n_genes), in log2(FPKM+1).
    ld_genes : list[str]
        Gene names for LeeDalley (should match l1_genes).
    ld_cells : list[str]
        Cell identifiers for LeeDalley.
    l1_expr : np.ndarray
        L1 expression, shape (404, n_genes), in raw FPKM.
    l1_genes : list[str]
        Gene names for L1.
    l1_cells : list[str]
        Cell identifiers for L1.
    metadata : pd.DataFrame
        Pre-built per-cell metadata with index matching cell identifiers.
        Must cover all cells in the union of ld_cells and l1_cells.

    Returns
    -------
    adata : sc.AnnData
        Combined AnnData with 1,139 cells x 50,281 genes in log2(FPKM+1).
        .obs contains the provided metadata columns.
    """
    # Verify gene lists match
    if ld_genes != l1_genes:
        raise ValueError(
            f"Gene lists do not match: LeeDalley has {len(ld_genes)} genes, "
            f"L1 has {len(l1_genes)} genes. "
            f"First mismatch at index {next(i for i, (a, b) in enumerate(zip(ld_genes, l1_genes)) if a != b)}."
        )
    gene_names = ld_genes

    # Convert L1 from FPKM to log2(FPKM+1)
    print("  Converting L1 FPKM -> log2(FPKM+1) to match LeeDalley...")
    l1_expr_log2 = np.log2(l1_expr + 1)

    # Identify overlap cells
    ld_set = set(ld_cells)
    l1_set = set(l1_cells)
    overlap_ids = ld_set & l1_set
    print(f"  Expression overlap: {len(overlap_ids)} cells")

    # Verify overlap agreement after conversion
    if overlap_ids:
        ld_cell_to_idx = {c: i for i, c in enumerate(ld_cells)}
        l1_cell_to_idx = {c: i for i, c in enumerate(l1_cells)}
        ov_sorted = sorted(overlap_ids)
        ld_ov_idx = [ld_cell_to_idx[c] for c in ov_sorted]
        l1_ov_idx = [l1_cell_to_idx[c] for c in ov_sorted]
        max_diff = np.abs(ld_expr[ld_ov_idx] - l1_expr_log2[l1_ov_idx]).max()
        print(f"  Max expression difference in overlap after conversion: {max_diff:.6f}")

    # Build combined cell list: all LD cells + L1-unique cells
    l1_unique_cells = [c for c in l1_cells if c not in overlap_ids]
    l1_unique_idx = [i for i, c in enumerate(l1_cells) if c not in overlap_ids]

    combined_cells = list(ld_cells) + l1_unique_cells
    combined_expr = np.vstack([
        ld_expr,
        l1_expr_log2[l1_unique_idx],
    ]).astype(np.float32)

    print(f"  Combined: {len(combined_cells)} cells x {len(gene_names)} genes")

    # Align metadata to combined cell order
    meta_aligned = metadata.loc[combined_cells].copy()

    # Mark overlap cells
    if "dataset" in meta_aligned.columns:
        for cid in overlap_ids:
            if cid in meta_aligned.index:
                meta_aligned.at[cid, "dataset"] = "both"

    # Build AnnData
    adata = sc.AnnData(
        X=combined_expr,
        obs=meta_aligned.reset_index(drop=True),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = pd.Index(combined_cells)

    print(f"  AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  X dtype: {adata.X.dtype}, range: [{adata.X.min():.1f}, {adata.X.max():.1f}]")

    return adata
