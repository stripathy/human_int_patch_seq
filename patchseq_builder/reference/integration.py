"""
integration.py -- Harmony integration of patch-seq with SEA-AD reference.

Loads the GABAergic interneuron subset of the SEA-AD snRNA-seq reference,
concatenates with patch-seq data (converting expression to a shared space),
then runs PCA + Harmony batch correction + UMAP.

Usage:
    from patchseq_builder.reference.integration import integrate_patchseq_with_reference
    combined = integrate_patchseq_with_reference("data/patchseq/patchseq_combined.h5ad")
"""

import gc
import time

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
from scipy.sparse import issparse, csr_matrix

from patchseq_builder.config import (
    N_HVGS,
    N_PCS,
    N_NEIGHBORS,
    MIN_DIST,
    RANDOM_STATE,
    SEAAD_H5AD,
    PATCHSEQ_H5AD,
)
from patchseq_builder.naming import normalize_subclass
from patchseq_builder.expression.normalize import log2fpkm_to_scanpy


def load_gabaergic_reference() -> sc.AnnData:
    """Load SEA-AD reference, subset to GABAergic neurons.

    Reads the full SEA-AD snRNA-seq h5ad, filters to cells with
    Class == "Neuronal: GABAergic", normalizes (total-count + log1p),
    and annotates with modality and plotting labels.

    Returns
    -------
    sc.AnnData
        GABAergic reference with normalized expression, all genes retained.
        Obs columns include: modality, subclass_for_plot, supertype_for_plot,
        Subclass, Supertype.
    """
    t0 = time.time()
    print("Loading SEA-AD reference...")
    ref = sc.read_h5ad(str(SEAAD_H5AD))
    print(f"  {ref.shape[0]:,} cells x {ref.shape[1]:,} genes ({time.time()-t0:.0f}s)")

    # Subset to GABAergic interneurons
    print("Subsetting to GABAergic interneurons...")
    gaba_mask = ref.obs["Class"] == "Neuronal: GABAergic"
    ref = ref[gaba_mask].copy()
    print(f"  {ref.shape[0]:,} GABAergic cells")

    # Normalize
    print("Normalizing reference...")
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    # Add metadata for downstream concatenation and plotting
    ref.obs["modality"] = "snRNA-seq"
    ref.obs["subclass_for_plot"] = ref.obs["Subclass"].values
    ref.obs["supertype_for_plot"] = ref.obs["Supertype"].values

    return ref


def _load_patchseq(patchseq_h5ad_path, shared_genes):
    """Load patch-seq h5ad, convert expression, subset to shared genes.

    Parameters
    ----------
    patchseq_h5ad_path : str or Path
        Path to the combined patch-seq h5ad file.
    shared_genes : list of str
        Genes present in both reference and patch-seq.

    Returns
    -------
    sc.AnnData
        Patch-seq data with expression in scanpy log1p(CPM/10K) space,
        subset to shared_genes.
    """
    print("Loading patch-seq data...")
    ps = sc.read_h5ad(str(patchseq_h5ad_path))
    print(f"  {ps.shape[0]:,} cells x {ps.shape[1]:,} genes")

    # Subset to shared genes
    ps = ps[:, shared_genes].copy()

    # Convert log2(FPKM+1) -> scanpy log1p(CPM/10K) using centralized function
    print("Converting patch-seq to reference expression space...")
    ps.X = log2fpkm_to_scanpy(ps.X)

    # Add metadata
    ps.obs["modality"] = "patch-seq"

    # Build subclass label for plotting using centralized name normalization
    if "subclass_label" in ps.obs.columns:
        ps_subclass = ps.obs["subclass_label"].astype(str)
    else:
        ps_subclass = pd.Series(["Unknown"] * len(ps), index=ps.obs_names)

    ps.obs["subclass_for_plot"] = ps_subclass.map(
        lambda x: normalize_subclass(x) or x
    ).values

    # Supertype label (scANVI -- only available for a subset of cells)
    if "supertype_scANVI" in ps.obs.columns:
        ps.obs["supertype_for_plot"] = ps.obs["supertype_scANVI"].astype(str).values
    else:
        ps.obs["supertype_for_plot"] = "Unknown"

    return ps


def _concatenate_datasets(ref, ps):
    """Concatenate reference and patch-seq into a single AnnData.

    Aligns obs columns, carries over reference labels for kNN transfer
    and scANVI labels for comparison.

    Parameters
    ----------
    ref : sc.AnnData
        GABAergic reference (normalized, subset to shared genes).
    ps : sc.AnnData
        Patch-seq data (normalized, subset to shared genes).

    Returns
    -------
    sc.AnnData
        Combined AnnData with ``batch`` column for Harmony.
    """
    print("Concatenating datasets...")

    # Densify patch-seq if sparse
    if issparse(ps.X):
        ps.X = ps.X.toarray()

    # Keep only shared obs columns
    shared_obs = ["modality", "subclass_for_plot", "supertype_for_plot"]
    ref_obs = ref.obs[shared_obs].copy()
    ps_obs = ps.obs[shared_obs].copy()

    # Add batch key for Harmony
    ref_obs["batch"] = "snRNA-seq"
    ps_obs["batch"] = "patch-seq"

    # Carry over reference labels for kNN transfer
    ref_obs["ref_subclass"] = ref.obs["Subclass"].values
    ref_obs["ref_supertype"] = ref.obs["Supertype"].values
    ps_obs["ref_subclass"] = np.nan
    ps_obs["ref_supertype"] = np.nan

    # Carry over scANVI labels and specimen_id for patch-seq cells
    for col in [
        "subclass_scANVI", "supertype_scANVI", "specimen_id",
        "subclass_label", "dataset",
        "transcriptomic_type_original", "l1_ttype",
        "cortical_layer",
    ]:
        if col in ps.obs.columns:
            ps_obs[col] = ps.obs[col].values
            ref_obs[col] = np.nan
        else:
            ps_obs[col] = np.nan
            ref_obs[col] = np.nan

    ref_slim = sc.AnnData(
        X=ref.X,
        obs=ref_obs,
        var=ref.var[[]].copy(),
    )
    ref_slim.obs_names = ref.obs_names

    ps_slim = sc.AnnData(
        X=csr_matrix(ps.X) if not issparse(ps.X) else ps.X,
        obs=ps_obs,
        var=ps.var[[]].copy(),
    )
    ps_slim.obs_names = [f"ps_{n}" for n in ps.obs_names]

    combined = sc.concat([ref_slim, ps_slim], join="inner")
    print(f"  Combined: {combined.shape[0]:,} cells x {combined.shape[1]:,} genes")

    # Free memory
    del ref_slim, ps_slim
    gc.collect()

    return combined


def _run_harmony_integration(combined, exclude_genes=None):
    """Run HVG selection, PCA, Harmony, and UMAP on the combined dataset.

    HVGs are computed on the reference (snRNA-seq) subset only, then PCA
    is computed on the full dataset restricted to those HVGs. Harmony
    corrects batch effects between snRNA-seq and patch-seq. Finally,
    neighbors and UMAP are computed on the Harmony-corrected PCs.

    Parameters
    ----------
    combined : sc.AnnData
        Combined AnnData from ``_concatenate_datasets``.
    exclude_genes : set, optional
        Gene names to exclude from HVG selection (e.g. off-target
        contamination genes).

    Returns
    -------
    sc.AnnData
        The input AnnData, modified in-place with:
        - ``.obsm["X_pca"]``
        - ``.obsm["X_pca_harmony"]``
        - ``.obsm["X_umap"]``
    """
    # -- HVGs on reference cells only -----------------------------------------
    print("Computing HVGs on GABAergic reference subset...")
    ref_mask = combined.obs["batch"] == "snRNA-seq"
    ref_subset = combined[ref_mask].copy()

    # Exclude contamination genes BEFORE HVG selection so the algorithm
    # uses its full budget on genuinely informative interneuron genes.
    if exclude_genes:
        clean_mask = ~ref_subset.var_names.isin(exclude_genes)
        n_excluded = int((~clean_mask).sum())
        ref_clean = ref_subset[:, clean_mask].copy()
        print(f"  Removed {n_excluded} contamination genes before HVG selection "
              f"({ref_subset.shape[1]} -> {ref_clean.shape[1]} genes)")
    else:
        ref_clean = ref_subset

    sc.pp.highly_variable_genes(ref_clean, n_top_genes=N_HVGS, flavor="seurat")
    hvg_genes = ref_clean.var_names[ref_clean.var["highly_variable"]].tolist()
    combined.var["highly_variable"] = combined.var_names.isin(hvg_genes)
    print(f"  {len(hvg_genes)} HVGs for integration")
    del ref_subset, ref_clean
    gc.collect()

    # -- Scale + PCA ----------------------------------------------------------
    print("Scaling and PCA...")
    combined_hvg = combined[:, hvg_genes].copy()

    # Densify for scaling (N_HVGS genes is manageable)
    if issparse(combined_hvg.X):
        combined_hvg.X = combined_hvg.X.toarray()

    sc.pp.scale(combined_hvg, max_value=10)
    sc.tl.pca(combined_hvg, n_comps=N_PCS)

    # Copy PCA back to full object
    combined.obsm["X_pca"] = combined_hvg.obsm["X_pca"].copy()
    print(f"  PCA: {combined.obsm['X_pca'].shape}")

    del combined_hvg
    gc.collect()

    # -- Harmony batch correction ---------------------------------------------
    print("Running Harmony integration...")
    ho = hm.run_harmony(
        combined.obsm["X_pca"],
        combined.obs,
        "batch",
        max_iter_harmony=20,
    )
    Z = ho.Z_corr

    # Handle both numpy and torch outputs; ensure shape is (n_cells, n_pcs)
    if hasattr(Z, "numpy"):
        Z = Z.numpy()
    Z = np.asarray(Z, dtype=np.float32)
    if Z.shape[0] == N_PCS and Z.shape[1] != N_PCS:
        Z = Z.T
    if Z.ndim == 1:
        Z = Z.reshape(-1, N_PCS)

    combined.obsm["X_pca_harmony"] = Z
    print(f"  Harmony corrected PCA: {combined.obsm['X_pca_harmony'].shape}")

    # -- Neighbors + UMAP on Harmony-corrected space --------------------------
    print("Computing UMAP on Harmony space...")
    sc.pp.neighbors(
        combined,
        use_rep="X_pca_harmony",
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(combined, random_state=RANDOM_STATE, min_dist=MIN_DIST)
    print(f"  UMAP: {combined.obsm['X_umap'].shape}")

    return combined


def integrate_patchseq_with_reference(
    patchseq_h5ad_path=None,
    reference=None,
    exclude_genes=None,
) -> sc.AnnData:
    """Concatenate patch-seq cells with reference, run Harmony integration.

    Full pipeline:
      1. Load reference (GABAergic subset of SEA-AD)
      2. Load patch-seq, convert expression to scanpy format
      3. Find shared genes, subset both datasets
      4. Concatenate into one AnnData
      5. Compute HVGs on reference only, PCA on HVGs
      6. Harmony integration on ``batch`` key (snRNA-seq vs patch-seq)
      7. Compute UMAP on corrected PCs

    Parameters
    ----------
    patchseq_h5ad_path : str or Path, optional
        Path to the combined patch-seq h5ad. Defaults to ``config.PATCHSEQ_H5AD``.
    reference : sc.AnnData, optional
        Pre-loaded GABAergic reference. If None, loads from ``config.SEAAD_H5AD``
        using ``load_gabaergic_reference()``.
    exclude_genes : set, optional
        Gene names to exclude from HVG selection. If None, loads the cached
        contamination blacklist automatically.

    Returns
    -------
    sc.AnnData
        Combined AnnData with ``.obsm["X_umap"]``, ``.obsm["X_pca_harmony"]``,
        and obs columns including ``batch``, ``ref_subclass``, ``ref_supertype``,
        ``subclass_for_plot``, ``supertype_for_plot``.
    """
    if patchseq_h5ad_path is None:
        patchseq_h5ad_path = PATCHSEQ_H5AD

    t0 = time.time()

    # Step 1: Load reference
    if reference is None:
        reference = load_gabaergic_reference()

    # Step 2: Find shared genes
    print("Finding shared genes...")
    ps_tmp = sc.read_h5ad(str(patchseq_h5ad_path), backed="r")
    shared_genes = sorted(set(reference.var_names) & set(ps_tmp.var_names))
    print(f"  {len(shared_genes)} shared genes")
    del ps_tmp

    # Step 3: Subset reference to shared genes
    reference = reference[:, shared_genes].copy()

    # Step 4: Load and convert patch-seq
    ps = _load_patchseq(patchseq_h5ad_path, shared_genes)

    # Step 5: Concatenate
    combined = _concatenate_datasets(reference, ps)

    # Free memory
    del reference, ps
    gc.collect()

    # Step 6-7: HVG + PCA + Harmony + UMAP
    combined = _run_harmony_integration(combined, exclude_genes=exclude_genes)

    print(f"Integration complete ({time.time()-t0:.0f}s)")
    return combined
