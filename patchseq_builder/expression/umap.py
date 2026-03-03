"""
umap.py -- UMAP embedding computation for patch-seq expression, ephys, and morphology data.

Three independent UMAP projections:
  - Expression UMAP: HVG selection -> PCA -> neighbors -> UMAP on log-FPKM
  - Ephys UMAP: feature selection -> subclass-median imputation -> PCA -> UMAP
  - Morphology UMAP: merge LeeDalley + L1 features -> standardize -> PCA -> UMAP

The ephys UMAP uses a careful imputation strategy that fills missing values
with the median of the cell's subclass, falling back to the global median
for cells in minor subclasses or without subclass labels.
"""

import numpy as np
import pandas as pd
import scanpy as sc

from patchseq_builder.config import (
    EPHYS_JOINED_CSV,
    EPHYS_MIN_CELLS_PER_SUBCLASS,
    LD_MORPHO_FEATURES_CSV,
    L1_MORPHO_FEATURES_CSV,
    MIN_DIST,
    N_NEIGHBORS,
    N_PCS,
    RANDOM_STATE,
)


def compute_expression_umap(
    adata: sc.AnnData,
    exclude_genes: set | None = None,
) -> np.ndarray:
    """Compute expression UMAP coordinates from log-FPKM data.

    Pipeline:
        1. Store original expression in a temporary layer
        2. Optionally exclude off-target contamination genes
        3. Apply log1p transform (on top of log2(FPKM+1) already stored in X)
        4. Select top 3000 HVGs (Seurat flavor)
        5. Scale with max_value=10
        6. PCA (N_PCS components) on HVGs
        7. Build neighbor graph (N_NEIGHBORS neighbors)
        8. UMAP

    After computation, the original expression matrix is restored in adata.X
    (including all genes, even those excluded during UMAP computation).

    Parameters
    ----------
    adata : sc.AnnData
        AnnData with X in log2(FPKM+1) space. Modified in-place: gains
        .obsm['X_umap'], .obsm['X_pca'], .var['highly_variable'], and
        neighbor graph in .obsp.
    exclude_genes : set or None
        Gene names to exclude before HVG selection and PCA. These are
        typically off-target contamination genes identified from a reference.

    Returns
    -------
    umap_coords : np.ndarray
        Shape (n_cells, 2) array of UMAP coordinates.
    """
    print("Computing expression UMAP...")

    # Preserve original expression and gene set
    adata.layers["fpkm"] = adata.X.copy()

    # Exclude off-target genes if provided
    if exclude_genes:
        gene_mask = np.array([g not in exclude_genes for g in adata.var_names])
        n_excluded = (~gene_mask).sum()
        print(f"  Excluding {n_excluded} off-target contamination genes")
        # Work on a subset for HVG/PCA/UMAP, but keep full adata intact
        adata_clean = adata[:, gene_mask].copy()
        adata_clean.X = np.log1p(adata_clean.X)
        sc.pp.highly_variable_genes(adata_clean, n_top_genes=3000, flavor="seurat")
        n_hvg = adata_clean.var["highly_variable"].sum()
        print(f"  HVGs: {n_hvg} (from {gene_mask.sum()} clean genes)")
        sc.pp.scale(adata_clean, max_value=10)
        sc.tl.pca(adata_clean, n_comps=N_PCS, use_highly_variable=True)
        sc.pp.neighbors(adata_clean, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=RANDOM_STATE)
        sc.tl.umap(adata_clean, random_state=RANDOM_STATE)
        # Copy results back to original adata
        adata.obsm["X_umap"] = adata_clean.obsm["X_umap"]
        adata.obsm["X_pca"] = adata_clean.obsm["X_pca"]
        # Store which genes were used
        adata.uns["umap_excluded_genes"] = sorted(exclude_genes)
        adata.uns["umap_n_excluded"] = n_excluded
    else:
        # Original pipeline: work directly on adata
        adata.X = np.log1p(adata.X)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
        n_hvg = adata.var["highly_variable"].sum()
        print(f"  HVGs: {n_hvg}")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=N_PCS, use_highly_variable=True)
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=RANDOM_STATE)
        sc.tl.umap(adata, random_state=RANDOM_STATE)

    # Restore original expression
    adata.X = adata.layers["fpkm"]
    del adata.layers["fpkm"]

    umap_coords = adata.obsm["X_umap"]
    print(f"  UMAP computed: {umap_coords.shape}")

    return umap_coords


def compute_ephys_umap(
    adata: sc.AnnData,
    ephys: pd.DataFrame | None = None,
    min_cells_per_subclass: int = 10,
) -> np.ndarray:
    """Compute ephys UMAP from electrophysiology features with subclass-median imputation.

    Strategy:
        1. Select all numeric ephys features (excluding QC/fail columns) where
           every major subclass (>= min_cells_per_subclass cells) has at least
           EPHYS_MIN_CELLS_PER_SUBCLASS non-NaN observations.
        2. Remove near-zero-variance features.
        3. Impute missing values with the subclass median for that feature.
           Cells in minor subclasses or without a subclass label get the global median.
        4. Z-score normalize -> PCA -> UMAP.

    Cells with >50% of features imputed are flagged in adata.obs['ephys_high_imputation'].
    Cells without any ephys data get NaN in the UMAP coordinates.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData with .obs containing 'specimen_id' and 'subclass_label'.
        Modified in-place: gains .obsm['X_umap_ephys'], .obsm['X_ephys_pca'],
        .obsm['X_ephys_scaled'], .obs['ephys_frac_imputed'],
        .obs['ephys_high_imputation'], and .uns ephys metadata.
    ephys : pd.DataFrame, optional
        Full ephys feature table indexed by specimen_id. If None, loaded from
        config.EPHYS_JOINED_CSV.
    min_cells_per_subclass : int, optional
        Minimum cells to consider a subclass "major" for feature selection.
        Default is 10.

    Returns
    -------
    umap_coords : np.ndarray
        Shape (n_cells, 2) array of UMAP coordinates. NaN for cells without
        ephys data.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    print("Computing ephys UMAP (subclass-median imputation)...")

    # ── Load ephys data ──────────────────────────────────────────────
    if ephys is None:
        ephys_df = pd.read_csv(str(EPHYS_JOINED_CSV))
        ephys_df = ephys_df.set_index("specimen_id")
    else:
        ephys_df = ephys.copy()
        if "specimen_id" in ephys_df.columns:
            ephys_df = ephys_df.set_index("specimen_id")

    # ── Identify numeric feature columns (exclude metadata/QC) ───────
    exclude_cols = {"dataset"}
    feat_cols = [
        c for c in ephys_df.columns
        if c not in exclude_cols
        and ephys_df[c].dtype in ["float64", "int64", "float32"]
        and "qc" not in c.lower()
        and "fail" not in c.lower()
    ]
    print(f"  Numeric non-QC features: {len(feat_cols)}")

    # ── Map specimen_ids to subclass labels ──────────────────────────
    if "subclass_label" in adata.obs.columns:
        sid_to_subclass = dict(zip(
            adata.obs["specimen_id"].astype(float).astype(int),
            adata.obs["subclass_label"],
        ))
    else:
        sid_to_subclass = {}

    ephys_df["_subclass"] = ephys_df.index.map(
        lambda x: sid_to_subclass.get(int(x), np.nan) if pd.notna(x) else np.nan
    )

    # ── Identify major subclasses ────────────────────────────────────
    sub_counts = ephys_df["_subclass"].value_counts()
    major_subs = sub_counts[sub_counts >= min_cells_per_subclass].index.tolist()
    print(f"  Major subclasses (>= {min_cells_per_subclass} cells): {major_subs}")

    # ── Select features with sufficient coverage in all major subclasses
    ephys_major = ephys_df[ephys_df["_subclass"].isin(major_subs)]
    imputable_feats = []
    for feat in feat_cols:
        sub_coverage = ephys_major.groupby("_subclass")[feat].apply(
            lambda x: x.notna().sum()
        )
        if (sub_coverage >= EPHYS_MIN_CELLS_PER_SUBCLASS).all():
            imputable_feats.append(feat)
    print(f"  Features imputable by subclass (all major subs >= "
          f"{EPHYS_MIN_CELLS_PER_SUBCLASS} obs): {len(imputable_feats)}")

    # ── Remove near-zero-variance features ───────────────────────────
    temp_data = ephys_df[imputable_feats].dropna()
    stds = temp_data.std()
    low_var = stds[stds < 1e-10].index.tolist()
    if low_var:
        print(f"  Removing {len(low_var)} zero-variance features: {low_var}")
        imputable_feats = [f for f in imputable_feats if f not in low_var]
    print(f"  Final feature count: {len(imputable_feats)}")

    # ── Map ephys specimen_ids to adata indices ──────────────────────
    if "specimen_id" in adata.obs.columns:
        adata_specimen_ids = adata.obs["specimen_id"].values
    else:
        adata_specimen_ids = adata.obs_names.values

    # ── Build raw ephys matrix aligned to adata (before imputation) ──
    n_feats = len(imputable_feats)
    ephys_raw = np.full((len(adata), n_feats), np.nan)
    for i, sid in enumerate(adata_specimen_ids):
        if pd.notna(sid):
            try:
                sid_int = int(float(sid))
            except (ValueError, TypeError):
                continue
            if sid_int in ephys_df.index:
                ephys_raw[i] = ephys_df.loc[sid_int, imputable_feats].values

    # ── Identify cells with any ephys data ───────────────────────────
    has_any_ephys = ~np.isnan(ephys_raw).all(axis=1)
    n_with_ephys = has_any_ephys.sum()
    print(f"  Cells with any ephys data: {n_with_ephys}/{len(adata)}")

    # ── Missing-value statistics before imputation ───────────────────
    n_missing_per_cell = np.isnan(ephys_raw[has_any_ephys]).sum(axis=1)
    frac_missing = n_missing_per_cell / n_feats
    print(f"  Missing fraction per cell: "
          f"median={np.median(frac_missing):.1%}, "
          f"mean={np.mean(frac_missing):.1%}, "
          f"max={np.max(frac_missing):.1%}")

    # ── Compute subclass medians for imputation ──────────────────────
    subclass_medians = {}
    for sub in major_subs:
        mask = ephys_df["_subclass"] == sub
        subclass_medians[sub] = ephys_df.loc[mask, imputable_feats].median().values
    global_median = ephys_df[imputable_feats].median().values

    # ── Impute missing values ────────────────────────────────────────
    ephys_imputed = ephys_raw.copy()
    n_imputed_total = 0
    for i in range(len(adata)):
        if not has_any_ephys[i]:
            continue
        nan_mask = np.isnan(ephys_imputed[i])
        if not nan_mask.any():
            continue

        sid = adata_specimen_ids[i]
        try:
            sid_int = int(float(sid))
        except (ValueError, TypeError):
            sid_int = None

        subclass = sid_to_subclass.get(sid_int, None) if sid_int else None

        if subclass in subclass_medians:
            ephys_imputed[i, nan_mask] = subclass_medians[subclass][nan_mask]
        else:
            ephys_imputed[i, nan_mask] = global_median[nan_mask]
        n_imputed_total += nan_mask.sum()

    total_vals = n_with_ephys * n_feats
    print(f"  Imputed {n_imputed_total}/{total_vals} values "
          f"({n_imputed_total / total_vals:.1%})")

    # ── Handle any remaining NaN (safety fallback) ───────────────────
    remaining_nan = np.isnan(ephys_imputed[has_any_ephys]).sum()
    if remaining_nan > 0:
        print(f"  WARNING: {remaining_nan} NaN values remain after imputation")
        for j in range(n_feats):
            col = ephys_imputed[has_any_ephys, j]
            nan_idx = np.isnan(col)
            if nan_idx.any():
                col[nan_idx] = global_median[j]

    # ── Flag heavily-imputed cells ───────────────────────────────────
    frac_imputed_per_cell = np.full(len(adata), np.nan)
    frac_imputed_per_cell[has_any_ephys] = frac_missing
    adata.obs["ephys_frac_imputed"] = frac_imputed_per_cell
    adata.obs["ephys_high_imputation"] = frac_imputed_per_cell > 0.50
    n_high_imp = (frac_imputed_per_cell > 0.50).sum()
    print(f"  Cells with >50% imputed: {n_high_imp}")

    if n_with_ephys < 20:
        print("  Too few cells for ephys UMAP, skipping.")
        full_umap = np.full((len(adata), 2), np.nan)
        adata.obsm["X_umap_ephys"] = full_umap
        return full_umap

    # ── Z-score normalize (using only cells with ephys) ──────────────
    ephys_valid = ephys_imputed[has_any_ephys]
    scaler = StandardScaler()
    ephys_scaled = scaler.fit_transform(ephys_valid)

    # Store scaled features
    ephys_scaled_full = np.full((len(adata), n_feats), np.nan)
    ephys_scaled_full[has_any_ephys] = ephys_scaled
    adata.obsm["X_ephys_scaled"] = ephys_scaled_full

    # ── PCA ───────────────────────────────────────────────────────────
    n_pcs = min(N_PCS, n_feats - 1, n_with_ephys - 1)
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    ephys_pca = pca.fit_transform(ephys_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_pcs} components, {var_explained * 100:.1f}% variance explained")

    # Store PCA
    ephys_pca_full = np.full((len(adata), n_pcs), np.nan)
    ephys_pca_full[has_any_ephys] = ephys_pca
    adata.obsm["X_ephys_pca"] = ephys_pca_full

    # ── UMAP on PCA ──────────────────────────────────────────────────
    reducer = UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=RANDOM_STATE)
    ephys_umap = reducer.fit_transform(ephys_pca)

    full_umap = np.full((len(adata), 2), np.nan)
    full_umap[has_any_ephys] = ephys_umap
    adata.obsm["X_umap_ephys"] = full_umap

    # ── Store metadata in .uns ────────────────────────────────────────
    adata.uns["ephys_features"] = imputable_feats
    adata.uns["ephys_n_pcs"] = n_pcs
    adata.uns["ephys_pca_var_explained"] = float(var_explained)
    adata.uns["ephys_imputation_method"] = "subclass_median"
    adata.uns["ephys_n_imputed_values"] = int(n_imputed_total)

    print(f"  Ephys UMAP computed for {n_with_ephys} cells "
          f"({n_feats} features, subclass-median imputation -> "
          f"{n_pcs} PCs -> 2D UMAP)")

    return full_umap


def compute_morphology_umap(adata: sc.AnnData) -> np.ndarray:
    """Compute morphology UMAP from quantitative morphology features.

    Merges pre-computed morphology feature tables from LeeDalley (140 cells)
    and L1 (86 cells) on their shared columns, then runs standardize -> PCA
    -> UMAP. No imputation is needed (both CSVs have zero NaN).

    For the ~11 overlapping specimen_ids, LeeDalley rows are preferred.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData with .obs containing 'specimen_id'. Modified in-place: gains
        .obsm['X_umap_morph'], .obsm['X_morph_pca'], .obsm['X_morph_scaled'],
        and .uns morphology metadata.

    Returns
    -------
    umap_coords : np.ndarray
        Shape (n_cells, 2). NaN for cells without morphology features.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    print("Computing morphology UMAP...")

    # ── Load both feature tables ─────────────────────────────────────
    ld_morph = pd.read_csv(str(LD_MORPHO_FEATURES_CSV))
    l1_morph = pd.read_csv(str(L1_MORPHO_FEATURES_CSV))

    # Drop unnamed index columns
    ld_morph = ld_morph.loc[:, ~ld_morph.columns.str.startswith("Unnamed")]
    l1_morph = l1_morph.loc[:, ~l1_morph.columns.str.startswith("Unnamed")]

    print(f"  LeeDalley: {len(ld_morph)} cells, {len(ld_morph.columns)} cols")
    print(f"  L1: {len(l1_morph)} cells, {len(l1_morph.columns)} cols")

    # ── Shared feature columns ───────────────────────────────────────
    shared_cols = sorted(set(ld_morph.columns) & set(l1_morph.columns))
    feat_cols = [c for c in shared_cols if c != "specimen_id"]
    print(f"  Shared feature columns: {len(feat_cols)}")

    # ── Merge: LeeDalley first (preferred for overlapping specimens) ──
    ld_sub = ld_morph[["specimen_id"] + feat_cols].copy()
    l1_sub = l1_morph[["specimen_id"] + feat_cols].copy()
    merged = pd.concat([ld_sub, l1_sub], ignore_index=True)
    merged = merged.drop_duplicates(subset="specimen_id", keep="first")
    merged = merged.set_index("specimen_id")
    print(f"  Merged: {len(merged)} unique cells")

    # ── Map to adata via specimen_id ─────────────────────────────────
    if "specimen_id" not in adata.obs.columns:
        print("  ERROR: specimen_id not in adata.obs, cannot compute morphology UMAP")
        full_umap = np.full((len(adata), 2), np.nan)
        adata.obsm["X_umap_morph"] = full_umap
        return full_umap

    n_feats = len(feat_cols)
    morph_raw = np.full((len(adata), n_feats), np.nan)
    for i, sid in enumerate(adata.obs["specimen_id"].values):
        if pd.notna(sid):
            try:
                sid_int = int(float(sid))
            except (ValueError, TypeError):
                continue
            if sid_int in merged.index:
                morph_raw[i] = merged.loc[sid_int, feat_cols].values

    has_morph = ~np.isnan(morph_raw).all(axis=1)
    n_with_morph = has_morph.sum()
    print(f"  Cells with morphology features: {n_with_morph}/{len(adata)}")

    if n_with_morph < 20:
        print("  Too few cells for morphology UMAP, skipping.")
        full_umap = np.full((len(adata), 2), np.nan)
        adata.obsm["X_umap_morph"] = full_umap
        return full_umap

    # ── Standardize ──────────────────────────────────────────────────
    morph_valid = morph_raw[has_morph]
    scaler = StandardScaler()
    morph_scaled = scaler.fit_transform(morph_valid)

    morph_scaled_full = np.full((len(adata), n_feats), np.nan)
    morph_scaled_full[has_morph] = morph_scaled
    adata.obsm["X_morph_scaled"] = morph_scaled_full

    # ── PCA ───────────────────────────────────────────────────────────
    n_pcs = min(N_PCS, n_feats - 1, n_with_morph - 1)
    pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    morph_pca = pca.fit_transform(morph_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_pcs} components, {var_explained * 100:.1f}% variance explained")

    morph_pca_full = np.full((len(adata), n_pcs), np.nan)
    morph_pca_full[has_morph] = morph_pca
    adata.obsm["X_morph_pca"] = morph_pca_full

    # ── UMAP ──────────────────────────────────────────────────────────
    reducer = UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=RANDOM_STATE)
    morph_umap = reducer.fit_transform(morph_pca)

    full_umap = np.full((len(adata), 2), np.nan)
    full_umap[has_morph] = morph_umap
    adata.obsm["X_umap_morph"] = full_umap

    # ── Store metadata ────────────────────────────────────────────────
    adata.uns["morph_features"] = feat_cols
    adata.uns["morph_n_pcs"] = n_pcs
    adata.uns["morph_pca_var_explained"] = float(var_explained)

    print(f"  Morphology UMAP computed for {n_with_morph} cells "
          f"({n_feats} features -> {n_pcs} PCs -> 2D UMAP)")

    return full_umap
