#!/usr/bin/env python
"""
build_combined_patchseq_h5ad.py — Build a combined AnnData h5ad from both
patch-seq datasets (LeeDalley + L1), with harmonized metadata, ephys, scANVI
labels, and UMAP embeddings computed from expression and ephys separately.

Output:
  data/patchseq/patchseq_combined.h5ad    — 1,139 cells x 50,281 genes (log2(FPKM+1))
  results/figures/patchseq_umap_expression.png
  results/figures/patchseq_umap_ephys.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pyreadr
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/shreejoy/Github/patch_seq_lee")
OUTPUT_DIR = PROJECT_ROOT / "data" / "patchseq"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_expression_data():
    """Load FPKM expression matrices from both datasets."""
    print("Loading expression data from .RData files...")

    ld = pyreadr.read_r(str(PROJECT_ROOT / "complete_patchseq_data_sets.RData"))
    l1 = pyreadr.read_r(str(PROJECT_ROOT / "patchseq_human_L1-main" / "data" / "ps_human.RData"))

    dat_ld = ld["datPatch"]      # 50281 genes x 778 cells
    anno_ld = ld["annoPatch"]
    meta_ld = ld["metaPatch"]
    dat_l1 = l1["datPS"]         # 50281 genes x 404 cells
    anno_l1 = l1["annoPS"]

    print(f"  LeeDalley: {dat_ld.shape[1]} cells x {dat_ld.shape[0]} genes")
    print(f"  L1:        {dat_l1.shape[1]} cells x {dat_l1.shape[0]} genes")

    return dat_ld, anno_ld, meta_ld, dat_l1, anno_l1


def build_cell_metadata(anno_ld, meta_ld, anno_l1, dat_ld, dat_l1):
    """Build harmonized per-cell metadata for both datasets."""

    # ── LeeDalley cells ──────────────────────────────────────────────
    ld_meta = pd.DataFrame({
        "sample_id": anno_ld["sample_id"],
        "specimen_id": anno_ld["spec_id_label"].astype(str).astype(int),
        "cell_name": anno_ld["cell_name_label"],
        "exp_component_name": anno_ld["exp_component_name_label"],
        "donor": meta_ld["Donor"] if "Donor" in meta_ld.columns else np.nan,
        "brain_region": meta_ld["structure"] if "structure" in meta_ld.columns else np.nan,
        "cortical_layer": anno_ld["layer_label"] if "layer_label" in anno_ld.columns else np.nan,
        "dataset": "LeeDalley",
    })
    ld_meta.index = dat_ld.columns

    # ── L1 cells ─────────────────────────────────────────────────────
    l1_meta = pd.DataFrame({
        "sample_id": anno_l1["sample_id"],
        "specimen_id": anno_l1["spec_id_label"].astype(str).astype(int),
        "cell_name": anno_l1["cell_name_label"],
        "exp_component_name": anno_l1["exp_component_name_label"],
        "donor": anno_l1.get("donor_label", pd.Series([np.nan] * len(anno_l1))),
        "brain_region": anno_l1.get("structure_label", pd.Series([np.nan] * len(anno_l1))),
        "cortical_layer": anno_l1["layer_label"] if "layer_label" in anno_l1.columns else np.nan,
        "dataset": "L1",
    })
    l1_meta.index = dat_l1.columns

    return ld_meta, l1_meta


def merge_expression(dat_ld, dat_l1, ld_meta, l1_meta):
    """Merge expression matrices, handling overlap cells.

    LeeDalley data is log2(FPKM+1), L1 data is raw FPKM.
    We convert L1 to log2(FPKM+1) to match, then store as log2(FPKM+1).
    """

    overlap_ids = set(dat_ld.columns) & set(dat_l1.columns)
    print(f"\n  Expression overlap: {len(overlap_ids)} cells")

    # Convert L1 from FPKM to log2(FPKM+1) to match LeeDalley
    print("  Converting L1 FPKM -> log2(FPKM+1) to match LeeDalley...")
    dat_l1 = np.log2(dat_l1 + 1)

    # Verify overlap agreement after conversion
    if overlap_ids:
        ov_list = sorted(overlap_ids)
        diff = (dat_ld[ov_list] - dat_l1[ov_list]).abs().max().max()
        print(f"  Max expression difference in overlap after conversion: {diff:.6f}")

    # Keep LD version for overlap, remove from L1
    l1_unique = [c for c in dat_l1.columns if c not in overlap_ids]
    dat_combined = pd.concat([dat_ld, dat_l1[l1_unique]], axis=1)

    # Build combined metadata
    l1_meta_unique = l1_meta.loc[l1_unique]
    ld_overlap_meta = ld_meta.loc[ld_meta.index.isin(overlap_ids)].copy()
    ld_overlap_meta["dataset"] = "both"

    # Fill overlap metadata with L1 info where missing
    for sid in overlap_ids:
        if sid in l1_meta.index:
            l1_row = l1_meta.loc[sid]
            ld_row = ld_overlap_meta.loc[sid]
            for col in ld_overlap_meta.columns:
                if pd.isna(ld_row[col]) and not pd.isna(l1_row.get(col, np.nan)):
                    ld_overlap_meta.at[sid, col] = l1_row[col]

    ld_nonov_meta = ld_meta.loc[~ld_meta.index.isin(overlap_ids)]
    meta_combined = pd.concat([ld_nonov_meta, ld_overlap_meta, l1_meta_unique])
    meta_combined = meta_combined.loc[dat_combined.columns]

    print(f"  Combined: {dat_combined.shape[1]} cells x {dat_combined.shape[0]} genes")

    return dat_combined, meta_combined


def add_scanvi_and_ephys(meta, combined_csv_path):
    """Add scANVI labels and key ephys features from the joined CSV."""

    if combined_csv_path.exists():
        joined = pd.read_csv(str(combined_csv_path))

        # Match on specimen_id
        meta_specs = meta["specimen_id"].values
        joined_indexed = joined.set_index("specimen_id")

        scanvi_cols = ["subclass_scANVI", "supertype_scANVI",
                       "subclass_conf_scANVI", "supertype_conf_scANVI"]
        ephys_cols = ["sag", "tau", "input_resistance", "rheobase_i",
                      "fi_fit_slope", "v_baseline"]

        for col in scanvi_cols + ephys_cols:
            if col in joined_indexed.columns:
                vals = []
                for spec in meta_specs:
                    if spec in joined_indexed.index:
                        v = joined_indexed.at[spec, col]
                        # Handle duplicate spec_ids (take first)
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        vals.append(v)
                    else:
                        vals.append(np.nan)
                meta[col] = vals

        n_scanvi = meta["subclass_scANVI"].notna().sum() if "subclass_scANVI" in meta.columns else 0
        n_ephys = meta["sag"].notna().sum() if "sag" in meta.columns else 0
        print(f"\n  scANVI labels (from combined CSV): {n_scanvi}/{len(meta)} cells")
        print(f"  Ephys features: {n_ephys}/{len(meta)} cells")

    # Also add original subclass from the joined CSV
    if combined_csv_path.exists():
        for col in ["subclass_label_original", "transcriptomic_type_original",
                     "l1_ttype", "disease_category"]:
            if col in joined_indexed.columns:
                vals = []
                for spec in meta_specs:
                    if spec in joined_indexed.index:
                        v = joined_indexed.at[spec, col]
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        vals.append(v)
                    else:
                        vals.append(np.nan)
                meta[col] = vals

    # ── Fill in missing scANVI annotations from the raw scANVI results CSV ──
    # The combined CSV only had scANVI for a subset of cells because it was
    # merged on the wrong key. The raw scANVI CSV uses exp_component_name
    # (Smart-seq library ID) as its index, which matches our exp_component_name column.
    scanvi_csv = combined_csv_path.parent / "iterative_scANVI_results_patchseq_only.2022-11-22.csv"
    if scanvi_csv.exists() and "exp_component_name" in meta.columns:
        scanvi_raw = pd.read_csv(str(scanvi_csv), index_col=0)
        scanvi_cols = ["subclass_scANVI", "supertype_scANVI",
                       "subclass_conf_scANVI", "supertype_conf_scANVI"]

        n_before = meta["subclass_scANVI"].notna().sum() if "subclass_scANVI" in meta.columns else 0

        # Build lookup: exp_component_name -> scANVI annotations
        for col in scanvi_cols:
            if col not in scanvi_raw.columns:
                continue
            if col not in meta.columns:
                meta[col] = np.nan

            # Only fill cells that are currently missing this annotation
            missing_mask = meta[col].isna()
            for i in meta.index[missing_mask]:
                ecn = str(meta.at[i, "exp_component_name"])
                if ecn in scanvi_raw.index:
                    meta.at[i, col] = scanvi_raw.at[ecn, col]

        n_after = meta["subclass_scANVI"].notna().sum()
        print(f"  scANVI labels after exp_component_name merge: {n_after}/{len(meta)} cells "
              f"(+{n_after - n_before} from raw scANVI CSV)")
    elif not scanvi_csv.exists():
        print(f"  WARNING: Raw scANVI CSV not found at {scanvi_csv}")

    return meta


def build_anndata(dat_combined, meta_combined):
    """Create AnnData object from expression matrix and metadata."""

    # Transpose: cells x genes
    X = dat_combined.T.values.astype(np.float32)
    gene_names = dat_combined.index.values

    adata = sc.AnnData(
        X=X,
        obs=meta_combined.reset_index(drop=True),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = meta_combined.index.values

    print(f"\n  AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  X dtype: {adata.X.dtype}, range: [{adata.X.min():.1f}, {adata.X.max():.1f}]")

    return adata


def compute_expression_umap(adata):
    """Compute UMAP from gene expression (log-FPKM, HVG, PCA)."""
    print("\nComputing expression UMAP...")

    # Work on a copy for processing
    adata.layers["fpkm"] = adata.X.copy()

    # Log-transform FPKM
    adata.X = np.log1p(adata.X)

    # HVG selection (use 'seurat' flavor which doesn't require skmisc)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
    print(f"  HVGs: {adata.var['highly_variable'].sum()}")

    # PCA on HVGs
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)

    # Neighbors + UMAP (fixed seed for reproducibility)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=42)
    sc.tl.umap(adata, random_state=42)

    # Restore original FPKM
    adata.X = adata.layers["fpkm"]
    del adata.layers["fpkm"]

    print(f"  UMAP computed: {adata.obsm['X_umap'].shape}")
    return adata


def compute_ephys_umap(adata):
    """Compute UMAP from electrophysiology features with subclass-median imputation.

    Strategy:
    1. Select all numeric ephys features where every major subclass (>=10 cells)
       has at least 5 non-NaN observations (82 features).
    2. Impute missing values with the subclass median for that feature.
    3. Exclude QC/fail features and near-zero-variance features.
    4. Z-score → PCA → UMAP.

    Cells in minor subclasses (<10 cells) or with no subclass label are imputed
    using the global median. Cells with >50% of features imputed are flagged in
    obs['ephys_high_imputation'] for downstream quality checks.
    """
    print("\nComputing ephys UMAP (subclass-median imputation)...")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from umap import UMAP

    # Load the full joined ephys table
    ephys_csv = PROJECT_ROOT / "data" / "patchseq" / "patchseq_ephys_joined.csv"
    ephys_df = pd.read_csv(ephys_csv)
    ephys_df = ephys_df.set_index("specimen_id")

    # Identify numeric feature columns (exclude metadata/QC)
    exclude_cols = {"dataset"}
    feat_cols = [c for c in ephys_df.columns
                 if c not in exclude_cols
                 and ephys_df[c].dtype in ["float64", "int64", "float32"]
                 and "qc" not in c.lower()
                 and "fail" not in c.lower()]
    print(f"  Numeric non-QC features: {len(feat_cols)}")

    # Get subclass labels for ephys cells
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

    # Identify major subclasses (>= 10 cells)
    sub_counts = ephys_df["_subclass"].value_counts()
    major_subs = sub_counts[sub_counts >= 10].index.tolist()
    print(f"  Major subclasses (>= 10 cells): {major_subs}")

    # Select features where ALL major subclasses have >= 5 non-NaN values
    ephys_major = ephys_df[ephys_df["_subclass"].isin(major_subs)]
    imputable_feats = []
    for feat in feat_cols:
        sub_coverage = ephys_major.groupby("_subclass")[feat].apply(
            lambda x: x.notna().sum()
        )
        if (sub_coverage >= 5).all():
            imputable_feats.append(feat)
    print(f"  Features imputable by subclass (all major subs >= 5 obs): "
          f"{len(imputable_feats)}")

    # Remove near-zero-variance features
    temp_data = ephys_df[imputable_feats].dropna()
    stds = temp_data.std()
    low_var = stds[stds < 1e-10].index.tolist()
    if low_var:
        print(f"  Removing {len(low_var)} zero-variance features: {low_var}")
        imputable_feats = [f for f in imputable_feats if f not in low_var]
    print(f"  Final feature count: {len(imputable_feats)}")

    # Map ephys specimen_ids to adata indices
    if "specimen_id" in adata.obs.columns:
        adata_specimen_ids = adata.obs["specimen_id"].values
    else:
        adata_specimen_ids = adata.obs_names.values

    # Build raw ephys matrix aligned to adata (before imputation)
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

    # Identify cells with ANY ephys data
    has_any_ephys = ~np.isnan(ephys_raw).all(axis=1)
    n_with_ephys = has_any_ephys.sum()
    print(f"  Cells with any ephys data: {n_with_ephys}/{len(adata)}")

    # Count missing values per cell (before imputation)
    n_missing_per_cell = np.isnan(ephys_raw[has_any_ephys]).sum(axis=1)
    frac_missing = n_missing_per_cell / n_feats
    print(f"  Missing fraction per cell: "
          f"median={np.median(frac_missing):.1%}, "
          f"mean={np.mean(frac_missing):.1%}, "
          f"max={np.max(frac_missing):.1%}")

    # Compute subclass medians for imputation
    subclass_medians = {}
    for sub in major_subs:
        mask = ephys_df["_subclass"] == sub
        subclass_medians[sub] = ephys_df.loc[mask, imputable_feats].median().values
    global_median = ephys_df[imputable_feats].median().values

    # Impute missing values
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
          f"({n_imputed_total/total_vals:.1%})")

    # Check for any remaining NaN (shouldn't happen)
    remaining_nan = np.isnan(ephys_imputed[has_any_ephys]).sum()
    if remaining_nan > 0:
        print(f"  WARNING: {remaining_nan} NaN values remain after imputation")
        # Fall back: fill remaining with global median
        for j in range(n_feats):
            col = ephys_imputed[has_any_ephys, j]
            nan_idx = np.isnan(col)
            if nan_idx.any():
                col[nan_idx] = global_median[j]

    # Flag heavily-imputed cells
    frac_imputed_per_cell = np.full(len(adata), np.nan)
    frac_imputed_per_cell[has_any_ephys] = frac_missing
    adata.obs["ephys_frac_imputed"] = frac_imputed_per_cell
    adata.obs["ephys_high_imputation"] = frac_imputed_per_cell > 0.50
    n_high_imp = (frac_imputed_per_cell > 0.50).sum()
    print(f"  Cells with >50% imputed: {n_high_imp}")

    if n_with_ephys < 20:
        print("  Too few cells for ephys UMAP, skipping.")
        return adata

    # Z-score normalize (using only cells with ephys)
    ephys_valid = ephys_imputed[has_any_ephys]
    scaler = StandardScaler()
    ephys_scaled = scaler.fit_transform(ephys_valid)

    # Store scaled features in obsm
    ephys_scaled_full = np.full((len(adata), n_feats), np.nan)
    ephys_scaled_full[has_any_ephys] = ephys_scaled
    adata.obsm["X_ephys_scaled"] = ephys_scaled_full

    # PCA
    n_pcs = min(30, n_feats - 1, n_with_ephys - 1)
    pca = PCA(n_components=n_pcs, random_state=42)
    ephys_pca = pca.fit_transform(ephys_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_pcs} components, {var_explained*100:.1f}% variance explained")

    # Store PCA in obsm
    ephys_pca_full = np.full((len(adata), n_pcs), np.nan)
    ephys_pca_full[has_any_ephys] = ephys_pca
    adata.obsm["X_ephys_pca"] = ephys_pca_full

    # UMAP on PCA
    reducer = UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    ephys_umap = reducer.fit_transform(ephys_pca)

    full_ephys_umap = np.full((len(adata), 2), np.nan)
    full_ephys_umap[has_any_ephys] = ephys_umap
    adata.obsm["X_umap_ephys"] = full_ephys_umap

    # Store metadata
    adata.uns["ephys_features"] = imputable_feats
    adata.uns["ephys_n_pcs"] = n_pcs
    adata.uns["ephys_pca_var_explained"] = float(var_explained)
    adata.uns["ephys_imputation_method"] = "subclass_median"
    adata.uns["ephys_n_imputed_values"] = int(n_imputed_total)

    print(f"  Ephys UMAP computed for {n_with_ephys} cells "
          f"({n_feats} features, subclass-median imputation → "
          f"{n_pcs} PCs → 2D UMAP)")
    return adata


def assign_subclass_labels(adata):
    """Assign consensus subclass labels from scANVI + original annotations.

    Uses scANVI where available, falls back to original subclass labels.
    Must be called BEFORE compute_ephys_umap so imputation can use subclass.
    """
    labels = adata.obs["subclass_scANVI"].copy()
    missing = labels.isna()
    if "subclass_label_original" in adata.obs.columns:
        orig = adata.obs.loc[missing, "subclass_label_original"]
        subclass_map = {
            "SST": "Sst",
            "PVALB": "Pvalb",
            "VIP": "Vip",
            "LAMP5/PAX6/Other": "Lamp5/Pax6",
        }
        labels.loc[missing] = orig.map(subclass_map).fillna(orig)
    adata.obs["subclass_label"] = labels.fillna("Unknown")
    print(f"\n  Subclass labels assigned: "
          f"{(adata.obs['subclass_label'] != 'Unknown').sum()}/{len(adata)} cells")
    return adata


def plot_umaps(adata):
    """Generate UMAP plots colored by scANVI subclass."""
    print("\nGenerating UMAP plots...")

    # Color palette for subclasses
    subclass_colors = {
        "Sst": "#FF6B6B",
        "Pvalb": "#4ECDC4",
        "Vip": "#45B7D1",
        "Lamp5": "#96CEB4",
        "Sncg": "#FFEAA7",
        "Pax6": "#DDA0DD",
        "L4 IT": "#98D8C8",
        "Lamp5_Lhx6": "#7FCDBB",
    }

    # ── Expression UMAP ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    umap_coords = adata.obsm["X_umap"]

    for subclass in sorted(adata.obs["subclass_label"].unique()):
        mask = adata.obs["subclass_label"] == subclass
        color = subclass_colors.get(subclass, "#CCCCCC")
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=color, s=15, alpha=0.7, label=f"{subclass} (n={mask.sum()})",
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title("Combined Patch-Seq: Gene Expression UMAP", fontsize=20)
    ax.legend(fontsize=12, markerscale=2, frameon=True, loc="best")
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "patchseq_umap_expression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: patchseq_umap_expression.png")

    # ── Ephys UMAP ───────────────────────────────────────────────────
    if "X_umap_ephys" in adata.obsm:
        ephys_umap = adata.obsm["X_umap_ephys"]
        has_ephys = ~np.isnan(ephys_umap[:, 0])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for subclass in sorted(adata.obs["subclass_label"].unique()):
            mask = (adata.obs["subclass_label"] == subclass) & has_ephys
            if mask.sum() == 0:
                continue
            color = subclass_colors.get(subclass, "#CCCCCC")
            ax.scatter(
                ephys_umap[mask, 0], ephys_umap[mask, 1],
                c=color, s=15, alpha=0.7, label=f"{subclass} (n={mask.sum()})",
                edgecolors="none",
            )

        ax.set_xlabel("UMAP 1", fontsize=16)
        ax.set_ylabel("UMAP 2", fontsize=16)
        ax.set_title("Combined Patch-Seq: Electrophysiology UMAP", fontsize=20)
        ax.legend(fontsize=12, markerscale=2, frameon=True, loc="best")
        ax.tick_params(labelsize=13)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "patchseq_umap_ephys.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: patchseq_umap_ephys.png")

    # ── Expression UMAP colored by dataset ───────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    dataset_colors = {"LeeDalley": "#E74C3C", "L1": "#3498DB", "both": "#2ECC71"}

    for ds in ["LeeDalley", "L1", "both"]:
        mask = adata.obs["dataset"] == ds
        if mask.sum() == 0:
            continue
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=dataset_colors[ds], s=15, alpha=0.6,
            label=f"{ds} (n={mask.sum()})", edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title("Combined Patch-Seq: Dataset of Origin", fontsize=20)
    ax.legend(fontsize=12, markerscale=2, frameon=True, loc="best")
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "patchseq_umap_dataset.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: patchseq_umap_dataset.png")


def main():
    print("=" * 60)
    print("Building Combined Patch-Seq H5AD")
    print("=" * 60)

    # 1. Load expression
    dat_ld, anno_ld, meta_ld, dat_l1, anno_l1 = load_expression_data()

    # 2. Build cell metadata
    print("\nBuilding cell metadata...")
    ld_meta, l1_meta = build_cell_metadata(anno_ld, meta_ld, anno_l1, dat_ld, dat_l1)

    # 3. Merge expression matrices
    print("\nMerging expression matrices...")
    dat_combined, meta_combined = merge_expression(dat_ld, dat_l1, ld_meta, l1_meta)

    # 4. Add scANVI labels and ephys from joined CSV
    combined_csv = OUTPUT_DIR / "patchseq_combined.csv"
    meta_combined = add_scanvi_and_ephys(meta_combined, combined_csv)

    # 5. Build AnnData
    print("\nBuilding AnnData...")
    adata = build_anndata(dat_combined, meta_combined)

    # 6. Expression UMAP
    adata = compute_expression_umap(adata)

    # 6b. Assign subclass labels (needed for ephys imputation)
    adata = assign_subclass_labels(adata)

    # 7. Ephys UMAP (with subclass-median imputation)
    adata = compute_ephys_umap(adata)

    # 8. Plots
    plot_umaps(adata)

    # 9. Save
    h5ad_path = OUTPUT_DIR / "patchseq_combined.h5ad"
    # Convert string columns to categorical for efficient storage
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype("category")

    adata.write_h5ad(str(h5ad_path))
    print(f"\nSaved: {h5ad_path.name} ({h5ad_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Cells: {adata.shape[0]}")
    print(f"Genes: {adata.shape[1]}")
    print(f"Dataset breakdown:")
    print(adata.obs["dataset"].value_counts().to_string())
    if "subclass_label" in adata.obs.columns:
        print(f"\nSubclass distribution:")
        print(adata.obs["subclass_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
