#!/usr/bin/env python
"""
build_joined_patchseq_tables.py — Join Lee/Dalley and L1 patch-seq datasets.

Produces three harmonized CSV files in data/patchseq/:
  1. patchseq_metadata_joined.csv   — cell-level metadata (all 1,154 unique cells)
  2. patchseq_ephys_joined.csv      — electrophysiology features (shared feature set)
  3. patchseq_combined.csv          — metadata + key ephys + scANVI labels in one table

Data sources:
  - LeeDalley: 779 cells, Lee & Dalley manuscript (MTG patch-seq, multiple layers)
  - L1: 419 cells, Lee et al. Science 2023 (human L1 interneurons)
  - scANVI: 4,549 cells, iterative scANVI label transfer to SEA-AD supertypes
  - 43 cells overlap between LeeDalley and L1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import scanpy as sc


PROJECT_ROOT = Path("/Users/shreejoy/Github/patch_seq_lee")
OUTPUT_DIR = PROJECT_ROOT / "data" / "patchseq"

# ── Input files ──────────────────────────────────────────────────────────
LD_META = OUTPUT_DIR / "LeeDalley_manuscript_metadata_v2.csv"
LD_EPHYS = OUTPUT_DIR / "LeeDalley_ephys_fx.csv"
L1_META = PROJECT_ROOT / "patchseq_human_L1-main" / "data" / "human_l1_dataset_2023_02_06.csv"
L1_EPHYS = PROJECT_ROOT / "patchseq_human_L1-main" / "data" / "aibs_features_E.csv"
SCANVI = OUTPUT_DIR / "iterative_scANVI_results_patchseq_only.2022-11-22.csv"
H5AD = OUTPUT_DIR / "patchseq_combined.h5ad"


def load_and_harmonize_metadata():
    """Load both metadata sources and harmonize to a common schema."""

    ld = pd.read_csv(str(LD_META))
    l1 = pd.read_csv(str(L1_META))

    # ── LeeDalley metadata ───────────────────────────────────────────
    ld_harmonized = pd.DataFrame({
        "specimen_id": ld["specimen_id_x"],
        "cell_name": ld["cell_name"],
        "donor": ld["Donor"],
        "sex": np.nan,  # not in LD metadata
        "age": np.nan,  # not in LD metadata
        "brain_region": ld["structure"],
        "lobe": ld["lobe"],
        "cortical_layer": ld["Cortical_layer"],
        "target_layer": np.nan,  # not in LD
        "disease_category": ld["disease_category"],
        "condition": ld["condition"],
        "days_in_culture": ld["Days_In_culture"],
        "genes_detected": ld["Genes_Detected"],
        "has_ephys": ld["Has_ephys"],
        "has_morphology": ld["Has_morphology"],
        "transcriptomic_type_original": ld["Transcriptomic_type "].str.strip(),
        "subclass_label_original": ld["subclass_label"],
        "revised_subclass_label": ld["Revised_subclass_label"],
        "broad_class": ld["broad_class_label"],
        "cross_species_type": ld["M_H_Hodge_cross_species_types"],
        # L1-specific columns
        "l1_ttype": np.nan,
        "l1_cluster": np.nan,
        "l1_homology_type": np.nan,
        "core_l1_type": np.nan,
        "dendrite_type": np.nan,
        "normalized_depth": np.nan,
        "soma_depth_um": np.nan,
        # Identifiers for linking
        "patched_cell_container": ld["patched_cell_container"],
        "exp_component_name": np.nan,
        "dataset": "LeeDalley",
    })

    # ── L1 metadata ──────────────────────────────────────────────────
    l1_harmonized = pd.DataFrame({
        "specimen_id": l1["spec_id"],
        "cell_name": l1["cell_name"],
        "donor": l1["donor"],
        "sex": l1["donor_sex"].replace("unknown", np.nan),
        "age": l1["donor_age"].replace("unknown", np.nan),
        "brain_region": l1["structure"],
        "lobe": np.nan,  # not directly in L1
        "cortical_layer": l1["layer_lims"],
        "target_layer": l1["target_layer"],
        "disease_category": l1["medical_conditions"].replace("", np.nan),
        "condition": np.nan,  # not in L1 (all are acute production)
        "days_in_culture": np.nan,
        "genes_detected": l1["genes"],
        "has_ephys": l1["has_ephys"],
        "has_morphology": l1["has_morph"],
        "transcriptomic_type_original": l1["cluster"],
        "subclass_label_original": l1["subclass"],
        "revised_subclass_label": np.nan,
        "broad_class": l1["broad_class"],
        "cross_species_type": np.nan,
        # L1-specific columns
        "l1_ttype": l1["t-type"],
        "l1_cluster": l1["cluster"],
        "l1_homology_type": l1["homology_type"],
        "core_l1_type": l1["core_l1_type"],
        "dendrite_type": l1["dendrite_type"].str.replace("dendrite type - ", ""),
        "normalized_depth": l1["normalized_depth"],
        "soma_depth_um": l1["soma_depth_um"],
        # Identifiers for linking
        "patched_cell_container": np.nan,
        "exp_component_name": l1["exp_component_name"],
        "dataset": "L1",
    })

    # ── Merge: outer join on specimen_id ─────────────────────────────
    # 43 cells overlap; for those, keep both records and merge
    overlap_ids = set(ld_harmonized["specimen_id"]) & set(l1_harmonized["specimen_id"])
    print(f"Overlap cells: {len(overlap_ids)}")

    # For overlap cells, fill LD gaps with L1 data
    ld_overlap = ld_harmonized[ld_harmonized["specimen_id"].isin(overlap_ids)].copy()
    l1_overlap = l1_harmonized[l1_harmonized["specimen_id"].isin(overlap_ids)].copy()

    # Merge overlap: start from LD, fill missing columns from L1
    merged_overlap = ld_overlap.set_index("specimen_id")
    l1_ov_idx = l1_overlap.set_index("specimen_id")

    # Fill NaN columns from L1 where LD is missing
    for col in merged_overlap.columns:
        if col == "dataset":
            continue
        mask = merged_overlap[col].isna()
        if mask.any() and col in l1_ov_idx.columns:
            merged_overlap.loc[mask, col] = l1_ov_idx.loc[
                mask.index[mask].intersection(l1_ov_idx.index), col
            ]

    # Also grab L1-specific columns
    for col in ["l1_ttype", "l1_cluster", "l1_homology_type", "core_l1_type",
                "dendrite_type", "normalized_depth", "soma_depth_um",
                "exp_component_name"]:
        if col in l1_ov_idx.columns:
            shared_idx = merged_overlap.index.intersection(l1_ov_idx.index)
            merged_overlap.loc[shared_idx, col] = l1_ov_idx.loc[shared_idx, col]

    merged_overlap["dataset"] = "both"
    merged_overlap = merged_overlap.reset_index()

    # Non-overlap cells
    ld_only = ld_harmonized[~ld_harmonized["specimen_id"].isin(overlap_ids)]
    l1_only = l1_harmonized[~l1_harmonized["specimen_id"].isin(overlap_ids)]

    # Concatenate all
    combined = pd.concat([ld_only, l1_only, merged_overlap], ignore_index=True)
    combined = combined.sort_values("specimen_id").reset_index(drop=True)

    print(f"Total unique cells: {len(combined)}")
    print(f"  LeeDalley-only: {(combined['dataset'] == 'LeeDalley').sum()}")
    print(f"  L1-only:        {(combined['dataset'] == 'L1').sum()}")
    print(f"  Both:           {(combined['dataset'] == 'both').sum()}")

    return combined


def load_and_harmonize_ephys():
    """Load both ephys sources and harmonize to shared feature columns."""

    ld_ephys = pd.read_csv(str(LD_EPHYS))
    l1_ephys = pd.read_csv(str(L1_EPHYS))

    # L1 ephys uses 'cell_name' as specimen_id (integer)
    l1_ephys = l1_ephys.rename(columns={"cell_name": "specimen_id"})

    # Find shared feature columns
    ld_features = set(ld_ephys.columns) - {"specimen_id"}
    l1_features = set(l1_ephys.columns) - {"specimen_id"}
    shared_features = sorted(ld_features & l1_features)

    print(f"\nEphys features — shared: {len(shared_features)}, "
          f"LD-only: {len(ld_features - l1_features)}, "
          f"L1-only: {len(l1_features - ld_features)}")

    # Use shared features + all unique features
    all_features = sorted(ld_features | l1_features)

    # Build harmonized ephys
    ld_out = ld_ephys[["specimen_id"] + [f for f in all_features if f in ld_ephys.columns]].copy()
    l1_out = l1_ephys[["specimen_id"] + [f for f in all_features if f in l1_ephys.columns]].copy()

    ld_out["dataset"] = "LeeDalley"
    l1_out["dataset"] = "L1"

    # For overlap cells, prefer LD ephys (they should be identical)
    overlap_ids = set(ld_out["specimen_id"]) & set(l1_out["specimen_id"])
    print(f"Ephys overlap cells: {len(overlap_ids)}")

    if overlap_ids:
        # Check consistency for overlap cells on shared features
        ld_ov = ld_out[ld_out["specimen_id"].isin(overlap_ids)].set_index("specimen_id")
        l1_ov = l1_out[l1_out["specimen_id"].isin(overlap_ids)].set_index("specimen_id")
        shared_idx = ld_ov.index.intersection(l1_ov.index)

        n_match = 0
        n_diff = 0
        for feat in shared_features:
            if feat in ld_ov.columns and feat in l1_ov.columns:
                ld_vals = ld_ov.loc[shared_idx, feat]
                l1_vals = l1_ov.loc[shared_idx, feat]
                # Compare non-NaN values
                both_valid = ld_vals.notna() & l1_vals.notna()
                if both_valid.any():
                    close = np.allclose(
                        ld_vals[both_valid].values,
                        l1_vals[both_valid].values,
                        rtol=1e-3, equal_nan=True,
                    )
                    if close:
                        n_match += 1
                    else:
                        n_diff += 1
        print(f"  Overlap feature agreement: {n_match} match, {n_diff} differ")

        # For overlap cells, merge: start with LD, fill missing from L1
        ld_ov_full = ld_ov.copy()
        for feat in all_features:
            if feat in l1_ov.columns and feat in ld_ov_full.columns:
                mask = ld_ov_full[feat].isna()
                ld_ov_full.loc[mask, feat] = l1_ov.loc[mask.index[mask].intersection(l1_ov.index), feat]
            elif feat in l1_ov.columns:
                ld_ov_full[feat] = l1_ov[feat]

        ld_ov_full["dataset"] = "both"
        ld_ov_full = ld_ov_full.reset_index()

        # Remove overlap from individual datasets
        ld_out = ld_out[~ld_out["specimen_id"].isin(overlap_ids)]
        l1_out = l1_out[~l1_out["specimen_id"].isin(overlap_ids)]

        ephys_combined = pd.concat([ld_out, l1_out, ld_ov_full], ignore_index=True)
    else:
        ephys_combined = pd.concat([ld_out, l1_out], ignore_index=True)

    ephys_combined = ephys_combined.sort_values("specimen_id").reset_index(drop=True)
    print(f"Total cells with ephys: {len(ephys_combined)}")

    return ephys_combined


def _fill_scanvi_from_h5ad(meta, scanvi_cols):
    """Fallback: pull scANVI labels from patchseq_combined.h5ad for cells
    that are missing them (e.g. LeeDalley cells without exp_component_name).

    The h5ad was built with specimen_id-based matching and has ~96% scANVI
    coverage, so this fills the gap left by the exp_component_name-only
    join in the CSV pipeline.
    """
    if not H5AD.exists():
        print(f"  WARNING: {H5AD} not found, skipping h5ad fallback")
        return meta

    adata = sc.read_h5ad(str(H5AD))
    h5_labels = adata.obs[["specimen_id"] + scanvi_cols].copy()
    h5_labels = h5_labels[h5_labels["subclass_scANVI"].notna()]
    h5_labels["specimen_id"] = h5_labels["specimen_id"].astype(int)
    h5_lookup = h5_labels.set_index("specimen_id")

    missing_mask = meta["subclass_scANVI"].isna()
    filled = 0
    for idx in meta.index[missing_mask]:
        sid = meta.loc[idx, "specimen_id"]
        if sid in h5_lookup.index:
            for col in scanvi_cols:
                meta.loc[idx, col] = h5_lookup.loc[sid, col]
            filled += 1

    print(f"  h5ad fallback filled: {filled} additional cells")
    return meta


def add_scanvi_labels(metadata):
    """Add scANVI supertype/subclass labels where available.

    Two-step approach:
      1. Primary: match via exp_component_name to the scANVI results CSV
         (works for L1 cells that have exp_component_name)
      2. Fallback: match via specimen_id to patchseq_combined.h5ad
         (covers LeeDalley cells that lack exp_component_name)
    """
    scanvi_cols = ["subclass_scANVI", "subclass_conf_scANVI",
                   "supertype_scANVI", "supertype_conf_scANVI"]

    scanvi = pd.read_csv(str(SCANVI), index_col=0)

    meta = metadata.copy()
    for col in scanvi_cols:
        meta[col] = np.nan

    # Step 1: match via exp_component_name (original approach)
    has_ecn = meta["exp_component_name"].notna()
    ecn_values = meta.loc[has_ecn, "exp_component_name"]
    in_scanvi = ecn_values.isin(scanvi.index)

    matched_ecn = ecn_values[in_scanvi]
    for col in scanvi_cols:
        meta.loc[matched_ecn.index, col] = scanvi.loc[matched_ecn.values, col].values

    n_step1 = meta["supertype_scANVI"].notna().sum()
    print(f"\nscANVI labels — step 1 (exp_component_name): {n_step1} / {len(meta)} cells")

    # Step 2: fallback to h5ad for remaining missing cells
    meta = _fill_scanvi_from_h5ad(meta, scanvi_cols)

    n_total = meta["supertype_scANVI"].notna().sum()
    print(f"scANVI labels — total: {n_total} / {len(meta)} cells ({100*n_total/len(meta):.1f}%)")
    print(f"  By dataset:")
    for ds in ["LeeDalley", "L1", "both"]:
        mask = meta["dataset"] == ds
        n = meta.loc[mask, "supertype_scANVI"].notna().sum()
        print(f"    {ds}: {n}/{mask.sum()}")

    return meta


def build_combined_table(metadata, ephys):
    """Build a single wide table with metadata + key ephys features + scANVI."""

    # Key ephys features to include in the combined table
    key_ephys = [
        "sag", "tau", "input_resistance", "rheobase_i", "fi_fit_slope",
        "v_baseline", "avg_rate_hero",
        "upstroke_downstroke_ratio_short_square",
        "threshold_v_short_square", "width_short_square",
        "adapt_hero", "latency_rheo",
        "sag_tau", "sag_area",
        "peak_freq_chirp", "peak_ratio_chirp",
    ]
    available = [f for f in key_ephys if f in ephys.columns]

    ephys_subset = ephys[["specimen_id"] + available].copy()

    combined = metadata.merge(ephys_subset, on="specimen_id", how="left")

    return combined


def main():
    print("=" * 60)
    print("Building Joined Patch-Seq Tables")
    print("=" * 60)

    # Step 1: Metadata
    print("\n── Metadata ──")
    metadata = load_and_harmonize_metadata()

    # Step 2: Ephys
    print("\n── Electrophysiology ──")
    ephys = load_and_harmonize_ephys()

    # Step 3: scANVI
    print("\n── scANVI Labels ──")
    metadata = add_scanvi_labels(metadata)

    # Step 4: Combined table
    print("\n── Combined Table ──")
    combined = build_combined_table(metadata, ephys)

    # ── Save ─────────────────────────────────────────────────────────
    meta_path = OUTPUT_DIR / "patchseq_metadata_joined.csv"
    ephys_path = OUTPUT_DIR / "patchseq_ephys_joined.csv"
    combined_path = OUTPUT_DIR / "patchseq_combined.csv"

    metadata.to_csv(str(meta_path), index=False)
    ephys.to_csv(str(ephys_path), index=False)
    combined.to_csv(str(combined_path), index=False)

    print(f"\n── Saved ──")
    print(f"  {meta_path.name}: {metadata.shape}")
    print(f"  {ephys_path.name}: {ephys.shape}")
    print(f"  {combined_path.name}: {combined.shape}")

    # ── Summary stats ────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"Total unique cells: {len(combined)}")
    print(f"  with ephys: {combined['sag'].notna().sum()}")
    print(f"  with scANVI supertype: {combined['supertype_scANVI'].notna().sum()}")
    print(f"  with both: {(combined['sag'].notna() & combined['supertype_scANVI'].notna()).sum()}")

    print(f"\nDataset breakdown:")
    for ds in ["LeeDalley", "L1", "both"]:
        mask = combined["dataset"] == ds
        sub = combined[mask]
        print(f"  {ds:12s}: {mask.sum():4d} cells, "
              f"{sub['sag'].notna().sum():3d} with ephys, "
              f"{sub['supertype_scANVI'].notna().sum():3d} with scANVI")

    print(f"\nSubclass distribution (scANVI-labeled cells):")
    labeled = combined[combined["subclass_scANVI"].notna()]
    print(labeled["subclass_scANVI"].value_counts().to_string())

    print(f"\nSubclass distribution (original labels, all cells):")
    print(combined["subclass_label_original"].value_counts().to_string())

    print(f"\nDonors: {combined['donor'].nunique()}")
    print(f"Brain regions: {combined['brain_region'].nunique()}")


if __name__ == "__main__":
    main()
