"""
harmonize.py — Load and harmonize cell-level metadata from LeeDalley + L1 datasets.

Reads two raw metadata CSVs (LeeDalley manuscript metadata, L1 human dataset)
and merges them into a single DataFrame with a common schema. The 43 cells
present in both datasets are merged (LD columns prioritized, L1 fills gaps).

Output: 1,155 unique cells with columns covering donor info, brain region,
transcriptomic labels, L1-specific annotations, and dataset linking IDs.
"""
import logging

import numpy as np
import pandas as pd

from patchseq_builder.config import LD_METADATA_CSV, L1_METADATA_CSV
from patchseq_builder.naming import (
    DATASET_LEEDALLEY,
    DATASET_L1,
    DATASET_BOTH,
    normalize_dendrite_type,
)

logger = logging.getLogger(__name__)


def harmonize_metadata() -> pd.DataFrame:
    """Load LeeDalley + L1 metadata and harmonize to common schema.

    Returns DataFrame with ~1,155 cells. Each row is one unique cell, with
    ``dataset`` indicating its source (``"LeeDalley"``, ``"L1"``, or ``"both"``
    for the 43 overlap cells).

    Column groups:
        - Identity: specimen_id, cell_name, donor, patched_cell_container,
          exp_component_name
        - Demographics: sex, age, disease_category, condition
        - Anatomy: brain_region, lobe, cortical_layer, target_layer,
          normalized_depth, soma_depth_um
        - QC: genes_detected, has_ephys, has_morphology, days_in_culture
        - Transcriptomics: transcriptomic_type_original,
          subclass_label_original, revised_subclass_label, broad_class,
          cross_species_type
        - L1-specific: l1_ttype, l1_cluster, l1_homology_type, core_l1_type,
          dendrite_type
    """
    ld = pd.read_csv(str(LD_METADATA_CSV))
    l1 = pd.read_csv(str(L1_METADATA_CSV))

    # Remove exact duplicate rows in source data (e.g. cell h19.03.002.12.01.01.01.04)
    n_before = len(ld)
    ld = ld.drop_duplicates(subset=["cell_name"], keep="first")
    if len(ld) < n_before:
        logger.info("Dropped %d duplicate rows from LeeDalley metadata", n_before - len(ld))

    logger.info("Loaded LeeDalley metadata: %d cells", len(ld))
    logger.info("Loaded L1 metadata: %d cells", len(l1))

    # ── LeeDalley metadata ───────────────────────────────────────────
    # NOTE: The raw LeeDalley CSV has a trailing space in the column name
    # "Transcriptomic_type " — we strip it during extraction rather than
    # renaming the source file to avoid touching upstream data.
    ld_harmonized = pd.DataFrame({
        "specimen_id": ld["specimen_id_x"],
        "cell_name": ld["cell_name"],
        "donor": ld["Donor"],
        "sex": np.nan,                       # not available in LD metadata
        "age": np.nan,                       # not available in LD metadata
        "brain_region": ld["structure"],
        "lobe": ld["lobe"],
        "cortical_layer": ld["Cortical_layer"],
        "target_layer": np.nan,              # not in LD
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
        # L1-specific columns — absent in LD
        "l1_ttype": np.nan,
        "l1_cluster": np.nan,
        "l1_homology_type": np.nan,
        "core_l1_type": np.nan,
        "dendrite_type": np.nan,
        "normalized_depth": np.nan,
        "soma_depth_um": np.nan,
        # Identifiers for downstream linking (scANVI, expression, etc.)
        "lab": "AIBS",                       # all LeeDalley cells are AIBS
        "patched_cell_container": ld["patched_cell_container"],
        "exp_component_name": np.nan,
        "dataset": DATASET_LEEDALLEY,
    })

    # ── L1 metadata ──────────────────────────────────────────────────
    l1_harmonized = pd.DataFrame({
        "specimen_id": l1["spec_id"],
        "cell_name": l1["cell_name"],
        "donor": l1["donor"],
        "sex": l1["donor_sex"].replace("unknown", np.nan),
        "age": l1["donor_age"].replace("unknown", np.nan),
        "brain_region": l1["structure"],
        "lobe": np.nan,                      # not directly in L1
        "cortical_layer": l1["layer_lims"],
        "target_layer": l1["target_layer"],
        "disease_category": l1["medical_conditions"].replace("", np.nan),
        "condition": np.nan,                 # not in L1 (all are acute production)
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
        "dendrite_type": l1["dendrite_type"].map(normalize_dendrite_type),
        "normalized_depth": l1["normalized_depth"],
        "soma_depth_um": l1["soma_depth_um"],
        # Lab / collaborator
        "lab": l1["collaborator"].map(
            {"Huib": "Mansvelder", "Gabor": "Tamás"}
        ).fillna("AIBS"),
        # Identifiers for downstream linking
        "patched_cell_container": np.nan,
        "exp_component_name": l1["exp_component_name"],
        "dataset": DATASET_L1,
    })

    # ── Handle overlap (cells present in both datasets) ──────────────
    overlap_ids = set(ld_harmonized["specimen_id"]) & set(l1_harmonized["specimen_id"])
    logger.info("Overlap cells: %d", len(overlap_ids))

    # For overlap cells: start from LD record, fill missing values from L1
    ld_overlap = ld_harmonized[ld_harmonized["specimen_id"].isin(overlap_ids)].copy()
    l1_overlap = l1_harmonized[l1_harmonized["specimen_id"].isin(overlap_ids)].copy()

    merged_overlap = ld_overlap.set_index("specimen_id")
    l1_ov_idx = l1_overlap.set_index("specimen_id")

    # Fill NaN columns from L1 where LD is missing
    for col in merged_overlap.columns:
        if col == "dataset":
            continue
        mask = merged_overlap[col].isna()
        if mask.any() and col in l1_ov_idx.columns:
            shared = mask.index[mask].intersection(l1_ov_idx.index)
            merged_overlap.loc[shared, col] = l1_ov_idx.loc[shared, col]

    # Explicitly copy L1-specific columns for overlap cells
    l1_specific_cols = [
        "l1_ttype", "l1_cluster", "l1_homology_type", "core_l1_type",
        "dendrite_type", "normalized_depth", "soma_depth_um",
        "exp_component_name", "lab",
    ]
    for col in l1_specific_cols:
        if col in l1_ov_idx.columns:
            shared_idx = merged_overlap.index.intersection(l1_ov_idx.index)
            merged_overlap.loc[shared_idx, col] = l1_ov_idx.loc[shared_idx, col]

    merged_overlap["dataset"] = DATASET_BOTH
    merged_overlap = merged_overlap.reset_index()

    # ── Concatenate non-overlap + overlap ────────────────────────────
    ld_only = ld_harmonized[~ld_harmonized["specimen_id"].isin(overlap_ids)]
    l1_only = l1_harmonized[~l1_harmonized["specimen_id"].isin(overlap_ids)]

    combined = pd.concat([ld_only, l1_only, merged_overlap], ignore_index=True)
    combined = combined.sort_values("specimen_id").reset_index(drop=True)

    logger.info("Total unique cells: %d", len(combined))
    logger.info(
        "  LeeDalley-only: %d  |  L1-only: %d  |  Both: %d",
        (combined["dataset"] == DATASET_LEEDALLEY).sum(),
        (combined["dataset"] == DATASET_L1).sum(),
        (combined["dataset"] == DATASET_BOTH).sum(),
    )

    return combined
