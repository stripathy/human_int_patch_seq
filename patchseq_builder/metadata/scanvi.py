"""
scanvi.py — Attach scANVI subclass/supertype labels to harmonized metadata.

Key design decision: this module breaks the previous circular dependency on
patchseq_combined.h5ad. Instead of falling back to the h5ad to get scANVI
labels for LeeDalley cells (which themselves were *built* from this metadata),
we extract ``exp_component_name`` directly from the LeeDalley RData annotation
(``annoPatch`` in ``complete_patchseq_data_sets.RData``). This lets us match
ALL cells to the scANVI results CSV purely through exp_component_name — no
h5ad required.

Two-step matching:
  1. L1 cells + overlap cells: already have ``exp_component_name`` in metadata
  2. LeeDalley-only cells: extract ``exp_component_name_label`` from annoPatch
     in the RData, join on ``specimen_id`` (= ``spec_id_label``)
"""
import logging

import numpy as np
import pandas as pd
import pyreadr

from patchseq_builder.config import (
    LD_RDATA,
    SCANVI_RESULTS_CSV,
    KEY_EPHYS_FEATURES,
)
from patchseq_builder.naming import DATASET_LEEDALLEY, DATASET_L1, DATASET_BOTH

logger = logging.getLogger(__name__)

# scANVI columns that we attach to the metadata
SCANVI_COLS = [
    "subclass_scANVI",
    "subclass_conf_scANVI",
    "supertype_scANVI",
    "supertype_conf_scANVI",
]


def _load_leedalley_exp_component_names() -> pd.DataFrame:
    """Extract specimen_id -> exp_component_name mapping from LeeDalley RData.

    Reads ``annoPatch`` from ``complete_patchseq_data_sets.RData`` and returns
    a two-column DataFrame:
        specimen_id        (int)
        exp_component_name (str, the Smart-seq library ID)

    This is the same extraction done in ``build_combined_patchseq_h5ad.py``
    (lines 54-58) but isolated so the metadata pipeline can use it without
    building the full h5ad.
    """
    logger.info("Loading LeeDalley RData for exp_component_name mapping: %s", LD_RDATA)
    rdata = pyreadr.read_r(str(LD_RDATA))
    anno_patch = rdata["annoPatch"]

    mapping = pd.DataFrame({
        "specimen_id": anno_patch["spec_id_label"].astype(str).astype(int),
        "exp_component_name": anno_patch["exp_component_name_label"],
    })

    # Drop rows where exp_component_name is missing or empty
    mapping = mapping[mapping["exp_component_name"].notna()]
    mapping = mapping[mapping["exp_component_name"].astype(str).str.strip() != ""]

    # De-duplicate (annoPatch can have multiple rows per specimen from
    # different annotation columns; specimen_id -> ecn should be 1:1)
    mapping = mapping.drop_duplicates(subset="specimen_id")

    logger.info(
        "  Extracted %d LeeDalley specimen_id -> exp_component_name mappings",
        len(mapping),
    )
    return mapping


def attach_scanvi_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add scANVI subclass/supertype labels to metadata.

    Two-step matching (no h5ad dependency):
      1. L1 cells + overlap ("both") cells: match via ``exp_component_name``
         already present in metadata (populated from L1 CSV or overlap merge).
      2. LeeDalley-only cells: extract ``exp_component_name`` from the RData
         ``annoPatch`` annotation, then match to the scANVI results CSV.

    Parameters
    ----------
    metadata : pd.DataFrame
        Output of :func:`harmonize_metadata` — must have ``specimen_id``,
        ``exp_component_name``, and ``dataset`` columns.

    Returns
    -------
    pd.DataFrame
        A copy of *metadata* with four new columns:
        ``subclass_scANVI``, ``subclass_conf_scANVI``,
        ``supertype_scANVI``, ``supertype_conf_scANVI``.
    """
    scanvi = pd.read_csv(str(SCANVI_RESULTS_CSV), index_col=0)
    meta = metadata.copy()

    # Initialize scANVI columns
    for col in SCANVI_COLS:
        meta[col] = np.nan

    # ── Step 1: match cells that already have exp_component_name ─────
    has_ecn = meta["exp_component_name"].notna()
    ecn_values = meta.loc[has_ecn, "exp_component_name"]
    in_scanvi = ecn_values.isin(scanvi.index)
    matched_ecn = ecn_values[in_scanvi]

    for col in SCANVI_COLS:
        if col in scanvi.columns:
            meta.loc[matched_ecn.index, col] = scanvi.loc[
                matched_ecn.values, col
            ].values

    n_step1 = meta["subclass_scANVI"].notna().sum()
    logger.info(
        "scANVI step 1 (existing exp_component_name): %d / %d cells matched",
        n_step1,
        len(meta),
    )

    # ── Step 2: fill LeeDalley-only cells via RData exp_component_name ─
    ld_ecn_map = _load_leedalley_exp_component_names()
    ecn_lookup = ld_ecn_map.set_index("specimen_id")["exp_component_name"]

    # Target: cells still missing scANVI labels
    still_missing = meta["subclass_scANVI"].isna()
    candidates = meta.index[still_missing]

    n_step2 = 0
    for idx in candidates:
        sid = meta.loc[idx, "specimen_id"]
        if sid in ecn_lookup.index:
            ecn = ecn_lookup[sid]
            if ecn in scanvi.index:
                for col in SCANVI_COLS:
                    if col in scanvi.columns:
                        meta.loc[idx, col] = scanvi.at[ecn, col]
                # Also backfill exp_component_name into metadata if it was
                # missing (LeeDalley-only cells)
                if pd.isna(meta.loc[idx, "exp_component_name"]):
                    meta.loc[idx, "exp_component_name"] = ecn
                n_step2 += 1

    logger.info(
        "scANVI step 2 (RData exp_component_name): %d additional cells matched",
        n_step2,
    )

    # ── Summary ──────────────────────────────────────────────────────
    n_total = meta["subclass_scANVI"].notna().sum()
    logger.info(
        "scANVI labels total: %d / %d cells (%.1f%%)",
        n_total,
        len(meta),
        100 * n_total / len(meta),
    )
    for ds in [DATASET_LEEDALLEY, DATASET_L1, DATASET_BOTH]:
        ds_mask = meta["dataset"] == ds
        n_ds = meta.loc[ds_mask, "subclass_scANVI"].notna().sum()
        logger.info("  %s: %d / %d", ds, n_ds, ds_mask.sum())

    return meta


def build_combined_table(
    metadata: pd.DataFrame, ephys: pd.DataFrame
) -> pd.DataFrame:
    """Merge metadata + key ephys features into one table.

    Performs a left join of *metadata* with a subset of *ephys* columns
    (the 16 key features defined in :data:`patchseq_builder.config.KEY_EPHYS_FEATURES`),
    joined on ``specimen_id``.

    Parameters
    ----------
    metadata : pd.DataFrame
        Output of :func:`attach_scanvi_labels` (metadata with scANVI columns).
    ephys : pd.DataFrame
        Output of :func:`~patchseq_builder.metadata.ephys_features.harmonize_ephys`.

    Returns
    -------
    pd.DataFrame
        Wide table with all metadata columns plus available key ephys features.
    """
    available = [f for f in KEY_EPHYS_FEATURES if f in ephys.columns]
    ephys_subset = ephys[["specimen_id"] + available].copy()

    combined = metadata.merge(ephys_subset, on="specimen_id", how="left")

    logger.info(
        "Combined table: %d cells x %d columns (%d ephys features included)",
        len(combined),
        combined.shape[1],
        len(available),
    )
    return combined
