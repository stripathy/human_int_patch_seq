"""
ephys_features.py — Load and harmonize electrophysiology features from
LeeDalley + L1 datasets.

Each dataset has its own ephys feature CSV. This module merges them into a
single DataFrame on the union of all feature columns, handling the 43 overlap
cells (LD values prioritized, L1 fills missing features).
"""
import logging

import numpy as np
import pandas as pd

from patchseq_builder.config import LD_EPHYS_CSV, L1_EPHYS_CSV
from patchseq_builder.naming import (
    DATASET_LEEDALLEY,
    DATASET_L1,
    DATASET_BOTH,
)

logger = logging.getLogger(__name__)


def harmonize_ephys() -> pd.DataFrame:
    """Load LeeDalley + L1 ephys features and harmonize.

    Returns DataFrame with ~929 cells. Feature columns are the union of
    both datasets; values are NaN where a feature is only measured in the
    other dataset. Overlap cells are merged (LD preferred, L1 fills gaps).

    Columns:
        - specimen_id: int, cell identifier (matches metadata)
        - dataset: str, one of ``"LeeDalley"``, ``"L1"``, or ``"both"``
        - <feature_name>: float, ephys measurement
    """
    ld_ephys = pd.read_csv(str(LD_EPHYS_CSV))
    l1_ephys = pd.read_csv(str(L1_EPHYS_CSV))

    # L1 ephys uses 'cell_name' as the specimen identifier (integer)
    l1_ephys = l1_ephys.rename(columns={"cell_name": "specimen_id"})

    logger.info("Loaded LD ephys: %d cells", len(ld_ephys))
    logger.info("Loaded L1 ephys: %d cells", len(l1_ephys))

    # ── Feature inventory ────────────────────────────────────────────
    ld_features = set(ld_ephys.columns) - {"specimen_id"}
    l1_features = set(l1_ephys.columns) - {"specimen_id"}
    shared_features = sorted(ld_features & l1_features)
    all_features = sorted(ld_features | l1_features)

    logger.info(
        "Ephys features -- shared: %d, LD-only: %d, L1-only: %d",
        len(shared_features),
        len(ld_features - l1_features),
        len(l1_features - ld_features),
    )

    # ── Build per-dataset tables with the full feature union ─────────
    ld_out = ld_ephys[
        ["specimen_id"] + [f for f in all_features if f in ld_ephys.columns]
    ].copy()
    l1_out = l1_ephys[
        ["specimen_id"] + [f for f in all_features if f in l1_ephys.columns]
    ].copy()

    ld_out["dataset"] = DATASET_LEEDALLEY
    l1_out["dataset"] = DATASET_L1

    # ── Overlap handling ─────────────────────────────────────────────
    overlap_ids = set(ld_out["specimen_id"]) & set(l1_out["specimen_id"])
    logger.info("Ephys overlap cells: %d", len(overlap_ids))

    if overlap_ids:
        # Consistency check: verify shared-feature values agree
        ld_ov = ld_out[ld_out["specimen_id"].isin(overlap_ids)].set_index("specimen_id")
        l1_ov = l1_out[l1_out["specimen_id"].isin(overlap_ids)].set_index("specimen_id")
        shared_idx = ld_ov.index.intersection(l1_ov.index)

        n_match, n_diff = 0, 0
        for feat in shared_features:
            if feat in ld_ov.columns and feat in l1_ov.columns:
                ld_vals = ld_ov.loc[shared_idx, feat]
                l1_vals = l1_ov.loc[shared_idx, feat]
                both_valid = ld_vals.notna() & l1_vals.notna()
                if both_valid.any():
                    close = np.allclose(
                        ld_vals[both_valid].values,
                        l1_vals[both_valid].values,
                        rtol=1e-3,
                        equal_nan=True,
                    )
                    if close:
                        n_match += 1
                    else:
                        n_diff += 1

        logger.info(
            "  Overlap feature agreement: %d match, %d differ", n_match, n_diff
        )

        # Merge overlap: start with LD, fill missing from L1
        ld_ov_full = ld_ov.copy()
        for feat in all_features:
            if feat in l1_ov.columns and feat in ld_ov_full.columns:
                mask = ld_ov_full[feat].isna()
                fillable = mask.index[mask].intersection(l1_ov.index)
                ld_ov_full.loc[fillable, feat] = l1_ov.loc[fillable, feat]
            elif feat in l1_ov.columns:
                ld_ov_full[feat] = l1_ov[feat]

        ld_ov_full["dataset"] = DATASET_BOTH
        ld_ov_full = ld_ov_full.reset_index()

        # Remove overlap cells from individual datasets before concat
        ld_out = ld_out[~ld_out["specimen_id"].isin(overlap_ids)]
        l1_out = l1_out[~l1_out["specimen_id"].isin(overlap_ids)]

        ephys_combined = pd.concat(
            [ld_out, l1_out, ld_ov_full], ignore_index=True
        )
    else:
        ephys_combined = pd.concat([ld_out, l1_out], ignore_index=True)

    ephys_combined = ephys_combined.sort_values("specimen_id").reset_index(drop=True)
    logger.info("Total cells with ephys: %d", len(ephys_combined))

    return ephys_combined
