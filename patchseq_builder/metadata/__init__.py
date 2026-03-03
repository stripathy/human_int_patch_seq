"""
patchseq_builder.metadata — Metadata harmonization for the patch-seq pipeline.

Public API:
  harmonize_metadata()    — Load + merge LeeDalley & L1 cell-level metadata
  harmonize_ephys()       — Load + merge LeeDalley & L1 ephys features
  attach_scanvi_labels()  — Add scANVI subclass/supertype to metadata (no h5ad needed)
  build_combined_table()  — Join metadata + key ephys features into one table
"""

from .harmonize import harmonize_metadata
from .ephys_features import harmonize_ephys
from .scanvi import attach_scanvi_labels, build_combined_table
