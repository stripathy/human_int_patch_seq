"""patchseq_builder — Clean, reproducible pipeline for patch-seq data preparation.

Builds harmonized metadata, expression matrices, UMAPs, morphology SVGs,
trace mappings, and an interactive viewer from raw patch-seq data sources.

Usage:
    python scripts/build_patchseq_viewer.py          # full pipeline
    python scripts/build_patchseq_viewer.py --from 5  # rebuild from stage 5 (viewer)
    python scripts/build_patchseq_viewer.py --only 1  # just metadata
"""

__version__ = "1.0.0"
