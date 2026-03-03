#!/usr/bin/env python3
"""build_patchseq_viewer.py — Build the complete patch-seq data pipeline.

Stages:
  1. Metadata harmonization (CSV-only, no h5ad dependency)
  2. Build h5ad with expression + UMAPs
  3. Reference integration + kNN labels
  4. Morphology download + render
  5. Trace SVG mapping + symlinks
  6. Build interactive viewer
  7. Generate comprehensive accounting

Usage:
  python scripts/build_patchseq_viewer.py           # full pipeline
  python scripts/build_patchseq_viewer.py --from 5   # rebuild from stage 5
  python scripts/build_patchseq_viewer.py --only 1   # just metadata
  python scripts/build_patchseq_viewer.py --force     # force regenerate cached outputs
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from patchseq_builder.config import (
    COMBINED_CSV,
    COMPREHENSIVE_METADATA_CSV,
    DATA_ACCOUNTING_TXT,
    EPHYS_JOINED_CSV,
    FIGURES_DIR,
    INTERMEDIATES_DIR,
    KNN_LABELS_CSV,
    METADATA_JOINED_CSV,
    MISSING_EPHYS_CSV,
    MORPHOLOGY_SVG_DIR,
    MORPHOLOGY_SVG_MAP_CSV,
    PATCHSEQ_H5AD,
    REFERENCE_COMBINED_H5AD,
    RESULTS_DIR,
    SWC_DIR,
    TABLES_DIR,
    TRACE_SVG_MAP_CSV,
    VIEWER_HTML,
)

# Configure logging so all patchseq_builder modules emit to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("pipeline")

N_STAGES = 7


# ============================================================================
# Utility helpers
# ============================================================================

def _header(stage_num: int, title: str):
    """Print a prominent stage header."""
    print(f"\n{'=' * 70}")
    print(f"  STAGE {stage_num}/{N_STAGES}: {title}")
    print(f"{'=' * 70}\n")


def _ensure_dirs():
    """Create all output directories if they don't already exist."""
    for d in [RESULTS_DIR, FIGURES_DIR, INTERMEDIATES_DIR, TABLES_DIR,
              SWC_DIR, MORPHOLOGY_SVG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _output_exists(path: Path) -> bool:
    """Check whether an output file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


# ============================================================================
# Stage 1: Metadata harmonization
# ============================================================================

def stage_1_metadata(force: bool = False):
    """Harmonize metadata, ephys, scANVI labels, and build combined table."""

    outputs = [METADATA_JOINED_CSV, EPHYS_JOINED_CSV, COMBINED_CSV]
    if not force and all(_output_exists(p) for p in outputs):
        print(f"  All metadata outputs exist, skipping. Use --force to regenerate.")
        return

    from patchseq_builder.metadata.harmonize import harmonize_metadata
    from patchseq_builder.metadata.ephys_features import harmonize_ephys
    from patchseq_builder.metadata.scanvi import attach_scanvi_labels, build_combined_table

    # 1a. Harmonize cell-level metadata
    print("Harmonizing cell-level metadata...")
    metadata = harmonize_metadata()
    metadata.to_csv(METADATA_JOINED_CSV, index=False)
    print(f"  Saved: {METADATA_JOINED_CSV} ({len(metadata)} cells)")

    # 1b. Harmonize ephys features
    print("\nHarmonizing ephys features...")
    ephys = harmonize_ephys()
    ephys.to_csv(EPHYS_JOINED_CSV, index=False)
    print(f"  Saved: {EPHYS_JOINED_CSV} ({len(ephys)} cells)")

    # 1c. Attach scANVI labels (requires RData but NOT h5ad)
    print("\nAttaching scANVI labels...")
    metadata = attach_scanvi_labels(metadata)

    # 1d. Build combined table (metadata + key ephys features)
    print("\nBuilding combined metadata + ephys table...")
    combined = build_combined_table(metadata, ephys)
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"  Saved: {COMBINED_CSV} ({len(combined)} cells x {combined.shape[1]} cols)")

    # 1e. Validate
    from patchseq_builder.validate import validate_metadata, validate_ephys
    report_meta = validate_metadata(METADATA_JOINED_CSV)
    report_meta.print_summary()
    report_meta.assert_ok()

    report_ephys = validate_ephys(EPHYS_JOINED_CSV)
    report_ephys.print_summary()
    report_ephys.assert_ok()


# ============================================================================
# Stage 2: Build h5ad with expression + UMAPs
# ============================================================================

def stage_2_build_h5ad(force: bool = False):
    """Load expression from RData, merge, compute UMAPs, save h5ad."""

    if not force and _output_exists(PATCHSEQ_H5AD):
        print(f"  h5ad exists: {PATCHSEQ_H5AD}, skipping. Use --force to regenerate.")
        return

    import pandas as pd
    import scanpy as sc

    from patchseq_builder.expression.load_rdata import (
        load_leedalley_expression,
        load_l1_expression,
        merge_expression,
    )
    from patchseq_builder.expression.umap import compute_expression_umap, compute_ephys_umap

    # 2a. Load expression matrices
    ld_expr, ld_genes, ld_cells = load_leedalley_expression()
    l1_expr, l1_genes, l1_cells = load_l1_expression()

    # 2b. Load metadata for obs (produced by stage 1)
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {COMBINED_CSV}\n"
            f"Run stage 1 first (--from 1) or provide the combined metadata CSV."
        )
    metadata = pd.read_csv(COMBINED_CSV)
    # The merge_expression function expects metadata indexed by cell identifier
    # (exp_component_name for expression data). Build a lookup.
    # Cell identifiers in the RData are exp_component_name values.
    meta_indexed = metadata.copy()
    if "exp_component_name" in meta_indexed.columns:
        # Use exp_component_name as index where available, fall back to specimen_id
        meta_indexed = meta_indexed.set_index("exp_component_name", drop=False)

    # 2c. Merge into AnnData
    print("\nMerging expression matrices...")
    adata = merge_expression(
        ld_expr, ld_genes, ld_cells,
        l1_expr, l1_genes, l1_cells,
        metadata=meta_indexed,
    )

    # 2d. Compute expression UMAP
    print("\nComputing expression UMAP...")
    expr_umap = compute_expression_umap(adata)

    # Rename to X_umap_expr for clarity (expression-only UMAP)
    adata.obsm["X_umap_expr"] = expr_umap.copy()

    # 2e. Compute ephys UMAP
    print("\nComputing ephys UMAP...")
    ephys_df = pd.read_csv(EPHYS_JOINED_CSV)
    compute_ephys_umap(adata, ephys=ephys_df)

    # 2f. Save
    print(f"\nSaving h5ad: {PATCHSEQ_H5AD}")
    adata.write(str(PATCHSEQ_H5AD))
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # 2g. Validate
    from patchseq_builder.validate import validate_h5ad
    report = validate_h5ad(PATCHSEQ_H5AD)
    report.print_summary()
    report.assert_ok()


# ============================================================================
# Stage 3: Reference integration + kNN labels
# ============================================================================

def stage_3_reference_integration(force: bool = False):
    """Integrate patch-seq with SEA-AD reference via Harmony, transfer labels via kNN."""

    outputs = [REFERENCE_COMBINED_H5AD, KNN_LABELS_CSV]
    if not force and all(_output_exists(p) for p in outputs):
        print(f"  Reference integration outputs exist, skipping. Use --force to regenerate.")
        return

    import scanpy as sc

    from patchseq_builder.reference.integration import integrate_patchseq_with_reference
    from patchseq_builder.reference.knn_transfer import knn_label_transfer, compare_knn_vs_scanvi

    if not PATCHSEQ_H5AD.exists():
        raise FileNotFoundError(
            f"Stage 2 output not found: {PATCHSEQ_H5AD}\n"
            f"Run stage 2 first (--from 2)."
        )

    # 3a. Harmony integration
    print("Running Harmony integration with SEA-AD reference...")
    combined = integrate_patchseq_with_reference(patchseq_h5ad_path=PATCHSEQ_H5AD)

    # 3b. kNN label transfer
    print("\nTransferring labels via kNN...")
    knn_labels = knn_label_transfer(combined)

    # 3c. Compare kNN vs scANVI
    print("\nComparing kNN vs scANVI labels...")
    confusion = compare_knn_vs_scanvi(combined, knn_labels)

    # 3d. Save outputs
    INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    combined.write(str(REFERENCE_COMBINED_H5AD))
    print(f"  Saved combined h5ad: {REFERENCE_COMBINED_H5AD}")

    knn_labels.to_csv(KNN_LABELS_CSV)
    print(f"  Saved kNN labels: {KNN_LABELS_CSV} ({len(knn_labels)} cells)")

    if not confusion.empty:
        confusion_path = TABLES_DIR / "knn_vs_scanvi_confusion.csv"
        confusion.to_csv(confusion_path)
        print(f"  Saved confusion matrix: {confusion_path}")


# ============================================================================
# Stage 4: Morphology download + render
# ============================================================================

def stage_4_morphology(force: bool = False):
    """Download SWC files from BIL and render to SVG."""

    import pandas as pd

    from patchseq_builder.morphology.download import download_swc, is_valid_swc
    from patchseq_builder.morphology.render import render_morphology_svg, load_layer_depths

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {COMBINED_CSV}\n"
            f"Run stage 1 first."
        )

    metadata = pd.read_csv(COMBINED_CSV)
    morpho_cells = metadata[metadata["has_morphology"] == True]
    specimen_ids = morpho_cells["specimen_id"].astype(int).tolist()
    sid_to_dataset = dict(zip(
        metadata["specimen_id"].astype(int), metadata["dataset"],
    ))
    print(f"  Cells with morphology flag: {len(specimen_ids)}")

    SWC_DIR.mkdir(parents=True, exist_ok=True)
    MORPHOLOGY_SVG_DIR.mkdir(parents=True, exist_ok=True)

    # 4a. Download SWC files (skip those already present unless --force)
    # Naming convention: {sid}_upright.swc (matches BIL filenames)
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    download_log = []

    for i, sid in enumerate(specimen_ids):
        swc_path = SWC_DIR / f"{sid}_upright.swc"

        if not force and is_valid_swc(swc_path):
            n_skipped += 1
            download_log.append({"specimen_id": sid, "status": "cached"})
            continue

        url, source = download_swc(sid, swc_path)
        if url:
            n_downloaded += 1
            download_log.append({"specimen_id": sid, "status": "downloaded", "source": source})
        else:
            n_failed += 1
            download_log.append({"specimen_id": sid, "status": "not_found"})

        # Progress every 50 cells
        if (i + 1) % 50 == 0 or (i + 1) == len(specimen_ids):
            print(f"  Download progress: {i + 1}/{len(specimen_ids)} "
                  f"(new: {n_downloaded}, cached: {n_skipped}, failed: {n_failed})")

    print(f"\n  Download summary: {n_downloaded} new, {n_skipped} cached, "
          f"{n_failed} not found on BIL")

    # 4b. Render SVGs
    # Naming convention: {sid}_morphology.svg
    print("\nRendering morphology SVGs...")
    layer_depths = load_layer_depths()

    n_rendered = 0
    n_render_skipped = 0
    svg_map_rows = []

    for sid in specimen_ids:
        swc_path = SWC_DIR / f"{sid}_upright.swc"
        svg_path = MORPHOLOGY_SVG_DIR / f"{sid}_morphology.svg"

        if not is_valid_swc(swc_path):
            continue

        if not force and svg_path.exists() and svg_path.stat().st_size > 100:
            n_render_skipped += 1
            svg_map_rows.append({
                "specimen_id": sid,
                "dataset": sid_to_dataset.get(sid, ""),
                "swc_path": str(swc_path),
                "svg_path": f"morphology_svgs/{sid}_morphology.svg",
                "status": "ok",
            })
            continue

        layer_info = layer_depths.get(sid, None)
        result = render_morphology_svg(swc_path, svg_path, layer_info=layer_info)
        if result:
            n_rendered += 1
            svg_map_rows.append({
                "specimen_id": sid,
                "dataset": sid_to_dataset.get(sid, ""),
                "swc_path": str(swc_path),
                "svg_path": f"morphology_svgs/{sid}_morphology.svg",
                "status": "ok",
            })

    print(f"  Render summary: {n_rendered} new, {n_render_skipped} cached")

    # 4c. Save morphology SVG map (schema: specimen_id, dataset, swc_path, svg_path, status)
    svg_map_df = pd.DataFrame(svg_map_rows)
    svg_map_df.to_csv(MORPHOLOGY_SVG_MAP_CSV, index=False)
    print(f"  Saved morphology SVG map: {MORPHOLOGY_SVG_MAP_CSV} ({len(svg_map_df)} entries)")


# ============================================================================
# Stage 5: Traces
# ============================================================================

def stage_5_traces(force: bool = False):
    """Generate missing trace SVGs, build SVG mapping, and create symlinks."""

    # 5a. Generate missing trace SVGs from DANDI using pyAPisolation
    print("Checking for missing trace SVGs...")
    try:
        from patchseq_builder.traces.generate import generate_missing_traces
        gen_report = generate_missing_traces(force=force)
        if gen_report.get("n_new", 0) > 0:
            print(f"  Generated {gen_report['n_new']} new trace SVGs")
            force = True  # force map rebuild since new SVGs exist
    except ImportError:
        print("  WARNING: pyAPisolation not installed, skipping trace generation.")
        print("  Install with: pip install 'git+https://github.com/smestern/pyAPisolation.git'")
    except Exception as e:
        print(f"  WARNING: Trace generation failed: {e}")
        print("  Continuing with existing SVGs...")

    if not force and _output_exists(TRACE_SVG_MAP_CSV):
        print(f"  Trace SVG map exists: {TRACE_SVG_MAP_CSV}, skipping. Use --force to regenerate.")
        # Still ensure symlinks even when skipping the map rebuild
        from patchseq_builder.traces.symlinks import ensure_trace_symlinks
        print("\nVerifying trace symlinks...")
        ensure_trace_symlinks()
        return

    from patchseq_builder.traces.svg_map import build_trace_svg_map, save_trace_svg_map
    from patchseq_builder.traces.symlinks import ensure_trace_symlinks

    # 5b. Build the specimen -> trace SVG mapping
    print("Building trace SVG mapping...")
    trace_map = build_trace_svg_map()
    save_trace_svg_map(trace_map)

    # 5c. Create symlinks for HTTP serving
    print("\nCreating trace symlinks for viewer...")
    ensure_trace_symlinks()

    # 5d. Validate
    from patchseq_builder.validate import validate_svg_map
    from patchseq_builder.config import TRACE_SYMLINK_DIR
    report = validate_svg_map(TRACE_SVG_MAP_CSV, trace_dir=TRACE_SYMLINK_DIR)
    report.print_summary()
    # Don't assert_ok here -- missing SVGs are common and non-fatal


# ============================================================================
# Stage 6: Build interactive viewer
# ============================================================================

def stage_6_viewer(force: bool = False):
    """Build the interactive HTML viewer."""

    if not force and _output_exists(VIEWER_HTML):
        print(f"  Viewer exists: {VIEWER_HTML}, skipping. Use --force to regenerate.")
        return

    from patchseq_builder.viewer.build_html import build_viewer_html

    print("Building interactive viewer...")
    build_viewer_html()

    # Validate
    from patchseq_builder.validate import validate_viewer
    report = validate_viewer(VIEWER_HTML)
    report.print_summary()
    report.assert_ok()


# ============================================================================
# Stage 7: Comprehensive accounting
# ============================================================================

def stage_7_accounting(force: bool = False):
    """Generate comprehensive metadata CSV and data source accounting."""

    if not force and _output_exists(COMPREHENSIVE_METADATA_CSV) and _output_exists(DATA_ACCOUNTING_TXT):
        print(f"  Accounting outputs exist, skipping. Use --force to regenerate.")
        return

    import numpy as np
    import pandas as pd

    # Load all available stage outputs
    print("Loading stage outputs for accounting...")

    # -- Base metadata from stage 1 --
    if COMBINED_CSV.exists():
        meta = pd.read_csv(COMBINED_CSV)
        print(f"  Combined metadata: {len(meta)} cells")
    else:
        print(f"  WARNING: Combined metadata not found at {COMBINED_CSV}")
        meta = pd.DataFrame()

    if meta.empty:
        print("  No metadata to account. Skipping.")
        return

    # -- kNN labels from stage 3 --
    if KNN_LABELS_CSV.exists():
        knn = pd.read_csv(KNN_LABELS_CSV)
        # kNN CSV uses h5ad index (patched_cell_container) as "specimen_id",
        # not numeric specimen_ids. We need to map via the h5ad obs table.
        knn_cell_name_col = "specimen_id"  # this is actually the h5ad index (cell name)

        # Build cell_name -> numeric specimen_id mapping from h5ad
        cell_to_sid = {}
        if PATCHSEQ_H5AD.exists():
            import scanpy as sc
            ps = sc.read_h5ad(str(PATCHSEQ_H5AD), backed="r")
            if "specimen_id" in ps.obs.columns:
                cell_to_sid = dict(zip(
                    ps.obs.index.astype(str),
                    ps.obs["specimen_id"].astype(str),
                ))
            ps.file.close()

        if cell_to_sid:
            # Map kNN cell names to numeric specimen_ids
            knn["numeric_sid"] = knn[knn_cell_name_col].astype(str).map(cell_to_sid)
            for col in ["knn_subclass", "knn_subclass_conf", "knn_supertype", "knn_supertype_conf"]:
                if col in knn.columns:
                    knn_map = dict(zip(knn["numeric_sid"].dropna(), knn.loc[knn["numeric_sid"].notna(), col]))
                    meta[col] = meta["specimen_id"].astype(str).map(knn_map)
            n_mapped = meta["knn_subclass"].notna().sum() if "knn_subclass" in meta.columns else 0
            print(f"  kNN labels merged: {n_mapped} cells (via h5ad index mapping)")
        else:
            print(f"  WARNING: Could not map kNN labels — h5ad not found or missing specimen_id")

    # -- Morphology SVG map from stage 4 --
    if MORPHOLOGY_SVG_MAP_CSV.exists():
        morph_map = pd.read_csv(MORPHOLOGY_SVG_MAP_CSV)
        morph_sids = set(morph_map["specimen_id"].astype(int))
        meta["has_morphology_svg"] = meta["specimen_id"].astype(int).isin(morph_sids)
        n_morph = meta["has_morphology_svg"].sum()
        print(f"  Morphology SVGs available: {n_morph}")

    # -- Trace SVG map from stage 5 --
    if TRACE_SVG_MAP_CSV.exists():
        trace_map = pd.read_csv(TRACE_SVG_MAP_CSV)
        has_trace = trace_map[trace_map["trace_svg"].notna() & (trace_map["trace_svg"] != "")]
        trace_sids = set(has_trace["specimen_id"].astype(int))
        meta["has_trace_svg"] = meta["specimen_id"].astype(int).isin(trace_sids)
        n_trace = meta["has_trace_svg"].sum()
        print(f"  Trace SVGs available: {n_trace}")

    # Save comprehensive metadata
    meta.to_csv(COMPREHENSIVE_METADATA_CSV, index=False)
    print(f"  Saved: {COMPREHENSIVE_METADATA_CSV} ({len(meta)} cells x {meta.shape[1]} cols)")

    # -- Build accounting text report --
    lines = []
    lines.append("=" * 70)
    lines.append("PATCH-SEQ DATA SOURCE ACCOUNTING")
    lines.append(f"Generated by build_patchseq_viewer.py")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Cell counts by dataset
    lines.append("CELL COUNTS BY DATASET:")
    for ds in sorted(meta["dataset"].unique()):
        n = (meta["dataset"] == ds).sum()
        lines.append(f"  {ds:20s}: {n:5d}")
    lines.append(f"  {'TOTAL':20s}: {len(meta):5d}")
    lines.append("")

    # Data modality coverage
    lines.append("DATA MODALITY COVERAGE:")
    for col, label in [
        ("has_ephys", "Electrophysiology"),
        ("has_morphology", "Morphology (flagged)"),
    ]:
        if col in meta.columns:
            n = meta[col].sum()
            pct = 100 * n / len(meta)
            lines.append(f"  {label:30s}: {n:5d} / {len(meta)} ({pct:.1f}%)")

    if "subclass_scANVI" in meta.columns:
        n = meta["subclass_scANVI"].notna().sum()
        pct = 100 * n / len(meta)
        lines.append(f"  {'scANVI labels':30s}: {n:5d} / {len(meta)} ({pct:.1f}%)")

    if "knn_subclass" in meta.columns:
        n = meta["knn_subclass"].notna().sum()
        pct = 100 * n / len(meta)
        lines.append(f"  {'kNN labels':30s}: {n:5d} / {len(meta)} ({pct:.1f}%)")

    if "has_morphology_svg" in meta.columns:
        n = meta["has_morphology_svg"].sum()
        pct = 100 * n / len(meta)
        lines.append(f"  {'Morphology SVGs':30s}: {n:5d} / {len(meta)} ({pct:.1f}%)")

    if "has_trace_svg" in meta.columns:
        n = meta["has_trace_svg"].sum()
        pct = 100 * n / len(meta)
        lines.append(f"  {'Trace SVGs':30s}: {n:5d} / {len(meta)} ({pct:.1f}%)")
    lines.append("")

    # Output file inventory
    lines.append("OUTPUT FILE INVENTORY:")
    output_files = [
        ("Metadata CSV", METADATA_JOINED_CSV),
        ("Ephys CSV", EPHYS_JOINED_CSV),
        ("Combined CSV", COMBINED_CSV),
        ("h5ad", PATCHSEQ_H5AD),
        ("Reference h5ad", REFERENCE_COMBINED_H5AD),
        ("kNN labels", KNN_LABELS_CSV),
        ("Morphology SVG map", MORPHOLOGY_SVG_MAP_CSV),
        ("Trace SVG map", TRACE_SVG_MAP_CSV),
        ("Viewer HTML", VIEWER_HTML),
        ("Comprehensive metadata", COMPREHENSIVE_METADATA_CSV),
    ]
    for label, path in output_files:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            lines.append(f"  {label:25s}: {path.name:45s} ({size_mb:.1f} MB)")
        else:
            lines.append(f"  {label:25s}: MISSING")
    lines.append("")

    # Missing ephys accounting
    if "has_ephys" in meta.columns:
        missing_ephys = meta[meta["has_ephys"] == False][["specimen_id", "dataset", "donor"]].copy()
        if not missing_ephys.empty:
            lines.append(f"CELLS WITHOUT EPHYS: {len(missing_ephys)}")
            for ds in sorted(missing_ephys["dataset"].unique()):
                n = (missing_ephys["dataset"] == ds).sum()
                lines.append(f"  {ds}: {n}")
            missing_ephys.to_csv(MISSING_EPHYS_CSV, index=False)
            lines.append(f"  Saved to: {MISSING_EPHYS_CSV}")
        lines.append("")

    accounting_text = "\n".join(lines)
    DATA_ACCOUNTING_TXT.write_text(accounting_text)
    print(f"  Saved: {DATA_ACCOUNTING_TXT}")
    print()
    print(accounting_text)


# ============================================================================
# Main orchestrator
# ============================================================================

STAGES = {
    1: ("Metadata harmonization", stage_1_metadata),
    2: ("Build h5ad with expression + UMAPs", stage_2_build_h5ad),
    3: ("Reference integration + kNN labels", stage_3_reference_integration),
    4: ("Morphology download + render", stage_4_morphology),
    5: ("Trace SVG mapping + symlinks", stage_5_traces),
    6: ("Build interactive viewer", stage_6_viewer),
    7: ("Comprehensive accounting", stage_7_accounting),
}


def main():
    parser = argparse.ArgumentParser(
        description="Build the complete patch-seq data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from", dest="start_from", type=int, default=1, metavar="N",
        help=f"Start from stage N (1-{N_STAGES}). Default: 1",
    )
    parser.add_argument(
        "--only", dest="only_stage", type=int, default=None, metavar="N",
        help=f"Run only stage N (1-{N_STAGES}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of cached outputs.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.start_from < 1 or args.start_from > N_STAGES:
        parser.error(f"--from must be between 1 and {N_STAGES}")
    if args.only_stage is not None and (args.only_stage < 1 or args.only_stage > N_STAGES):
        parser.error(f"--only must be between 1 and {N_STAGES}")

    print("=" * 70)
    print("  PATCH-SEQ VIEWER BUILD PIPELINE")
    print("=" * 70)
    print(f"  Force recompute: {args.force}")
    print(f"  Start from:      stage {args.start_from}")
    if args.only_stage:
        print(f"  Run only:        stage {args.only_stage}")
    print(f"  Project root:    {PROJECT_ROOT}")
    print()

    _ensure_dirs()

    total_t0 = time.time()
    timings = []

    for stage_num in range(1, N_STAGES + 1):
        title, func = STAGES[stage_num]

        if args.only_stage is not None and stage_num != args.only_stage:
            continue
        if stage_num < args.start_from:
            print(f"  Skipping stage {stage_num}: {title}")
            continue

        _header(stage_num, title)
        t0 = time.time()

        try:
            func(force=args.force)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR in stage {stage_num} after {elapsed:.1f}s: {e}")
            logger.exception("Stage %d failed", stage_num)
            sys.exit(1)

        elapsed = time.time() - t0
        timings.append((stage_num, title, elapsed))
        print(f"\n  Stage {stage_num} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nTiming summary:")
    for stage_num, title, elapsed in timings:
        print(f"  Stage {stage_num}: {elapsed:8.1f}s  {title}")
    print(f"  {'---':40s}")
    print(f"  Total: {total_elapsed:8.1f}s")

    print(f"\nOutputs:")
    print(f"  Metadata:    {COMBINED_CSV}")
    print(f"  h5ad:        {PATCHSEQ_H5AD}")
    print(f"  Viewer:      {VIEWER_HTML}")
    print(f"  Accounting:  {DATA_ACCOUNTING_TXT}")


if __name__ == "__main__":
    main()
