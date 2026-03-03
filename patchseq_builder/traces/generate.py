"""
generate.py -- Generate trace SVGs from DANDI NWB files using pyAPisolation.

Downloads NWB files from DANDI S3 for specimens that have DANDI mappings
but no existing trace SVGs, then uses pyAPisolation's plot_data() to
generate voltage trace and FI curve SVGs.

The generated SVGs are placed in intraDANDI_explorer-master/data/traces/
where svg_map.py auto-discovers them.

Usage:
    python -m patchseq_builder.traces.generate                     # all missing
    python -m patchseq_builder.traces.generate --specimen 643599478  # one cell
    python -m patchseq_builder.traces.generate --dry-run             # list only
    python -m patchseq_builder.traces.generate --force               # regenerate
    python -m patchseq_builder.traces.generate --keep-cache          # keep NWBs
"""

import argparse
import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from patchseq_builder.config import (
    DATA_DIR,
    INTRADANDI_TRACES_DIR,
    NWB_CACHE_DIR,
    TRACE_GENERATION_LOG_CSV,
    TRACE_SVG_MAP_CSV,
    TRACE_TARGET_AMPS,
)

logger = logging.getLogger(__name__)

# DANDI asset map for looking up asset UUIDs
DANDI_ASSET_MAP_636 = DATA_DIR / "dandi_636_asset_map.csv"
SPECIMEN_TO_DANDI_MAP = DATA_DIR / "specimen_to_dandi_map.csv"

# DANDI API download URL template (redirects to signed S3 URL).
# The old direct S3 blob URLs become stale when dandisets are re-versioned,
# so we use the API endpoint which always resolves correctly.
DANDI_API_DOWNLOAD_URL = (
    "https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/"
    "assets/{identifier}/download/"
)


# ---------------------------------------------------------------------------
# Identify missing cells
# ---------------------------------------------------------------------------

def identify_missing_cells():
    """Find specimens with DANDI NWB mappings but no existing trace SVGs.

    Cross-references the SVG map (which has empty trace_svg for missing cells)
    with the DANDI asset map (which has S3 download URLs).

    Returns
    -------
    pd.DataFrame
        Rows for missing cells with columns: specimen_id, dandiset, dandi_path,
        subject, nwb_basename, download_url
    """
    # Load SVG map
    if not TRACE_SVG_MAP_CSV.exists():
        logger.error("SVG map not found: %s", TRACE_SVG_MAP_CSV)
        return pd.DataFrame()

    svg_map = pd.read_csv(TRACE_SVG_MAP_CSV)
    # Missing = has dandi_path but empty trace_svg
    missing = svg_map[
        svg_map["dandi_path"].notna()
        & (svg_map["dandi_path"] != "")
        & (svg_map["trace_svg"].isna() | (svg_map["trace_svg"] == ""))
    ].copy()

    if missing.empty:
        logger.info("No missing trace SVGs found — all cells have SVGs")
        return pd.DataFrame()

    logger.info("Found %d cells with DANDI mappings but no trace SVGs", len(missing))

    # Load asset maps to get asset UUIDs → build DANDI API download URLs
    # Maps (dandiset, dandi_path) → download URL
    asset_links = {}
    asset_map_configs = [
        ("000636", DANDI_ASSET_MAP_636),
    ]
    for dandiset_id, asset_csv in asset_map_configs:
        if asset_csv.exists():
            df = pd.read_csv(asset_csv)
            for _, row in df.iterrows():
                path = str(row.get("path", ""))
                identifier = str(row.get("identifier", ""))
                if path and identifier and identifier != "nan":
                    url = DANDI_API_DOWNLOAD_URL.format(
                        dandiset=dandiset_id,
                        identifier=identifier,
                    )
                    asset_links[path] = url

    # Build output with download URLs
    rows = []
    for _, row in missing.iterrows():
        dandi_path = str(row["dandi_path"])
        download_url = asset_links.get(dandi_path, "")

        if not download_url:
            logger.warning("No download URL for specimen %s (%s)",
                          row["specimen_id"], dandi_path)
            continue

        # Extract subject and NWB basename from dandi_path
        # e.g. "sub-636948822/sub-636948822_ses-636982248_icephys.nwb"
        parts = dandi_path.split("/")
        subject = parts[0] if len(parts) > 1 else ""
        nwb_basename = parts[-1]
        dandiset = str(int(row.get("dandiset", 636))).zfill(6)

        rows.append({
            "specimen_id": int(row["specimen_id"]),
            "dandiset": dandiset,
            "dandi_path": dandi_path,
            "subject": subject,
            "nwb_basename": nwb_basename,
            "download_url": download_url,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Download NWB
# ---------------------------------------------------------------------------

def download_nwb(download_url, out_path, timeout=120):
    """Download an NWB file from the DANDI API.

    The DANDI API download endpoint redirects to a signed S3 URL.

    Parameters
    ----------
    download_url : str
        The DANDI API download URL (redirects to signed S3 URL).
    out_path : str or Path
        Local path to save the NWB file.
    timeout : int
        Download timeout in seconds.

    Returns
    -------
    bool
        True if download succeeded and file is valid.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already cached (must be at least 1 MB to be a valid NWB)
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        logger.info("  Using cached NWB: %s (%.1f MB)",
                     out_path.name, out_path.stat().st_size / 1024 / 1024)
        return True

    try:
        logger.info("  Downloading: %s", out_path.name)
        # Use urlopen with timeout instead of urlretrieve (which has no timeout)
        req = urllib.request.Request(download_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(str(out_path), "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)

        # Basic validation
        if out_path.stat().st_size < 1000:
            logger.warning("  Downloaded file too small: %d bytes", out_path.stat().st_size)
            out_path.unlink()
            return False

        logger.info("  Downloaded: %.1f MB", out_path.stat().st_size / 1024 / 1024)
        return True

    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        logger.error("  Download failed: %s", e)
        if out_path.exists():
            out_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Generate SVGs using pyAPisolation
# ---------------------------------------------------------------------------

def generate_trace_svgs(nwb_path):
    """Generate trace and FI SVGs from a local NWB file using pyAPisolation.

    Parameters
    ----------
    nwb_path : str or Path
        Path to the local NWB file.

    Returns
    -------
    dict
        Keys: trace_svg (Path or None), fi_svg (Path or None),
        status (str), error (str or None)
    """
    from pyAPisolation.database.build_database import plot_data

    nwb_path = Path(nwb_path)
    trace_svg = Path(str(nwb_path) + ".svg")
    fi_svg = Path(str(nwb_path) + "_FI.svg")

    try:
        # Use stim_override='' to accept all protocols.
        # AIBS NWBs use code-style names (e.g. C1LSFINEST150112_DA_0)
        # that don't match the default 'long' / '1000' filter.
        # pyAPisolation's match_protocol() still checks waveform shape.
        result = plot_data(
            specimen_id=0,
            file_list=[str(nwb_path)],
            target_amps=TRACE_TARGET_AMPS,
            overwrite=True,
            save=True,
            stim_override='',
        )

        # Check if SVGs were created
        has_trace = trace_svg.exists() and trace_svg.stat().st_size > 100
        has_fi = fi_svg.exists() and fi_svg.stat().st_size > 100

        if has_trace:
            return {
                "trace_svg": trace_svg,
                "fi_svg": fi_svg if has_fi else None,
                "status": "ok",
                "error": None,
            }
        else:
            # plot_data ran but didn't produce SVGs (no Long Square sweeps)
            return {
                "trace_svg": None,
                "fi_svg": None,
                "status": "no_sweeps_found",
                "error": "plot_data produced no SVG output",
            }

    except Exception as e:
        logger.error("  pyAPisolation error: %s", e)
        return {
            "trace_svg": None,
            "fi_svg": None,
            "status": "plot_error",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Organize SVGs into target directory
# ---------------------------------------------------------------------------

def organize_svgs(trace_svg, fi_svg, dandiset, subject, target_base=None):
    """Move generated SVGs to the intraDANDI explorer trace directory.

    Parameters
    ----------
    trace_svg : Path
        Path to the generated trace SVG.
    fi_svg : Path or None
        Path to the generated FI curve SVG.
    dandiset : str
        Dandiset ID (e.g. "000636").
    subject : str
        Subject directory name (e.g. "sub-636948822").
    target_base : Path, optional
        Base trace directory. Defaults to INTRADANDI_TRACES_DIR.

    Returns
    -------
    tuple
        (trace_dest, fi_dest) paths of the moved files.
    """
    if target_base is None:
        target_base = INTRADANDI_TRACES_DIR

    target_dir = target_base / dandiset / subject
    target_dir.mkdir(parents=True, exist_ok=True)

    trace_dest = target_dir / trace_svg.name
    shutil.copy2(str(trace_svg), str(trace_dest))
    logger.info("  Moved trace SVG → %s", trace_dest)

    fi_dest = None
    if fi_svg and fi_svg.exists():
        fi_dest = target_dir / fi_svg.name
        shutil.copy2(str(fi_svg), str(fi_dest))
        logger.info("  Moved FI SVG → %s", fi_dest)

    return trace_dest, fi_dest


# ---------------------------------------------------------------------------
# Process a single cell
# ---------------------------------------------------------------------------

def process_single_cell(row, cache_dir=None):
    """Full pipeline for one cell: download → generate → organize.

    Parameters
    ----------
    row : dict or pd.Series
        Must have: specimen_id, dandiset, dandi_path, subject, nwb_basename, download_url
    cache_dir : Path, optional
        NWB cache directory. Defaults to NWB_CACHE_DIR.

    Returns
    -------
    dict
        Log entry with: specimen_id, dandiset, dandi_path, status, error_message,
        processing_time_s
    """
    if cache_dir is None:
        cache_dir = NWB_CACHE_DIR

    t0 = time.time()
    sid = row["specimen_id"]
    dandiset = row["dandiset"]
    subject = row["subject"]
    dandi_path = row["dandi_path"]
    download_url = row["download_url"]

    logger.info("Processing specimen %s (%s)", sid, dandi_path)

    # 0. Check if SVG already exists in target directory (from a previous run)
    target_svg = INTRADANDI_TRACES_DIR / dandiset / subject / (row["nwb_basename"] + ".svg")
    if target_svg.exists() and target_svg.stat().st_size > 100:
        logger.info("  SVG already exists in target: %s", target_svg.name)
        return {
            "specimen_id": sid, "dandiset": dandiset, "dandi_path": dandi_path,
            "status": "already_exists", "error_message": "",
            "processing_time_s": time.time() - t0,
        }

    # 1. Download
    nwb_local = cache_dir / dandiset / subject / row["nwb_basename"]
    if not download_nwb(download_url, nwb_local):
        return {
            "specimen_id": sid, "dandiset": dandiset, "dandi_path": dandi_path,
            "status": "download_failed", "error_message": "S3 download failed",
            "processing_time_s": time.time() - t0,
        }

    # 2. Generate SVGs
    result = generate_trace_svgs(nwb_local)
    if result["status"] != "ok":
        return {
            "specimen_id": sid, "dandiset": dandiset, "dandi_path": dandi_path,
            "status": result["status"], "error_message": result["error"],
            "processing_time_s": time.time() - t0,
        }

    # 3. Organize (move to intraDANDI traces directory)
    organize_svgs(result["trace_svg"], result["fi_svg"], dandiset, subject)

    return {
        "specimen_id": sid, "dandiset": dandiset, "dandi_path": dandi_path,
        "status": "ok", "error_message": "",
        "processing_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_missing_traces(force=False, specimen_id=None, dry_run=False,
                            keep_cache=False):
    """Generate trace SVGs for all cells with DANDI NWBs but no existing SVGs.

    Parameters
    ----------
    force : bool
        If True, regenerate SVGs even if they already exist.
    specimen_id : int, optional
        If provided, only process this one specimen.
    dry_run : bool
        If True, list missing cells without processing.
    keep_cache : bool
        If True, don't delete downloaded NWB files after processing.

    Returns
    -------
    dict
        Summary with keys: n_missing, n_processed, n_new, n_failed, log_path
    """
    print("\n" + "=" * 60)
    print("Trace SVG Generation from DANDI NWB Files")
    print("=" * 60)

    # Identify missing cells
    missing = identify_missing_cells()
    if missing.empty:
        print("No missing trace SVGs found. All cells have SVGs.")
        return {"n_missing": 0, "n_processed": 0, "n_new": 0, "n_failed": 0}

    # Filter to specific specimen if requested
    if specimen_id is not None:
        missing = missing[missing["specimen_id"] == int(specimen_id)]
        if missing.empty:
            print(f"Specimen {specimen_id} not in missing cells list.")
            return {"n_missing": 0, "n_processed": 0, "n_new": 0, "n_failed": 0}

    print(f"\nFound {len(missing)} cells with DANDI NWBs but no trace SVGs")
    n_subjects = missing["subject"].nunique()
    print(f"  Across {n_subjects} subjects in {missing['dandiset'].nunique()} dandiset(s)")

    if dry_run:
        print("\n[DRY RUN] Would process these cells:")
        for _, row in missing.iterrows():
            print(f"  {row['specimen_id']} → {row['dandi_path']}")
        return {"n_missing": len(missing), "n_processed": 0, "n_new": 0, "n_failed": 0}

    # Process each cell
    log_entries = []
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for i, (_, row) in enumerate(missing.iterrows()):
        print(f"\n[{i+1}/{len(missing)}] ", end="")
        entry = process_single_cell(row)
        log_entries.append(entry)

        if entry["status"] in ("ok", "already_exists"):
            n_ok += 1
            status_char = "✓" if entry["status"] == "ok" else "↩"
            print(f"  {status_char} specimen {entry['specimen_id']} ({entry['processing_time_s']:.1f}s)")
        else:
            n_fail += 1
            print(f"  ✗ specimen {entry['specimen_id']}: {entry['status']} "
                  f"({entry.get('error_message', '')})")

    # Save log
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv(TRACE_GENERATION_LOG_CSV, index=False)
    print(f"\nLog saved: {TRACE_GENERATION_LOG_CSV}")

    # Clean up cache
    if not keep_cache and NWB_CACHE_DIR.exists():
        cache_size = sum(f.stat().st_size for f in NWB_CACHE_DIR.rglob("*") if f.is_file())
        print(f"\nCleaning up NWB cache ({cache_size / 1024 / 1024:.0f} MB)...")
        shutil.rmtree(NWB_CACHE_DIR)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Trace generation complete:")
    print(f"  Processed: {len(log_entries)}")
    print(f"  Success:   {n_ok}")
    print(f"  Failed:    {n_fail}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    return {
        "n_missing": len(missing),
        "n_processed": len(log_entries),
        "n_new": n_ok,
        "n_failed": n_fail,
        "log_path": str(TRACE_GENERATION_LOG_CSV),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate trace SVGs from DANDI NWB files using pyAPisolation"
    )
    parser.add_argument("--specimen", type=int, default=None,
                        help="Process only this specimen ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing cells without processing")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate SVGs even if they exist")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep downloaded NWB files after processing")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    generate_missing_traces(
        force=args.force,
        specimen_id=args.specimen,
        dry_run=args.dry_run,
        keep_cache=args.keep_cache,
    )


if __name__ == "__main__":
    main()
