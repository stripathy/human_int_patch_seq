"""
svg_map.py -- Build the specimen -> trace SVG mapping.

Auto-discovers trace SVGs in the intraDANDI_explorer trace directories and
matches them to specimen_ids using the specimen_to_dandi_map.csv (which maps
each specimen_id to its NWB file path within a DANDI dandiset).

The resulting DataFrame has the same schema as the existing
data/patchseq/specimen_to_svg_map.csv:
    specimen_id, ses_id, dandiset, dandi_path, trace_svg, fi_svg, file_link

For HTTP serving in the interactive viewer, the trace SVG paths are stored
as relative paths under intraDANDI_explorer-master/data/traces/... which
get rewritten to traces/... via symlinks at serve time.
"""

import re
from pathlib import Path

import pandas as pd

from patchseq_builder.config import (
    DATA_DIR,
    INTRADANDI_TRACES_DIR,
    DANDI_DANDISETS,
    TRACE_SVG_MAP_CSV,
)


# Path to the specimen -> DANDI NWB mapping (produced by metadata pipeline)
SPECIMEN_TO_DANDI_MAP_CSV = DATA_DIR / "specimen_to_dandi_map.csv"

# Optional DANDI asset maps (for constructing download links)
DANDI_ASSET_MAPS = {
    "000636": DATA_DIR / "dandi_636_asset_map.csv",
    "000630": DATA_DIR / "dandi_630_asset_map.csv",
}

# DANDI S3 blob URL template
DANDI_BLOB_URL = "https://dandiarchive.s3.amazonaws.com/blobs/{p1}/{p2}/{identifier}"


def _scan_trace_svgs() -> dict:
    """Scan intraDANDI trace directories for available SVGs.

    Returns
    -------
    dict
        Mapping (dandiset_str, nwb_basename) -> {trace_svg, fi_svg}
        where nwb_basename is like 'sub-X_ses-Y_icephys.nwb' and paths
        are relative to the project root.
    """
    svg_index = {}

    if not INTRADANDI_TRACES_DIR.exists():
        print(f"  WARNING: Trace directory not found: {INTRADANDI_TRACES_DIR}")
        return svg_index

    # Iterate over dandiset dirs that we care about
    for dandiset_id in DANDI_DANDISETS:
        dandiset_str = f"000{dandiset_id}" if len(str(dandiset_id)) == 3 else str(dandiset_id)
        dandiset_dir = INTRADANDI_TRACES_DIR / dandiset_str
        if not dandiset_dir.exists():
            continue

        # Scan all SVGs under this dandiset
        for svg_path in dandiset_dir.rglob("*.svg"):
            fname = svg_path.name
            # Skip FI curve SVGs (handled as companion to trace SVG)
            if fname.endswith("_FI.svg"):
                continue

            # The SVG corresponds to an NWB file:
            # sub-X_ses-Y_icephys.nwb.svg -> nwb_basename = sub-X_ses-Y_icephys.nwb
            if fname.endswith(".nwb.svg"):
                nwb_basename = fname[:-4]  # strip .svg
            else:
                continue

            # Build relative path from project root
            rel_path = svg_path.relative_to(INTRADANDI_TRACES_DIR.parent.parent.parent)
            rel_trace = str(rel_path)

            # Check for companion FI SVG
            fi_path = svg_path.parent / (fname.replace(".nwb.svg", ".nwb_FI.svg"))
            rel_fi = str(fi_path.relative_to(
                INTRADANDI_TRACES_DIR.parent.parent.parent
            )) if fi_path.exists() else ""

            # Key: (dandiset, nwb_basename) -- nwb_basename includes sub-X prefix
            svg_index[(dandiset_str, nwb_basename)] = {
                "trace_svg": rel_trace,
                "fi_svg": rel_fi,
            }

    return svg_index


def _load_asset_links() -> dict:
    """Load DANDI asset UUIDs to build download links.

    Returns
    -------
    dict
        nwb_path -> download URL
    """
    links = {}
    for dandiset_str, csv_path in DANDI_ASSET_MAPS.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            nwb_path = str(row.get("path", ""))
            identifier = str(row.get("identifier", ""))
            if nwb_path and identifier and identifier != "nan":
                # Build S3 URL: blobs/{first3}/{next3}/{full_uuid}
                url = DANDI_BLOB_URL.format(
                    p1=identifier[:3], p2=identifier[3:6],
                    identifier=identifier,
                )
                links[nwb_path] = url
    return links


def _extract_ses_id(dandi_path: str):
    """Extract session ID from a DANDI NWB path.

    Examples:
        'sub-636948822/sub-636948822_ses-636982248_icephys.nwb' -> 636982248.0
        'sub-H18-28-011/sub-H18-28-011_icephys.nwb' -> NaN (no session)
    """
    m = re.search(r"_ses-(\d+)_", str(dandi_path))
    if m:
        return float(m.group(1))
    return float("nan")


def build_trace_svg_map(metadata_csv=None) -> pd.DataFrame:
    """Build specimen -> trace SVG mapping by scanning DANDI trace directories.

    Loads the specimen_to_dandi_map.csv to know which NWB file each specimen
    maps to, then scans intraDANDI_explorer-master/data/traces/{dandiset}/{subject}/
    for matching SVG files.

    Parameters
    ----------
    metadata_csv : str or Path, optional
        Path to the specimen_to_dandi_map.csv. If None, uses the default
        location in DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Columns: specimen_id, ses_id, dandiset, dandi_path, trace_svg,
        fi_svg, file_link
    """
    if metadata_csv is None:
        metadata_csv = SPECIMEN_TO_DANDI_MAP_CSV

    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        print(f"  ERROR: specimen_to_dandi_map.csv not found: {metadata_csv}")
        return pd.DataFrame(columns=[
            "specimen_id", "ses_id", "dandiset", "dandi_path",
            "trace_svg", "fi_svg", "file_link",
        ])

    # Load specimen -> DANDI NWB mapping
    dandi_map = pd.read_csv(metadata_csv)
    print(f"  Loaded {len(dandi_map)} specimen->DANDI mappings")

    # Scan available trace SVGs
    svg_index = _scan_trace_svgs()
    print(f"  Found {len(svg_index)} trace SVGs across DANDI directories")

    # Load asset download links
    asset_links = _load_asset_links()

    # Build the mapping
    rows = []
    n_matched = 0

    for _, row in dandi_map.iterrows():
        specimen_id = int(row["specimen_id"])
        dandi_path = str(row["dandi_path"])
        dandiset = str(row["dandiset"])

        # Pad dandiset to 6 chars (e.g. '636' -> '000636')
        dandiset_padded = dandiset.zfill(6)

        # Extract NWB basename from dandi_path
        nwb_basename = Path(dandi_path).name  # e.g. sub-X_ses-Y_icephys.nwb

        # Look up in SVG index
        svg_info = svg_index.get((dandiset_padded, nwb_basename), {})
        trace_svg = svg_info.get("trace_svg", "")
        fi_svg = svg_info.get("fi_svg", "")

        if trace_svg:
            n_matched += 1

        # Extract session ID
        ses_id = _extract_ses_id(dandi_path)

        # Look up download link
        file_link = asset_links.get(dandi_path, "")

        rows.append({
            "specimen_id": specimen_id,
            "ses_id": ses_id,
            "dandiset": int(dandiset),
            "dandi_path": dandi_path,
            "trace_svg": trace_svg,
            "fi_svg": fi_svg,
            "file_link": file_link,
        })

    df = pd.DataFrame(rows)
    print(f"  Matched {n_matched}/{len(df)} specimens to trace SVGs")

    return df


def save_trace_svg_map(df=None, out_path=None):
    """Build and save the trace SVG mapping CSV.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built mapping. If None, calls build_trace_svg_map().
    out_path : str or Path, optional
        Output path. Defaults to config.TRACE_SVG_MAP_CSV.
    """
    if df is None:
        df = build_trace_svg_map()
    if out_path is None:
        out_path = TRACE_SVG_MAP_CSV

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved trace SVG map: {out_path} ({len(df)} rows)")
    return out_path
