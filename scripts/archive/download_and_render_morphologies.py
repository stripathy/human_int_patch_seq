#!/usr/bin/env python3
"""Download SWC files from BIL and render morphology SVGs.

Downloads upright/transformed SWC reconstructions from the Brain Image Library
for all patch-seq cells that have morphology data, then renders them to SVG.

BIL submissions containing our data (6 total):
  1. group/20230426  — LeeDalley batch upload: {sid}_upright.swc
  2. d833ba8bd931f23f — L1 cells: {sid}_upright.swc
  3. efb9b12ba2fab63d — LeeDalley cells: {sid}_upright.swc
  4. 241a10cde842c99b — L1 cells: {sid}_transformed.swc
  5. 49e6114ba67eda01 — L1 cells: {sid}_upright.swc
  6. 85f4b93699151f1c — L1 cells: {sid}_upright.swc
  7. 69fe931fee2b2215 — L1 cells: non-standard H*_m.swc filenames (need dir listing)
"""

import html.parser
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SWC_DIR = PROJECT_ROOT / "data" / "morphology" / "swc"
SVG_DIR = PROJECT_ROOT / "results" / "figures" / "morphology_svgs"
METADATA_CSV = PROJECT_ROOT / "data" / "patchseq" / "patchseq_metadata_joined.csv"
OUTPUT_MAP_CSV = PROJECT_ROOT / "data" / "patchseq" / "specimen_to_morphology_svg_map.csv"
LAYER_DEPTHS_CSV = PROJECT_ROOT / "patchseq_human_L1-main" / "data" / "human_layer_depths_2023_02_06.csv"
MORPHO_FEATURES_CSV = PROJECT_ROOT / "data" / "patchseq" / "LeeDalley_morpho_features.csv"

# Population-average layer boundary proportions (fraction of cortex thickness)
# Computed from 289 specimens with measured layer data in human_layer_depths_2023_02_06.csv
AVG_LAYER_BOUNDARIES = {
    "L1_L2": 0.089,   # L1/L2 boundary at 8.9% of cortex thickness
    "L2_L3": 0.149,   # L2/L3 boundary at 14.9%
    "L3_L4": 0.427,   # L3/L4 boundary at 42.7%
}
AVG_CORTEX_THICKNESS = 3006  # µm (mean from measured data)

# ── BIL URL templates (tried in order) ───────────────────────────────
# Each entry: (name, url_template) where {sid} is specimen_id
# Templates with specific filenames
BIL_DIRECT_URLS = [
    # Upright SWC files (preferred — already pia-aligned)
    ("group_20230426", "https://download.brainimagelibrary.org/group/20230426/swc/{sid}/{sid}_upright.swc"),
    ("d833_upright", "https://download.brainimagelibrary.org/d8/33/d833ba8bd931f23f/{sid}/{sid}_upright.swc"),
    ("efb9_upright", "https://download.brainimagelibrary.org/ef/b9/efb9b12ba2fab63d/{sid}/{sid}_upright.swc"),
    ("49e6_upright", "https://download.brainimagelibrary.org/49/e6/49e6114ba67eda01/{sid}/{sid}_upright.swc"),
    ("85f4_upright", "https://download.brainimagelibrary.org/85/f4/85f4b93699151f1c/{sid}/{sid}_upright.swc"),
    # Transformed SWC (similar to upright)
    ("241a_transformed", "https://download.brainimagelibrary.org/24/1a/241a10cde842c99b/{sid}/{sid}_transformed.swc"),
]

# Directory-listing approach for 69fe submission (non-standard filenames)
BIL_69FE_DIR = "https://download.brainimagelibrary.org/69/fe/69fe931fee2b2215/{sid}/"

# Compartment colors
MORPH_COLORS = {2: "steelblue", 3: "firebrick", 4: "salmon"}


class DirListingParser(html.parser.HTMLParser):
    """Extract href links from a directory listing HTML page."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val:
                    self.links.append(val)


def parse_swc(swc_path):
    """Parse SWC file into dict of nodes."""
    nodes = {}
    with open(swc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            nodes[nid] = {
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            }
    return nodes


def render_morphology_svg(swc_path, out_path, figsize=(4, 6), scalebar=True, dpi=72,
                          layer_info=None):
    """Render SWC to SVG with optional cortical layer boundaries.

    Parameters
    ----------
    swc_path : Path
    out_path : Path
    figsize : tuple
    scalebar : bool
    dpi : int
    layer_info : dict, optional
        Per-specimen layer depth data with keys:
          absolute_depth, layer_depth, layer_thickness, cortex_thickness, layer
        Used to draw horizontal layer boundary lines.
    """
    nodes = parse_swc(swc_path)
    if not nodes:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    for comp_type, color in MORPH_COLORS.items():
        lines_x, lines_y = [], []
        for nid, node in nodes.items():
            if node["type"] != comp_type:
                continue
            pid = node["parent"]
            if pid < 0 or pid not in nodes:
                continue
            parent = nodes[pid]
            lines_x.extend([parent["x"], node["x"], None])
            lines_y.extend([parent["y"], node["y"], None])
        if lines_x:
            ax.plot(lines_x, lines_y, color=color, linewidth=0.8, solid_capstyle="round")

    # Soma — find soma y for layer boundary computation
    soma_y = None
    for nid, node in nodes.items():
        if node["type"] == 1:
            ax.plot(node["x"], node["y"], "o", color="black", markersize=4, zorder=10)
            soma_y = node["y"]

    ax.set_aspect("equal")

    # ── Layer boundaries ──────────────────────────────────────────────
    if layer_info and soma_y is not None:
        abs_depth = layer_info.get("absolute_depth")
        is_estimated = layer_info.get("estimated", False)

        if abs_depth:
            # Pia position in SWC y-coords: soma_y + absolute_depth
            # (works for both Group A [y=0 is pia] and Group B [positive y])
            pia_y = soma_y + abs_depth
            cortex_thickness = layer_info.get("cortex_thickness", AVG_CORTEX_THICKNESS)

            # Sanity check: cap estimated pia if too far above morphology
            # Deep-layer cells (PVALB, SST) often have overestimated
            # soma_aligned_dist_from_pia, placing pia absurdly high.
            # Cap pia to at most 30% of morphology height above top of tree,
            # and scale cortex thickness proportionally.
            if is_estimated:
                all_ys = [n["y"] for n in nodes.values()]
                morph_y_max = max(all_ys)
                morph_height = morph_y_max - min(all_ys)
                if morph_height > 0:
                    overshoot = (pia_y - morph_y_max) / morph_height
                    if overshoot > 0.5:
                        capped_pia = morph_y_max + 0.3 * morph_height
                        scale = (capped_pia - soma_y) / abs_depth
                        pia_y = capped_pia
                        cortex_thickness = cortex_thickness * scale

            # Build list of boundary lines to draw: [(y_pos, label, is_major)]
            boundaries = []

            if not is_estimated:
                # ── Measured data: draw exact boundaries of the soma's layer ──
                layer_thickness = layer_info.get("layer_thickness")
                layer_depth = layer_info.get("layer_depth")
                layer_name = layer_info.get("layer", "")

                if layer_thickness and layer_depth is not None:
                    soma_layer_top_depth = abs_depth - layer_depth
                    soma_layer_bottom_depth = soma_layer_top_depth + layer_thickness
                    layer_top_y = pia_y - soma_layer_top_depth
                    layer_bottom_y = pia_y - soma_layer_bottom_depth

                    boundaries.append((pia_y, "Pia", True))
                    if soma_layer_top_depth > 10:
                        boundaries.append((layer_top_y, "", False))
                    boundaries.append((layer_bottom_y, "", False))

                    # Layer label position
                    layer_label = layer_name.replace("Layer", "L")
                    if soma_layer_top_depth <= 10:
                        mid_y = (pia_y + layer_bottom_y) / 2
                    else:
                        mid_y = (layer_top_y + layer_bottom_y) / 2
                    boundaries.append((mid_y, layer_label, "label"))
            else:
                # ── Estimated data: draw population-average layer boundaries ──
                boundaries.append((pia_y, "Pia", True))
                for bname, frac in AVG_LAYER_BOUNDARIES.items():
                    by = pia_y - frac * cortex_thickness
                    boundaries.append((by, "", False))

                # Layer labels between boundaries
                prev_frac = 0.0
                for lname, frac in [("L1", AVG_LAYER_BOUNDARIES["L1_L2"]),
                                     ("L2", AVG_LAYER_BOUNDARIES["L2_L3"]),
                                     ("L3", AVG_LAYER_BOUNDARIES["L3_L4"])]:
                    mid_frac = (prev_frac + frac) / 2
                    mid_y = pia_y - mid_frac * cortex_thickness
                    boundaries.append((mid_y, lname, "label"))
                    prev_frac = frac

            # Get morphology extent and draw boundaries
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            data_height = y1 - y0

            # Expand y-range to include boundaries, capped at 40% of data height
            max_expand = data_height * 0.4
            boundary_ys = [b[0] for b in boundaries if b[2] is not True or b[2] != "label"]
            if boundary_ys:
                new_y0 = max(y0 - max_expand, min(y0, min(boundary_ys) - 15))
                new_y1 = min(y1 + max_expand, max(y1, max(boundary_ys) + 15))
                ax.set_ylim(new_y0, new_y1)

            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            label_x = x1 + (x1 - x0) * 0.03

            # Lighter style for estimated boundaries
            base_alpha = 0.45 if is_estimated else 0.6
            label_color = "#777777" if is_estimated else "#444444"
            line_kw = dict(color="#666666", alpha=base_alpha, zorder=1)

            for by, label, kind in boundaries:
                if kind == "label":
                    if y0 <= by <= y1:
                        ax.text(label_x, by, label, fontsize=11, va="center",
                                color=label_color, fontweight="bold",
                                fontstyle="italic", clip_on=False)
                elif kind is True:  # major line (Pia)
                    if y0 - 10 <= by <= y1 + 10:
                        ax.axhline(y=by, linewidth=2.5, linestyle="-", **line_kw)
                        ax.text(label_x, by, label, fontsize=11, va="center",
                                color=label_color, fontweight="bold",
                                clip_on=False)
                else:  # minor boundary line
                    if y0 - 10 <= by <= y1 + 10:
                        ax.axhline(y=by, linewidth=1.8, linestyle="--", **line_kw)

    # ── Scale bar ─────────────────────────────────────────────────────
    if scalebar:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bar_length = 100
        ax.plot([x1 - bar_length, x1], [y0, y0], "k", linewidth=3)
        ax.text(x1 - bar_length, y0 + (y1 - y0) * 0.02, "100 µm", fontsize=8)

    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, transparent=True, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def is_valid_swc(path):
    """Check that a file is a valid SWC file (not an HTML 404 page)."""
    if not path.exists() or path.stat().st_size < 200:
        return False
    with open(path) as f:
        first_line = f.readline()
    return "<html>" not in first_line.lower()


def download_url(url, out_path, timeout=15):
    """Download a URL to a file. Returns True on success."""
    try:
        urllib.request.urlretrieve(url, out_path)
        return is_valid_swc(out_path)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        if out_path.exists():
            out_path.unlink()
        return False


def download_swc_direct(specimen_id, out_path):
    """Try downloading SWC from known direct URL patterns.
    Returns (url, source_name) on success, or (None, None).
    """
    sid = str(specimen_id)
    for name, url_template in BIL_DIRECT_URLS:
        url = url_template.format(sid=sid)
        if download_url(url, out_path):
            return url, name
        # Clean up failed download
        if out_path.exists():
            out_path.unlink()
    return None, None


def download_swc_69fe(specimen_id, out_path):
    """Download SWC from the 69fe submission (non-standard filenames).
    Lists the directory, finds the .swc file, and downloads it.
    Returns (url, source_name) on success, or (None, None).
    """
    sid = str(specimen_id)
    dir_url = BIL_69FE_DIR.format(sid=sid)

    try:
        req = urllib.request.Request(dir_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            page_html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None, None

    # Parse directory listing for .swc files
    parser = DirListingParser()
    parser.feed(page_html)
    swc_files = [l for l in parser.links if l.endswith(".swc")]

    if not swc_files:
        return None, None

    # Prefer _m.swc (manual reconstruction) if available
    m_files = [f for f in swc_files if f.endswith("_m.swc")]
    target = m_files[0] if m_files else swc_files[0]

    file_url = dir_url + target
    if download_url(file_url, out_path):
        return file_url, "69fe_dirlist"
    if out_path.exists():
        out_path.unlink()
    return None, None


def download_swc(specimen_id, out_path):
    """Try all download strategies. Returns (url, source) or (None, None)."""
    # Strategy 1: Direct URL patterns (fast)
    url, source = download_swc_direct(specimen_id, out_path)
    if url:
        return url, source

    # Strategy 2: 69fe directory listing (slower, but handles non-standard names)
    url, source = download_swc_69fe(specimen_id, out_path)
    if url:
        return url, source

    return None, None


def load_layer_depths():
    """Load per-specimen cortical layer depth data from multiple sources.

    Priority:
      1. Measured layer data from human_layer_depths CSV (87 cells)
      2. Estimated from soma_aligned_dist_from_pia + population-average
         layer proportions (128 additional cells)

    Returns dict: specimen_id -> {absolute_depth, layer_depth, layer_thickness,
                                   cortex_thickness, layer, estimated}
    """
    layer_info = {}

    # ── Source 1: Measured layer data ──
    if LAYER_DEPTHS_CSV.exists():
        df = pd.read_csv(LAYER_DEPTHS_CSV)
        for _, row in df.iterrows():
            if row.get("errors") not in ("[]",):
                continue
            sid = int(row["specimen_id"])
            try:
                info = {
                    "absolute_depth": float(row["absolute_depth"]),
                    "layer_depth": float(row["layer_depth"]),
                    "layer_thickness": float(row["layer_thickness"]),
                    "cortex_thickness": float(row["cortex_thickness"]),
                    "layer": str(row.get("layer", "")),
                    "estimated": False,
                }
                if np.isnan(info["absolute_depth"]) or np.isnan(info["layer_thickness"]):
                    continue
                layer_info[sid] = info
            except (ValueError, TypeError):
                continue
        print(f"  Measured layer data: {len(layer_info)} specimens")
    else:
        print(f"  WARNING: Layer depths CSV not found: {LAYER_DEPTHS_CSV}")

    # ── Source 2: Estimated from morphology features (soma_aligned_dist_from_pia) ──
    n_estimated = 0
    if MORPHO_FEATURES_CSV.exists():
        mf = pd.read_csv(MORPHO_FEATURES_CSV)
        for _, row in mf.iterrows():
            sid = int(row["specimen_id"])
            if sid in layer_info:
                continue  # already have measured data
            pia_dist = row.get("soma_aligned_dist_from_pia", np.nan)
            if pd.isna(pia_dist) or float(pia_dist) <= 0:
                continue
            pia_dist = float(pia_dist)
            # Use pia distance as approximate absolute_depth
            # Assign estimated layer based on population-average proportions
            norm_depth = pia_dist / AVG_CORTEX_THICKNESS
            if norm_depth < AVG_LAYER_BOUNDARIES["L1_L2"]:
                est_layer = "Layer1"
            elif norm_depth < AVG_LAYER_BOUNDARIES["L2_L3"]:
                est_layer = "Layer2"
            elif norm_depth < AVG_LAYER_BOUNDARIES["L3_L4"]:
                est_layer = "Layer3"
            else:
                est_layer = "Layer4+"
            layer_info[sid] = {
                "absolute_depth": pia_dist,
                "cortex_thickness": AVG_CORTEX_THICKNESS,
                "layer": est_layer,
                "estimated": True,
            }
            n_estimated += 1
        print(f"  Estimated from morpho features: {n_estimated} specimens")
    else:
        print(f"  WARNING: Morpho features CSV not found: {MORPHO_FEATURES_CSV}")

    return layer_info


def main():
    SWC_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata to get specimen IDs with morphology
    meta = pd.read_csv(METADATA_CSV)
    morph_cells = meta[meta["has_morphology"] == True].copy()
    print(f"Cells with has_morphology=True: {len(morph_cells)}")

    # Load layer depth data for drawing cortical boundaries
    layer_depths = load_layer_depths()
    print(f"Layer depth data available for: {len(layer_depths)} specimens")

    # Check --rerender flag (force re-render all SVGs even if they exist)
    force_rerender = "--rerender" in sys.argv
    if force_rerender:
        print("  --rerender: forcing re-render of all SVGs")

    # Track results
    results = []
    n_downloaded = 0
    n_already = 0
    n_failed = 0
    n_rendered = 0
    n_with_layers = 0
    source_counts = {}

    for i, (_, row) in enumerate(morph_cells.iterrows()):
        sid = int(row["specimen_id"])
        dataset = row.get("dataset", "unknown")
        swc_path = SWC_DIR / f"{sid}_upright.swc"
        svg_path = SVG_DIR / f"{sid}_morphology.svg"

        # Download if needed
        if swc_path.exists() and is_valid_swc(swc_path):
            n_already += 1
            status = "cached"
        else:
            url, source = download_swc(sid, swc_path)
            if url:
                n_downloaded += 1
                source_counts[source] = source_counts.get(source, 0) + 1
                status = "downloaded"
                print(f"  [{i+1}/{len(morph_cells)}] {sid} ({dataset}): downloaded [{source}]")
            else:
                n_failed += 1
                status = "not_found"
                print(f"  [{i+1}/{len(morph_cells)}] {sid} ({dataset}): NOT FOUND on BIL")
                results.append({
                    "specimen_id": sid,
                    "dataset": dataset,
                    "swc_path": "",
                    "svg_path": "",
                    "status": "not_found",
                })
                continue

        # Render SVG (force if --rerender, or if missing/empty)
        needs_render = force_rerender or not svg_path.exists() or svg_path.stat().st_size < 100
        if needs_render:
            li = layer_depths.get(sid)
            if li:
                n_with_layers += 1
            result = render_morphology_svg(swc_path, svg_path, layer_info=li)
            if result:
                n_rendered += 1
                size_kb = svg_path.stat().st_size / 1024
                layer_tag = f" [+layers: {li['layer']}]" if li else ""
                if status != "cached" or force_rerender:
                    print(f"  [{i+1}/{len(morph_cells)}] {sid}: SVG {size_kb:.0f} KB{layer_tag}")
            else:
                print(f"    -> SVG render failed for {sid}")
                results.append({
                    "specimen_id": sid,
                    "dataset": dataset,
                    "swc_path": str(swc_path),
                    "svg_path": "",
                    "status": "render_failed",
                })
                continue

        # Record relative SVG path for the webapp
        rel_svg = f"morphology_svgs/{sid}_morphology.svg"
        results.append({
            "specimen_id": sid,
            "dataset": dataset,
            "swc_path": str(swc_path),
            "svg_path": rel_svg,
            "status": "ok",
        })

    # Save mapping CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_MAP_CSV, index=False)

    n_ok = len(df[df["status"] == "ok"])
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Already cached SWCs: {n_already}")
    print(f"  Newly downloaded: {n_downloaded}")
    if source_counts:
        for src, cnt in sorted(source_counts.items()):
            print(f"    {src}: {cnt}")
    print(f"  Not found on BIL: {n_failed}")
    print(f"  SVGs rendered: {n_rendered}")
    print(f"  With layer boundaries: {n_with_layers}")
    print(f"  Total with SVG: {n_ok}/{len(morph_cells)}")
    print(f"  Map saved: {OUTPUT_MAP_CSV}")


if __name__ == "__main__":
    main()
