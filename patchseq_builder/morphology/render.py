"""
render.py -- Render SWC morphology files to SVG with optional cortical layer boundaries.

Loads per-specimen cortical layer depth data from two sources:
  1. Measured layer data (human_layer_depths CSV) -- 87 cells with precise boundaries
  2. Estimated from soma_aligned_dist_from_pia + population-average layer proportions

Renders each SWC to an SVG showing dendrites (blue), axon (red), and soma (black),
with horizontal lines marking cortical layer boundaries when available.

All constants and paths are imported from config.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from patchseq_builder.config import (
    AVG_CORTEX_THICKNESS,
    AVG_LAYER_BOUNDARIES,
    PIA_OVERSHOOT_CAP,
    PIA_OVERSHOOT_TARGET,
    MORPH_COLORS,
    L1_LAYER_DEPTHS_CSV,
    LD_MORPHO_FEATURES_CSV,
)
from patchseq_builder.morphology.download import parse_swc
from patchseq_builder.morphology.orientation import INVERTED_SPECIMEN_IDS, flip_swc_y


def _layer_to_norm_depth_midpoint(cortical_layer_val):
    """Map cortical_layer metadata value to expected normalized depth midpoint.

    Returns the midpoint of the normalized depth range for the given layer,
    using AVG_LAYER_BOUNDARIES to define layer extents. Returns None for
    unrecognized values.
    """
    layer = str(cortical_layer_val).strip().replace(".0", "")
    if layer == "1":
        return AVG_LAYER_BOUNDARIES["L1_L2"] / 2
    elif layer == "2":
        return (AVG_LAYER_BOUNDARIES["L1_L2"] + AVG_LAYER_BOUNDARIES["L2_L3"]) / 2
    elif layer == "3":
        return (AVG_LAYER_BOUNDARIES["L2_L3"] + AVG_LAYER_BOUNDARIES["L3_L4"]) / 2
    elif layer in ("4", "5", "5_6", "6"):
        return (AVG_LAYER_BOUNDARIES["L3_L4"] + 1.0) / 2
    return None


def load_layer_depths(cortical_layer_map=None) -> dict:
    """Load per-specimen cortical layer depth data from measured + estimated sources.

    Priority:
      1. Measured layer data from human_layer_depths CSV (~87 cells)
      2. Estimated from soma_aligned_dist_from_pia + population-average
         layer proportions (~128 additional cells)

    Parameters
    ----------
    cortical_layer_map : dict, optional
        Mapping of specimen_id (int) -> cortical_layer value from metadata.
        When provided, estimated cells use this to derive a cell-specific
        cortex thickness that places the soma within its annotated layer.

    Returns
    -------
    dict
        specimen_id -> {absolute_depth, layer_depth, layer_thickness,
                        cortex_thickness, layer, estimated}
    """
    layer_info = {}

    # -- Source 1: Measured layer data --
    if L1_LAYER_DEPTHS_CSV.exists():
        df = pd.read_csv(L1_LAYER_DEPTHS_CSV)
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
        print(f"  WARNING: Layer depths CSV not found: {L1_LAYER_DEPTHS_CSV}")

    # -- Source 2: Estimated from morphology features (soma_aligned_dist_from_pia) --
    n_estimated = 0
    if LD_MORPHO_FEATURES_CSV.exists():
        mf = pd.read_csv(LD_MORPHO_FEATURES_CSV)
        for _, row in mf.iterrows():
            sid = int(row["specimen_id"])
            if sid in layer_info:
                continue  # already have measured data
            pia_dist = row.get("soma_aligned_dist_from_pia", np.nan)
            if pd.isna(pia_dist) or float(pia_dist) <= 0:
                continue
            pia_dist = float(pia_dist)
            # Use cortical_layer metadata to derive cell-specific cortex thickness
            # so the soma appears in the correct layer in the SVG
            cortical_layer_val = cortical_layer_map.get(sid) if cortical_layer_map else None
            midpoint = _layer_to_norm_depth_midpoint(cortical_layer_val) if cortical_layer_val is not None else None

            if midpoint is not None:
                cell_cortex_thickness = pia_dist / midpoint
                layer_num = str(cortical_layer_val).strip().replace(".0", "")
                est_layer = f"Layer{layer_num}"
            else:
                cell_cortex_thickness = AVG_CORTEX_THICKNESS
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
                "cortex_thickness": cell_cortex_thickness,
                "layer": est_layer,
                "estimated": True,
            }
            n_estimated += 1
        print(f"  Estimated from morpho features: {n_estimated} specimens")
    else:
        print(f"  WARNING: Morpho features CSV not found: {LD_MORPHO_FEATURES_CSV}")

    return layer_info


def render_morphology_svg(swc_path, out_path, layer_info=None, figsize=(4, 6),
                          scalebar=True, dpi=72):
    """Render SWC to SVG with optional cortical layer boundaries.

    Draws dendrites (steelblue), axon (firebrick/salmon), and soma (black).
    When layer_info is provided, draws horizontal lines marking layer
    boundaries. For estimated depths, pia overshoot is capped to prevent
    absurdly high pia placement on deep-layer cells.

    Parameters
    ----------
    swc_path : str or Path
        Path to the SWC file.
    out_path : str or Path
        Path for the output SVG file.
    layer_info : dict, optional
        Per-specimen layer depth data with keys:
          absolute_depth, layer_depth, layer_thickness, cortex_thickness,
          layer, estimated
    figsize : tuple
        Figure size in inches.
    scalebar : bool
        Whether to draw a 100 um scale bar.
    dpi : int
        Resolution for the SVG output.

    Returns
    -------
    Path or None
        The output path on success, None on failure.
    """
    nodes = parse_swc(swc_path)
    if not nodes:
        return None

    # Flip y-axis for known inverted specimens
    swc_stem = Path(swc_path).stem  # e.g. "548480353_upright"
    try:
        specimen_id = int(swc_stem.split("_")[0])
    except (ValueError, IndexError):
        specimen_id = None
    if specimen_id in INVERTED_SPECIMEN_IDS:
        nodes = flip_swc_y(nodes)

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
            ax.plot(lines_x, lines_y, color=color, linewidth=0.8,
                    solid_capstyle="round")

    # Soma -- find soma y for layer boundary computation
    soma_y = None
    for nid, node in nodes.items():
        if node["type"] == 1:
            ax.plot(node["x"], node["y"], "o", color="black", markersize=4,
                    zorder=10)
            soma_y = node["y"]

    ax.set_aspect("equal")

    # -- Layer boundaries --
    if layer_info and soma_y is not None:
        abs_depth = layer_info.get("absolute_depth")
        is_estimated = layer_info.get("estimated", False)

        if abs_depth:
            # Pia position in SWC y-coords: soma_y + absolute_depth
            pia_y = soma_y + abs_depth
            cortex_thickness = layer_info.get("cortex_thickness",
                                              AVG_CORTEX_THICKNESS)

            # Sanity check: cap estimated pia if too far above morphology.
            # Deep-layer cells (PVALB, SST) often have overestimated
            # soma_aligned_dist_from_pia, placing pia absurdly high.
            if is_estimated:
                all_ys = [n["y"] for n in nodes.values()]
                morph_y_max = max(all_ys)
                morph_height = morph_y_max - min(all_ys)
                if morph_height > 0:
                    overshoot = (pia_y - morph_y_max) / morph_height
                    if overshoot > PIA_OVERSHOOT_CAP:
                        capped_pia = morph_y_max + PIA_OVERSHOOT_TARGET * morph_height
                        scale = (capped_pia - soma_y) / abs_depth
                        pia_y = capped_pia
                        cortex_thickness = cortex_thickness * scale

            # Build list of boundary lines: [(y_pos, label, kind)]
            # kind: True = major (Pia), False = minor, "label" = text only
            boundaries = []

            if not is_estimated:
                # Measured data: draw exact boundaries of the soma's layer
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
                # Estimated data: draw population-average layer boundaries
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
            boundary_ys = [b[0] for b in boundaries
                           if b[2] is not True or b[2] != "label"]
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
                        ax.axhline(y=by, linewidth=2.5, linestyle="-",
                                   **line_kw)
                        ax.text(label_x, by, label, fontsize=11, va="center",
                                color=label_color, fontweight="bold",
                                clip_on=False)
                else:  # minor boundary line
                    if y0 - 10 <= by <= y1 + 10:
                        ax.axhline(y=by, linewidth=1.8, linestyle="--",
                                   **line_kw)

    # -- Scale bar --
    if scalebar:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bar_length = 100
        ax.plot([x1 - bar_length, x1], [y0, y0], "k", linewidth=3)
        ax.text(x1 - bar_length, y0 + (y1 - y0) * 0.02, "100 \u00b5m",
                fontsize=8)

    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, transparent=True, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
