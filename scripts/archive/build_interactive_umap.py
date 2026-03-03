#!/usr/bin/env python
"""
build_interactive_umap.py — Generate an interactive HTML viewer with linked
gene expression and electrophysiology UMAPs for patch-seq cells integrated
with the GABAergic SEA-AD snRNA-seq reference.

Features:
  - Side-by-side: expression UMAP (left) and ephys UMAP (right)
  - Linked highlighting: hover on a cell in one UMAP highlights it in the other
  - Hover tooltips on patch-seq cells only (snRNA-seq = background, no tooltips)
  - Toggle subclasses on/off via legend clicks
  - Dropdown to switch between Subclass / Supertype / Modality coloring
  - Allen Institute standard colors

Requires: No additional Python packages (json, pandas, numpy, anndata only)

Output:
  results/figures/patchseq_umap_interactive.html
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import anndata as ad

PROJECT_ROOT = Path("/Users/shreejoy/Github/patch_seq_lee")
CACHE_H5AD = PROJECT_ROOT / "results" / "intermediates" / "patchseq_reference_combined.h5ad"
PATCHSEQ_H5AD = PROJECT_ROOT / "data" / "patchseq" / "patchseq_combined.h5ad"
COLORS_JSON = Path("/Users/shreejoy/Github/SCZ_Xenium/output/deploy/index.json")
SVG_MAP_CSV = PROJECT_ROOT / "data" / "patchseq" / "specimen_to_svg_map.csv"
MORPH_MAP_CSV = PROJECT_ROOT / "data" / "patchseq" / "specimen_to_morphology_svg_map.csv"
OUTPUT_HTML = PROJECT_ROOT / "results" / "figures" / "patchseq_umap_interactive.html"


def load_data():
    """Load expression UMAP (from combined) and ephys UMAP (from patchseq)."""
    print("Loading cached combined AnnData (expression UMAP)...")
    combined = ad.read_h5ad(str(CACHE_H5AD))
    print(f"  {combined.shape[0]:,} cells")

    print("Loading patchseq_combined.h5ad (ephys UMAP)...")
    ps = ad.read_h5ad(str(PATCHSEQ_H5AD))
    print(f"  {ps.shape[0]:,} cells")

    # Expression UMAP data
    expr_umap = combined.obsm["X_umap"]
    expr_obs = combined.obs.copy()
    expr_obs["umap_1"] = expr_umap[:, 0]
    expr_obs["umap_2"] = expr_umap[:, 1]

    # Ephys UMAP data (only cells with valid coordinates)
    ephys_umap = np.array(ps.obsm["X_umap_ephys"])
    valid_ephys = ~np.isnan(ephys_umap).any(axis=1)
    ps_ephys = ps[valid_ephys].copy()
    ephys_umap_valid = ephys_umap[valid_ephys]

    ephys_obs = ps_ephys.obs.copy()
    ephys_obs["umap_1"] = ephys_umap_valid[:, 0]
    ephys_obs["umap_2"] = ephys_umap_valid[:, 1]

    # Add kNN labels from combined to ephys data
    # Map: patchseq obs_names → combined obs_names (with ps_ prefix)
    ps_to_combined = {name: f"ps_{name}" for name in ephys_obs.index}

    for col in ["knn_subclass", "knn_subclass_conf", "knn_supertype", "knn_supertype_conf"]:
        if col in combined.obs.columns:
            vals = []
            for name in ephys_obs.index:
                cname = ps_to_combined[name]
                if cname in combined.obs.index:
                    vals.append(combined.obs.loc[cname, col])
                else:
                    vals.append(np.nan)
            ephys_obs[col] = vals

    # Carry over transcriptomic type columns from patchseq h5ad to expression obs
    # (these may not be in the combined cache if it was generated before this update)
    for col in ["transcriptomic_type_original", "l1_ttype",
                 "subclass_scANVI", "supertype_scANVI",
                 "subclass_conf_scANVI", "supertype_conf_scANVI"]:
        if col in ps.obs.columns and col not in expr_obs.columns:
            ps_vals = ps.obs[col]
            expr_obs[col] = np.nan
            for ps_name in ps_vals.index:
                cname = f"ps_{ps_name}"
                if cname in expr_obs.index:
                    expr_obs.loc[cname, col] = ps_vals[ps_name]

    # Also carry scANVI labels to ephys_obs
    for col in ["subclass_scANVI", "supertype_scANVI",
                 "subclass_conf_scANVI", "supertype_conf_scANVI"]:
        if col in ps.obs.columns and col not in ephys_obs.columns:
            vals = []
            for name in ephys_obs.index:
                if name in ps.obs.index:
                    vals.append(ps.obs.at[name, col])
                else:
                    vals.append(np.nan)
            ephys_obs[col] = vals

    # ── Build consensus labels: prefer scANVI, fall back to kNN ──
    # Normalize scANVI names to match Allen color keys (e.g. Lamp5_Lhx6 -> Lamp5 Lhx6)
    scanvi_name_norm = {"Lamp5_Lhx6": "Lamp5 Lhx6"}

    for obs_df in [expr_obs, ephys_obs]:
        # Subclass: scANVI preferred, kNN fallback
        # Convert from Categorical to str to avoid dtype mismatch on assignment
        display_sub = obs_df.get("subclass_scANVI", pd.Series(dtype=str)).astype(str).copy()
        display_sub = display_sub.replace("nan", np.nan)
        display_sub = display_sub.map(lambda x: scanvi_name_norm.get(x, x) if pd.notna(x) else x)
        knn_sub = obs_df.get("knn_subclass", pd.Series(dtype=str)).astype(str).replace("nan", np.nan)
        missing = display_sub.isna()
        display_sub.loc[missing] = knn_sub.loc[missing]
        obs_df["display_subclass"] = display_sub

        # Supertype: scANVI preferred, kNN fallback
        display_sup = obs_df.get("supertype_scANVI", pd.Series(dtype=str)).astype(str).copy()
        display_sup = display_sup.replace("nan", np.nan)
        knn_sup = obs_df.get("knn_supertype", pd.Series(dtype=str)).astype(str).replace("nan", np.nan)
        missing = display_sup.isna()
        display_sup.loc[missing] = knn_sup.loc[missing]
        obs_df["display_supertype"] = display_sup

    n_scanvi_expr = expr_obs.loc[expr_obs.get("batch", "") == "patch-seq", "subclass_scANVI"].notna().sum()
    n_scanvi_ephys = ephys_obs["subclass_scANVI"].notna().sum()
    print(f"  Ephys UMAP: {len(ephys_obs)} cells with valid coordinates")
    print(f"  Consensus labels (scANVI→kNN): expr={n_scanvi_expr} scANVI, ephys={n_scanvi_ephys} scANVI")

    # Load SVG trace mapping (specimen_id -> trace SVG path + FI SVG path)
    svg_map = {}
    if SVG_MAP_CSV.exists():
        svg_df = pd.read_csv(SVG_MAP_CSV)
        for _, srow in svg_df.iterrows():
            sid = int(srow["specimen_id"])
            trace_svg = str(srow.get("trace_svg", ""))
            fi_svg = str(srow.get("fi_svg", ""))
            # Convert paths to relative paths under traces/ for HTTP serving
            # Handles both absolute and relative source paths
            for prefix in [
                str(PROJECT_ROOT / "intraDANDI_explorer-master" / "data" / "traces") + "/",
                "intraDANDI_explorer-master/data/traces/",
            ]:
                trace_svg = trace_svg.replace(prefix, "traces/")
                fi_svg = fi_svg.replace(prefix, "traces/")
            if trace_svg == "nan" or not trace_svg:
                trace_svg = ""
            if fi_svg == "nan" or not fi_svg:
                fi_svg = ""
            svg_map[sid] = {"trace_svg": trace_svg, "fi_svg": fi_svg}
        print(f"  SVG trace mapping: {len(svg_map)} cells")
    else:
        print(f"  WARNING: SVG map not found at {SVG_MAP_CSV}")

    # Load morphology SVG mapping (specimen_id -> morphology SVG path)
    morph_map = {}
    if MORPH_MAP_CSV.exists():
        morph_df = pd.read_csv(MORPH_MAP_CSV)
        for _, mrow in morph_df.iterrows():
            if mrow.get("status") != "ok":
                continue
            mid = int(mrow["specimen_id"])
            morph_map[mid] = str(mrow.get("svg_path", ""))
        print(f"  Morphology SVG mapping: {len(morph_map)} cells")
    else:
        print(f"  WARNING: Morphology map not found at {MORPH_MAP_CSV}")

    # Add SVG paths to expr_obs and ephys_obs
    for obs_df in [expr_obs, ephys_obs]:
        trace_svgs = []
        fi_svgs = []
        morph_svgs = []
        for _, row in obs_df.iterrows():
            sid = row.get("specimen_id", np.nan)
            try:
                sid_int = int(float(sid))
            except (ValueError, TypeError):
                sid_int = None
            if sid_int and sid_int in svg_map:
                trace_svgs.append(svg_map[sid_int]["trace_svg"])
                fi_svgs.append(svg_map[sid_int]["fi_svg"])
            else:
                trace_svgs.append("")
                fi_svgs.append("")
            morph_svgs.append(morph_map.get(sid_int, "") if sid_int else "")
        obs_df["trace_svg"] = trace_svgs
        obs_df["fi_svg"] = fi_svgs
        obs_df["morph_svg"] = morph_svgs

    if "batch" in expr_obs.columns:
        n_with_svg = (expr_obs.loc[expr_obs["batch"] == "patch-seq", "trace_svg"] != "").sum()
        n_with_morph = (expr_obs.loc[expr_obs["batch"] == "patch-seq", "morph_svg"] != "").sum()
    else:
        n_with_svg = (expr_obs["trace_svg"] != "").sum()
        n_with_morph = (expr_obs["morph_svg"] != "").sum()
    print(f"  Patch-seq cells with trace SVGs: {n_with_svg}")
    print(f"  Patch-seq cells with morphology SVGs: {n_with_morph}")

    return expr_obs, ephys_obs


def load_colors():
    """Load Allen Institute standard colors."""
    with open(COLORS_JSON) as f:
        data = json.load(f)
    return data["subclass_colors"], data["supertype_colors"]


def build_hover_text_patchseq(row, source="expression"):
    """Build rich hover text for a patch-seq cell."""
    parts = [f"<b>PATCH-SEQ</b>"]

    sid = row.get("specimen_id", "")
    if pd.notna(sid):
        parts.append(f"Specimen: {sid}")

    ds = row.get("dataset", "")
    if pd.notna(ds):
        parts.append(f"Dataset: {ds}")

    layer = row.get("cortical_layer", "")
    if pd.notna(layer) and str(layer) not in ("", "nan", "ZZ_Missing"):
        parts.append(f"Layer: {layer}")

    orig = row.get("subclass_label", "")
    if pd.notna(orig):
        parts.append(f"Original subclass: {orig}")

    # Original transcriptomic cluster name
    ttype = row.get("transcriptomic_type_original", np.nan)
    if pd.notna(ttype) and str(ttype) != "nan":
        parts.append(f"T-type: {ttype}")

    l1tt = row.get("l1_ttype", np.nan)
    if pd.notna(l1tt) and str(l1tt) != "nan":
        parts.append(f"L1 t-type: {l1tt}")

    parts.append("---")

    disp_sub = row.get("display_subclass", "?")
    if pd.notna(disp_sub) and str(disp_sub) != "nan":
        parts.append(f"Subclass: {disp_sub}")

    disp_sup = row.get("display_supertype", "?")
    if pd.notna(disp_sup) and str(disp_sup) != "nan":
        parts.append(f"Supertype: {disp_sup}")

    # Ephys features — only show in expression UMAP tooltip (not ephys UMAP)
    if source == "expression":
        ephys_cols = {"sag": "Sag", "tau": "Tau (ms)", "input_resistance": "Rin (MOhm)",
                      "rheobase_i": "Rheobase (pA)", "fi_fit_slope": "FI slope",
                      "v_baseline": "Vm (mV)"}
        ephys_parts = []
        for col, label in ephys_cols.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                ephys_parts.append(f"{label}: {float(val):.1f}")

        if ephys_parts:
            parts.append("---")
            parts.extend(ephys_parts)

    return "<br>".join(parts)


def build_html(expr_obs, ephys_obs, subclass_colors, supertype_colors):
    """Build standalone HTML with linked side-by-side UMAPs."""

    is_ref = expr_obs["batch"] == "snRNA-seq"
    is_ps = expr_obs["batch"] == "patch-seq"

    # ── Expression UMAP traces ───────────────────────────────────────

    # Reference: single grey trace, no hover
    ref_data = expr_obs.loc[is_ref]
    ref_trace = {
        "type": "scattergl",
        "mode": "markers",
        "x": ref_data["umap_1"].tolist(),
        "y": ref_data["umap_2"].tolist(),
        "hoverinfo": "skip",
        "marker": {
            "size": 2,
            "color": "#D0D0D0",
            "opacity": 0.15,
        },
        "name": f"snRNA-seq ({len(ref_data):,})",
        "showlegend": True,
    }

    # Build per-cell data for ref (for supertype coloring)
    ref_supertype_colors = [
        supertype_colors.get(st, "#D0D0D0")
        for st in ref_data["ref_supertype"].values
    ]
    ref_subclass_colors = [
        subclass_colors.get(sc, "#D0D0D0")
        for sc in ref_data["ref_subclass"].values
    ]

    # Patch-seq on expression UMAP: one trace per display_subclass (scANVI preferred)
    ps_expr = expr_obs.loc[is_ps].copy()
    expr_ps_traces = []
    expr_ps_meta = []

    # Build cell ID → ephys index mapping
    # ps_expr index is like "ps_P1S4_170214_008_A01"
    # ephys_obs index is like "P1S4_170214_008_A01"
    ps_in_ephys = set(ephys_obs.index)

    for sc_name in sorted(ps_expr["display_subclass"].dropna().unique()):
        mask = ps_expr["display_subclass"] == sc_name
        subset = ps_expr.loc[mask]
        color = subclass_colors.get(sc_name, "#333333")

        hover_texts = []
        cell_ids = []
        custom_data_rows = []
        supertype_list = []
        has_ephys_list = []

        for idx, row in subset.iterrows():
            hover_texts.append(build_hover_text_patchseq(row))
            # Cell ID without ps_ prefix for linking
            cid = idx.replace("ps_", "") if idx.startswith("ps_") else idx
            cell_ids.append(cid)
            supertype_list.append(row.get("display_supertype", "Unknown"))
            has_ephys_list.append(cid in ps_in_ephys)
            # customdata: [cell_id, trace_svg, fi_svg, specimen_id, subclass, dataset, supertype, morph_svg]
            sid = row.get("specimen_id", "")
            custom_data_rows.append([
                cid,
                row.get("trace_svg", ""),
                row.get("fi_svg", ""),
                str(int(float(sid))) if pd.notna(sid) else "",
                str(row.get("display_subclass", "")),
                str(row.get("dataset", "")),
                str(row.get("display_supertype", "")),
                row.get("morph_svg", ""),
            ])

        expr_ps_traces.append({
            "type": "scattergl",
            "mode": "markers",
            "x": subset["umap_1"].tolist(),
            "y": subset["umap_2"].tolist(),
            "text": hover_texts,
            "hoverinfo": "text",
            "customdata": custom_data_rows,
            "marker": {
                "size": 9,
                "color": color,
                "opacity": 0.9,
                "line": {"width": 0.8, "color": "black"},
            },
            "name": f"{sc_name} ({len(subset)})",
            "legendgroup": sc_name,
            "showlegend": True,
        })
        expr_ps_meta.append({
            "subclass": sc_name,
            "color": color,
            "supertypes": supertype_list,
            "has_ephys": has_ephys_list,
        })

    # ── Ephys UMAP traces ────────────────────────────────────────────

    ephys_traces = []
    ephys_meta = []

    # Use display_subclass (scANVI preferred, kNN fallback)
    ephys_subclass_col = "display_subclass"

    for sc_name in sorted(ephys_obs[ephys_subclass_col].dropna().unique()):
        mask = ephys_obs[ephys_subclass_col] == sc_name
        subset = ephys_obs.loc[mask]
        color = subclass_colors.get(sc_name, "#333333")

        hover_texts = []
        cell_ids = []
        custom_data_rows = []
        supertype_list = []

        for idx, row in subset.iterrows():
            hover_texts.append(build_hover_text_patchseq(row, source="ephys"))
            cell_ids.append(idx)
            sup = row.get("display_supertype", "Unknown")
            supertype_list.append(sup if pd.notna(sup) else "Unknown")
            sid = row.get("specimen_id", "")
            custom_data_rows.append([
                idx,
                row.get("trace_svg", ""),
                row.get("fi_svg", ""),
                str(int(float(sid))) if pd.notna(sid) else "",
                str(row.get("display_subclass", sc_name)),
                str(row.get("dataset", "")),
                str(sup),
                row.get("morph_svg", ""),
            ])

        ephys_traces.append({
            "type": "scattergl",
            "mode": "markers",
            "x": subset["umap_1"].tolist(),
            "y": subset["umap_2"].tolist(),
            "text": hover_texts,
            "hoverinfo": "text",
            "customdata": custom_data_rows,
            "marker": {
                "size": 9,
                "color": color,
                "opacity": 0.9,
                "line": {"width": 0.8, "color": "black"},
            },
            "name": f"{sc_name} ({len(subset)})",
            "legendgroup": sc_name,
            "showlegend": False,  # shared legend with expression
        })
        ephys_meta.append({
            "subclass": sc_name,
            "color": color,
            "supertypes": supertype_list,
        })

    # ── Count totals ─────────────────────────────────────────────────
    n_ref = is_ref.sum()
    n_ps = is_ps.sum()
    n_ephys = len(ephys_obs)

    # ── Build HTML ───────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linked UMAP Viewer — Expression + Electrophysiology</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #fafafa;
  }}
  .header {{
    padding: 10px 20px;
    background: white;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .header h1 {{ font-size: 16px; color: #333; white-space: nowrap; }}
  .header .stats {{ color: #666; font-size: 12px; }}
  .controls {{
    display: flex; align-items: center; gap: 8px;
  }}
  .controls label {{ font-size: 12px; font-weight: 600; color: #555; }}
  .controls select {{
    padding: 3px 6px; border: 1px solid #ccc;
    border-radius: 4px; font-size: 12px;
  }}
  .controls button {{
    padding: 3px 10px; border: 1px solid #ccc;
    border-radius: 4px; font-size: 11px; cursor: pointer; background: white;
  }}
  .controls button:hover {{ background: #f0f0f0; }}
  .plots-container {{
    display: flex;
    width: 100%;
    height: 48vh;
  }}
  .plot-panel {{
    flex: 1;
    position: relative;
    border-right: 1px solid #e0e0e0;
  }}
  .plot-panel:last-child {{ border-right: none; }}
  .plot-panel .panel-title {{
    position: absolute; top: 6px; left: 50%;
    transform: translateX(-50%);
    font-size: 13px; font-weight: 600; color: #444;
    background: rgba(255,255,255,0.85); padding: 2px 10px;
    border-radius: 4px; z-index: 10;
    pointer-events: none;
  }}
  .plot-panel .plot {{ width: 100%; height: 100%; }}
  /* Highlight ring */
  .highlight-info {{
    position: absolute; bottom: 8px; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; color: #666; background: rgba(255,255,255,0.9);
    padding: 2px 8px; border-radius: 3px; z-index: 10;
    pointer-events: none; white-space: nowrap;
  }}
  /* Trace panel */
  .trace-panel {{
    border-top: 2px solid #e0e0e0;
    background: white;
    padding: 4px 20px 2px;
    height: 42vh;
    min-height: 200px;
    max-height: 50vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .trace-panel .trace-header {{
    font-size: 12px; font-weight: 600; color: #333;
    margin-bottom: 2px;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
  }}
  .trace-panel .trace-header .trace-info {{
    font-weight: 400; color: #555; font-size: 11px;
  }}
  .trace-images {{
    flex: 1;
    display: flex;
    gap: 6px;
    align-items: stretch;
    justify-content: center;
    overflow: hidden;
    min-height: 0;
  }}
  .trace-images .trace-panel-wrapper {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    position: relative;
  }}
  .trace-images .trace-svg-container {{
    height: 100%;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background: white;
    position: relative;
  }}
  .trace-images .trace-svg-container svg {{
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
  }}
  .panel-empty-msg {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #bbb;
    font-size: 12px;
    font-style: italic;
    pointer-events: none;
  }}
  .svg-axis-label {{
    font-size: 8px;
    color: #888;
    pointer-events: none;
    position: absolute;
    z-index: 5;
    background: rgba(255,255,255,0.7);
    padding: 0 2px;
  }}
  .svg-axis-label.x-label {{
    bottom: 3px;
    left: 50%;
    transform: translateX(-50%);
  }}
  .svg-axis-label.y-label {{
    left: 4px;
    top: 50%;
    transform: translateY(-50%) rotate(-90deg);
    transform-origin: center center;
  }}
  .trace-placeholder {{
    color: #999;
    font-size: 13px;
    text-align: center;
    padding: 20px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Linked UMAP Viewer</h1>
  <span class="stats">
    Expression: {n_ref:,} snRNA-seq + {n_ps:,} patch-seq &nbsp;|&nbsp;
    Ephys: {n_ephys:,} patch-seq cells
  </span>
  <div class="controls">
    <label for="colorBy">Color by:</label>
    <select id="colorBy" onchange="updateColoring()">
      <option value="subclass">Subclass</option>
      <option value="supertype">Supertype</option>
      <option value="modality">Modality</option>
    </select>
    <button onclick="resetViews()">Reset views</button>
  </div>
</div>

<div class="plots-container">
  <div class="plot-panel">
    <div class="panel-title">Gene Expression UMAP (Harmony)</div>
    <div id="exprPlot" class="plot"></div>
    <div id="exprHighlight" class="highlight-info" style="display:none;"></div>
  </div>
  <div class="plot-panel">
    <div class="panel-title">Electrophysiology UMAP</div>
    <div id="ephysPlot" class="plot"></div>
    <div id="ephysHighlight" class="highlight-info" style="display:none;"></div>
  </div>
</div>

<div class="trace-panel">
  <div class="trace-header">
    <span>Morphology &amp; Electrophysiology</span>
    <span id="traceInfo" class="trace-info" style="display:none;"></span>
  </div>
  <div class="trace-images">
    <div class="trace-panel-wrapper" style="display:none;">
      <div id="morphContainer" class="trace-svg-container"></div>
      <div id="morphEmpty" class="panel-empty-msg">No morphology</div>
    </div>
    <div class="trace-panel-wrapper" style="display:none;">
      <div id="traceContainer" class="trace-svg-container"></div>
      <div id="traceEmpty" class="panel-empty-msg">No trace</div>
    </div>
    <div class="trace-panel-wrapper" style="display:none;">
      <div id="fiContainer" class="trace-svg-container"></div>
      <div id="fiEmpty" class="panel-empty-msg">No FI curve</div>
    </div>
    <div id="tracePlaceholder" class="trace-placeholder">
      Hover over a patch-seq cell in either UMAP to view its electrophysiology recording and morphology
    </div>
  </div>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────
const refTrace = {json.dumps(ref_trace)};
const refSubclassColors = {json.dumps(ref_subclass_colors)};
const refSupertypeColors = {json.dumps(ref_supertype_colors)};
const exprPsTraces = {json.dumps(expr_ps_traces)};
const exprPsMeta = {json.dumps(expr_ps_meta)};
const ephysTraces = {json.dumps(ephys_traces)};
const ephysMeta = {json.dumps(ephys_meta)};
const subclassColors = {json.dumps(subclass_colors)};
const supertypeColors = {json.dumps(supertype_colors)};

// ── Cell ID index for cross-highlighting ─────────────────────────
// Build lookup: cellId -> {{plotId, traceIdx, pointIdx, x, y}}
const exprCellIndex = {{}};
const ephysCellIndex = {{}};

// Expression: ref trace is index 0, ps traces start at 1
// customdata format: [cell_id, trace_svg, fi_svg, specimen_id, subclass, dataset]
const exprAllTraces = [refTrace, ...exprPsTraces];
for (let ti = 1; ti < exprAllTraces.length; ti++) {{
  const t = exprAllTraces[ti];
  if (!t.customdata) continue;
  for (let pi = 0; pi < t.customdata.length; pi++) {{
    const cd = t.customdata[pi];
    exprCellIndex[cd[0]] = {{traceIdx: ti, pointIdx: pi, x: t.x[pi], y: t.y[pi], cd: cd}};
  }}
}}

for (let ti = 0; ti < ephysTraces.length; ti++) {{
  const t = ephysTraces[ti];
  if (!t.customdata) continue;
  for (let pi = 0; pi < t.customdata.length; pi++) {{
    const cd = t.customdata[pi];
    ephysCellIndex[cd[0]] = {{traceIdx: ti, pointIdx: pi, x: t.x[pi], y: t.y[pi], cd: cd}};
  }}
}}

// ── Layout ────────────────────────────────────────────────────────
const exprLayout = {{
  xaxis: {{ title: 'UMAP 1', zeroline: false, showgrid: false }},
  yaxis: {{ title: 'UMAP 2', zeroline: false, showgrid: false }},
  hovermode: 'closest',
  margin: {{ l: 50, r: 10, t: 30, b: 50 }},
  paper_bgcolor: '#fafafa',
  plot_bgcolor: 'white',
  legend: {{
    font: {{ size: 10 }},
    itemclick: 'toggle',
    itemdoubleclick: 'toggleothers',
    x: 1.02, y: 1, xanchor: 'left',
  }},
}};

const ephysLayout = {{
  xaxis: {{ title: 'UMAP 1', zeroline: false, showgrid: false }},
  yaxis: {{ title: 'UMAP 2', zeroline: false, showgrid: false }},
  hovermode: 'closest',
  margin: {{ l: 50, r: 10, t: 30, b: 50 }},
  paper_bgcolor: '#fafafa',
  plot_bgcolor: 'white',
  showlegend: false,
}};

const plotConfig = {{
  responsive: true, scrollZoom: true, displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
}};

// ── Render plots ──────────────────────────────────────────────────
Plotly.newPlot('exprPlot', exprAllTraces, exprLayout, plotConfig);
Plotly.newPlot('ephysPlot', ephysTraces, ephysLayout, plotConfig);

// ── Cross-highlighting using SVG annotations (renders on top of WebGL) ──

function makeHighlightAnnotation(x, y) {{
  return {{
    x: x, y: y, xref: 'x', yref: 'y',
    text: '&nbsp;', showarrow: false,
    bordercolor: '#00FF00', borderwidth: 4, borderpad: 12,
    bgcolor: 'rgba(0,255,0,0.18)',
    font: {{ size: 1, color: 'rgba(0,0,0,0)' }},
  }};
}}

function clearHighlights() {{
  Plotly.relayout('exprPlot', {{ annotations: [] }});
  Plotly.relayout('ephysPlot', {{ annotations: [] }});
  document.getElementById('exprHighlight').style.display = 'none';
  document.getElementById('ephysHighlight').style.display = 'none';
}}

function highlightCell(cellId, source) {{
  // Highlight in the OTHER plot
  if (source !== 'expr' && exprCellIndex[cellId]) {{
    const info = exprCellIndex[cellId];
    Plotly.relayout('exprPlot', {{
      annotations: [makeHighlightAnnotation(info.x, info.y)],
    }});
    document.getElementById('exprHighlight').textContent = '\\u2190 linked';
    document.getElementById('exprHighlight').style.display = 'block';
  }} else {{
    Plotly.relayout('exprPlot', {{ annotations: [] }});
    document.getElementById('exprHighlight').style.display = 'none';
  }}

  if (source !== 'ephys' && ephysCellIndex[cellId]) {{
    const info = ephysCellIndex[cellId];
    Plotly.relayout('ephysPlot', {{
      annotations: [makeHighlightAnnotation(info.x, info.y)],
    }});
    document.getElementById('ephysHighlight').textContent = '\\u2190 linked';
    document.getElementById('ephysHighlight').style.display = 'block';
  }} else {{
    Plotly.relayout('ephysPlot', {{ annotations: [] }});
    document.getElementById('ephysHighlight').style.display = 'none';
  }}
}}

// Unhover — clear cross-highlight rings but keep trace panel showing last hovered cell
document.getElementById('exprPlot').on('plotly_unhover', clearHighlights);
document.getElementById('ephysPlot').on('plotly_unhover', clearHighlights);

// ── Trace panel — hover-triggered with inline SVG + color matching ──

// SVG cache: url -> svg text
const svgCache = {{}};
// Track current state
let currentTraceCustomdata = null;
let hoverDebounceTimer = null;
const HOVER_DEBOUNCE_MS = 80;

function getCurrentColor(customdata) {{
  // customdata: [cell_id, trace_svg, fi_svg, specimen_id, subclass, dataset, supertype]
  const mode = document.getElementById('colorBy').value;
  const subclass = customdata[4];
  const supertype = customdata[6] || '';
  if (mode === 'subclass') {{
    return subclassColors[subclass] || '#333333';
  }} else if (mode === 'supertype') {{
    return supertypeColors[supertype] || '#333333';
  }} else {{
    return '#E74C3C';  // modality mode: red for all patch-seq
  }}
}}

async function fetchSvg(url) {{
  if (svgCache[url]) return svgCache[url];
  try {{
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const text = await resp.text();
    svgCache[url] = text;
    return text;
  }} catch (e) {{
    return null;
  }}
}}

function addAxisLabels(containerId, yLabel, xLabel) {{
  const container = document.getElementById(containerId);
  if (!container) return;
  // Remove existing labels
  container.querySelectorAll('.svg-axis-label').forEach(el => el.remove());
  // Y-axis label (rotated, left side)
  const yEl = document.createElement('div');
  yEl.className = 'svg-axis-label y-label';
  yEl.textContent = yLabel;
  container.appendChild(yEl);
  // X-axis label (bottom center)
  const xEl = document.createElement('div');
  xEl.className = 'svg-axis-label x-label';
  xEl.textContent = xLabel;
  container.appendChild(xEl);
}}

function injectAndColorSvg(containerId, svgText, color) {{
  const container = document.getElementById(containerId);
  container.innerHTML = svgText;
  const svg = container.querySelector('svg');
  if (!svg) {{ container.style.display = 'none'; return; }}

  // Make SVG responsive
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  svg.style.width = '100%';
  svg.style.height = '100%';

  // DOM-based recoloring: target data elements, leave axes untouched
  // Strategy: find all styled elements and recolor based on their stroke-width
  svg.querySelectorAll('path, line, use').forEach(el => {{
    const style = el.getAttribute('style') || '';
    // Skip axis tick marks (stroke-width: 0.8) and box borders (stroke-linejoin: miter)
    if (style.includes('stroke-width: 0.8') || style.includes('stroke-linejoin: miter')) return;
    // Skip background fill patches (no stroke, just fill: #ffffff)
    if (style.includes('fill: #ffffff') && !style.includes('stroke:')) return;

    // Data trace lines: stroke-width: 1.5 (voltage sweeps and FI curve line)
    if (style.includes('stroke-width: 1.5')) {{
      el.style.stroke = color;
      el.style.strokeOpacity = '0.7';
    }}
    // Data point markers: <use> elements with stroke but NO stroke-width
    // (tick <use> elements have stroke-width: 0.8, already skipped above)
    else if (el.tagName === 'use' && style.includes('stroke:') && !style.includes('stroke-width')) {{
      el.style.stroke = color;
      el.style.fill = color;
    }}
    // Marker defs: <path> with id starting with 'm' (circle marker shapes for FI dots)
    else if (el.tagName === 'path' && el.id && /^m[a-f0-9]{{8,}}$/.test(el.id)) {{
      if (style.includes('stroke: #000000') || style.includes('stroke:#000000')) {{
        el.style.stroke = color;
        el.style.fill = color;
      }}
    }}
  }});

  container.style.display = 'block';
}}

function recolorCurrentTrace() {{
  // Recolor already-injected SVGs in the DOM (called on dropdown change)
  if (!currentTraceCustomdata) return;
  const color = getCurrentColor(currentTraceCustomdata);
  // Recolor ephys traces only (not morphology — keep axon/dendrite colors)
  ['traceContainer', 'fiContainer'].forEach(containerId => {{
    const container = document.getElementById(containerId);
    const svg = container.querySelector('svg');
    if (!svg) return;
    svg.querySelectorAll('path, line, use').forEach(el => {{
      const style = el.getAttribute('style') || '';
      if (style.includes('stroke-width: 0.8') || style.includes('stroke-linejoin: miter')) return;
      if (style.includes('fill: #ffffff') && !style.includes('stroke:')) return;
      if (style.includes('stroke-width: 1.5')) {{
        el.style.stroke = color;
        el.style.strokeOpacity = '0.7';
      }} else if (el.tagName === 'use' && style.includes('stroke:') && !style.includes('stroke-width')) {{
        el.style.stroke = color;
        el.style.fill = color;
      }} else if (el.tagName === 'path' && el.id && /^m[a-f0-9]{{8,}}$/.test(el.id)) {{
        const origStyle = el.getAttribute('style') || '';
        if (origStyle.includes('stroke:') && !origStyle.includes('stroke-width:')) {{
          el.style.stroke = color;
          el.style.fill = color;
        }}
      }}
    }});
  }});
  // Update info color dot
  const traceInfo = document.getElementById('traceInfo');
  if (traceInfo) {{
    const dot = traceInfo.querySelector('span[style]');
    if (dot) dot.style.color = color;
  }}
}}

async function showTrace(customdata) {{
  const traceSvgUrl = customdata[1];
  const fiSvgUrl = customdata[2];
  const specId = customdata[3];
  const subclass = customdata[4];
  const dataset = customdata[5];
  const morphSvgUrl = customdata[7] || '';
  const color = getCurrentColor(customdata);

  const traceContainer = document.getElementById('traceContainer');
  const fiContainer = document.getElementById('fiContainer');
  const morphContainer = document.getElementById('morphContainer');
  const traceInfo = document.getElementById('traceInfo');
  const tracePlaceholder = document.getElementById('tracePlaceholder');

  // Show all three panel wrappers (fixed layout)
  morphContainer.parentElement.style.display = '';
  traceContainer.parentElement.style.display = '';
  fiContainer.parentElement.style.display = '';

  // Remember current cell for recoloring on dropdown change
  currentTraceCustomdata = customdata;

  const morphEmpty = document.getElementById('morphEmpty');
  const traceEmpty = document.getElementById('traceEmpty');
  const fiEmpty = document.getElementById('fiEmpty');

  // Morphology SVG (no recoloring — keep original axon/dendrite colors)
  if (morphSvgUrl) {{
    const svgText = await fetchSvg(morphSvgUrl);
    if (svgText) {{
      morphContainer.innerHTML = svgText;
      const svg = morphContainer.querySelector('svg');
      if (svg) {{
        svg.removeAttribute('width');
        svg.removeAttribute('height');
        svg.style.width = '100%';
        svg.style.height = '100%';
      }}
      morphContainer.style.display = 'block';
      morphEmpty.style.display = 'none';
    }} else {{
      morphContainer.style.display = 'none';
      morphEmpty.style.display = 'block';
    }}
  }} else {{
    morphContainer.innerHTML = '';
    morphContainer.style.display = 'none';
    morphEmpty.style.display = 'block';
  }}

  if (traceSvgUrl) {{
    const svgText = await fetchSvg(traceSvgUrl);
    if (svgText) {{
      injectAndColorSvg('traceContainer', svgText, color);
      addAxisLabels('traceContainer', 'Voltage (mV)', 'Time (s)');
      traceEmpty.style.display = 'none';
    }} else {{
      traceContainer.style.display = 'none';
      traceEmpty.style.display = 'block';
    }}
  }} else {{
    traceContainer.innerHTML = '';
    traceContainer.style.display = 'none';
    traceEmpty.style.display = 'block';
  }}

  if (fiSvgUrl) {{
    const svgText = await fetchSvg(fiSvgUrl);
    if (svgText) {{
      injectAndColorSvg('fiContainer', svgText, color);
      addAxisLabels('fiContainer', 'Spikes', 'Current (pA)');
      fiEmpty.style.display = 'none';
    }} else {{
      fiContainer.style.display = 'none';
      fiEmpty.style.display = 'block';
    }}
  }} else {{
    fiContainer.innerHTML = '';
    fiContainer.style.display = 'none';
    fiEmpty.style.display = 'block';
  }}

  const hasAny = traceSvgUrl || fiSvgUrl || morphSvgUrl;
  if (hasAny) {{
    tracePlaceholder.style.display = 'none';
    const parts = ['<b>Specimen ' + specId + '</b>'];
    parts.push('<span style="color:' + color + '; font-weight:700;">&#9679;</span> ' + subclass);
    parts.push('Dataset: ' + dataset);
    const modalities = [];
    if (morphSvgUrl) modalities.push('morphology');
    if (traceSvgUrl) modalities.push('ephys');
    if (modalities.length) parts.push('[' + modalities.join(' + ') + ']');
    traceInfo.innerHTML = parts.join(' &nbsp;|&nbsp; ');
    traceInfo.style.display = 'block';
  }} else {{
    tracePlaceholder.textContent = 'No morphology or electrophysiology data available for specimen ' + specId;
    tracePlaceholder.style.display = 'block';
    traceInfo.style.display = 'none';
  }}
}}

// Hover handlers — show ephys trace on hover with debounce
function onCellHover(data) {{
  const pt = data.points[0];
  if (!pt.customdata || !pt.customdata[0]) return;
  // Debounce to avoid rapid SVG fetching
  clearTimeout(hoverDebounceTimer);
  const cd = pt.customdata;
  hoverDebounceTimer = setTimeout(() => {{
    // Only update if this is a different cell
    if (!currentTraceCustomdata || cd[0] !== currentTraceCustomdata[0]) {{
      showTrace(cd);
    }}
  }}, HOVER_DEBOUNCE_MS);
}}

document.getElementById('exprPlot').on('plotly_hover', function(data) {{
  const pt = data.points[0];
  if (pt.customdata) {{
    highlightCell(pt.customdata[0], 'expr');
    onCellHover(data);
  }}
}});

document.getElementById('ephysPlot').on('plotly_hover', function(data) {{
  const pt = data.points[0];
  if (pt.customdata) {{
    highlightCell(pt.customdata[0], 'ephys');
    onCellHover(data);
  }}
}});

// ── View switching ────────────────────────────────────────────────
function updateColoring() {{
  const mode = document.getElementById('colorBy').value;

  // Update reference trace (index 0 in exprPlot)
  if (mode === 'subclass') {{
    Plotly.restyle('exprPlot', {{
      'marker.color': [refSubclassColors],
      'marker.opacity': 0.15, 'marker.size': 2,
      'name': 'snRNA-seq ({n_ref:,})',
    }}, [0]);
  }} else if (mode === 'supertype') {{
    Plotly.restyle('exprPlot', {{
      'marker.color': [refSupertypeColors],
      'marker.opacity': 0.15, 'marker.size': 2,
      'name': 'snRNA-seq ({n_ref:,})',
    }}, [0]);
  }} else {{
    Plotly.restyle('exprPlot', {{
      'marker.color': ['#D0D0D0'],
      'marker.opacity': 0.1, 'marker.size': 2,
      'name': 'snRNA-seq ({n_ref:,})',
    }}, [0]);
  }}

  // Update patch-seq traces in expression UMAP (indices 1..N)
  for (let i = 0; i < exprPsTraces.length; i++) {{
    const meta = exprPsMeta[i];
    const n = exprPsTraces[i].x.length;
    const traceIdx = i + 1;

    if (mode === 'subclass') {{
      Plotly.restyle('exprPlot', {{
        'marker.color': [Array(n).fill(meta.color)],
        'name': meta.subclass + ' (' + n + ')',
      }}, [traceIdx]);
    }} else if (mode === 'supertype') {{
      const colors = meta.supertypes.map(s => supertypeColors[s] || '#333333');
      Plotly.restyle('exprPlot', {{
        'marker.color': [colors],
        'name': meta.subclass + ' (' + n + ')',
      }}, [traceIdx]);
    }} else {{
      Plotly.restyle('exprPlot', {{
        'marker.color': [Array(n).fill('#E74C3C')],
        'name': 'patch-seq (' + n + ')',
      }}, [traceIdx]);
    }}
  }}

  // Update ephys traces
  for (let i = 0; i < ephysTraces.length; i++) {{
    const meta = ephysMeta[i];
    const n = ephysTraces[i].x.length;

    if (mode === 'subclass') {{
      Plotly.restyle('ephysPlot', {{
        'marker.color': [Array(n).fill(meta.color)],
      }}, [i]);
    }} else if (mode === 'supertype') {{
      const colors = meta.supertypes.map(s => supertypeColors[s] || '#333333');
      Plotly.restyle('ephysPlot', {{
        'marker.color': [colors],
      }}, [i]);
    }} else {{
      Plotly.restyle('ephysPlot', {{
        'marker.color': [Array(n).fill('#E74C3C')],
      }}, [i]);
    }}
  }}

  // Recolor currently displayed trace to match new coloring mode
  recolorCurrentTrace();
}}

function resetViews() {{
  Plotly.relayout('exprPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
  Plotly.relayout('ephysPlot', {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
}}

// ── Legend sync: clicking legend in expr should toggle both plots ──
document.getElementById('exprPlot').on('plotly_restyle', function(data) {{
  // When a trace visibility changes in expr, sync to ephys
  if (!data || !data[0] || !data[0].visible) return;
  const traceIndices = data[1];
  if (!traceIndices) return;

  for (const ti of traceIndices) {{
    if (ti === 0) continue; // skip ref trace
    const psMeta = exprPsMeta[ti - 1];
    if (!psMeta) continue;
    // Find matching ephys trace by subclass
    const ephysIdx = ephysMeta.findIndex(m => m.subclass === psMeta.subclass);
    if (ephysIdx >= 0) {{
      Plotly.restyle('ephysPlot', {{ visible: data[0].visible }}, [ephysIdx]);
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("Building Linked Interactive UMAP Viewer")
    print("=" * 60)
    t0 = time.time()

    if not CACHE_H5AD.exists():
        print(f"ERROR: Cache file not found: {CACHE_H5AD}")
        print("Run build_patchseq_in_reference_umap.py first.")
        sys.exit(1)

    expr_obs, ephys_obs = load_data()
    subclass_colors, supertype_colors = load_colors()
    print(f"Loaded {len(subclass_colors)} subclass colors, {len(supertype_colors)} supertype colors")

    print("\nBuilding HTML...")
    html = build_html(expr_obs, ephys_obs, subclass_colors, supertype_colors)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    size_mb = OUTPUT_HTML.stat().st_size / 1024 / 1024
    print(f"\nSaved: {OUTPUT_HTML} ({size_mb:.1f} MB)")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"\nOpen in browser:")
    print(f"  open {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
