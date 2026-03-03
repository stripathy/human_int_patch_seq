"""
build_html.py -- Build the interactive UMAP viewer HTML file.

Orchestrates data loading (via data_prep), Plotly trace construction,
and Jinja2 template rendering to produce a self-contained HTML viewer
with linked expression and electrophysiology UMAPs.

Usage:
    from patchseq_builder.viewer.build_html import build_viewer_html
    build_viewer_html()                   # default output path
    build_viewer_html("/tmp/viewer.html") # custom path
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from patchseq_builder.config import VIEWER_HTML
from patchseq_builder.viewer.data_prep import load_viewer_data

# Template directory is the same directory as this file
_TEMPLATE_DIR = Path(__file__).parent


# ============================================================================
# Hover text builder
# ============================================================================

def _build_hover_text_patchseq(row, source="expression"):
    """Build rich hover text for a patch-seq cell.

    Parameters
    ----------
    row : pd.Series
        A row from expr_obs or ephys_obs.
    source : str
        "expression" or "ephys" -- controls which ephys features to show.
    """
    parts = ["<b>PATCH-SEQ</b>"]

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

    # Ephys features -- only show in expression UMAP tooltip (not ephys UMAP)
    if source == "expression":
        ephys_cols = {
            "sag": "Sag", "tau": "Tau (ms)",
            "input_resistance": "Rin (MOhm)",
            "rheobase_i": "Rheobase (pA)", "fi_fit_slope": "FI slope",
            "v_baseline": "Vm (mV)",
        }
        ephys_parts = []
        for col, label in ephys_cols.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                ephys_parts.append(f"{label}: {float(val):.1f}")

        if ephys_parts:
            parts.append("---")
            parts.extend(ephys_parts)

    return "<br>".join(parts)


# ============================================================================
# Plotly trace construction
# ============================================================================

def _build_plotly_traces(expr_obs, ephys_obs, subclass_colors, supertype_colors):
    """Build all Plotly trace dicts and metadata for both UMAPs.

    Returns
    -------
    dict with keys:
        ref_trace, ref_subclass_colors, ref_supertype_colors,
        expr_ps_traces, expr_ps_meta, ephys_traces, ephys_meta,
        n_ref, n_ps, n_ephys
    """
    is_ref = expr_obs["batch"] == "snRNA-seq"
    is_ps = expr_obs["batch"] == "patch-seq"

    # ── Expression UMAP: reference trace ──────────────────────────
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

    # Per-cell colors for reference (for subclass/supertype coloring modes)
    ref_supertype_colors_list = [
        supertype_colors.get(st, "#D0D0D0")
        for st in ref_data["ref_supertype"].values
    ]
    ref_subclass_colors_list = [
        subclass_colors.get(sc, "#D0D0D0")
        for sc in ref_data["ref_subclass"].values
    ]

    # ── Expression UMAP: patch-seq traces (one per display_subclass) ──
    ps_expr = expr_obs.loc[is_ps].copy()
    expr_ps_traces = []
    expr_ps_meta = []
    ps_in_ephys = set(ephys_obs.index)

    for sc_name in sorted(ps_expr["display_subclass"].dropna().unique()):
        mask = ps_expr["display_subclass"] == sc_name
        subset = ps_expr.loc[mask]
        color = subclass_colors.get(sc_name, "#333333")

        hover_texts = []
        custom_data_rows = []
        supertype_list = []
        has_ephys_list = []

        for idx, row in subset.iterrows():
            hover_texts.append(_build_hover_text_patchseq(row))
            cid = idx.replace("ps_", "") if idx.startswith("ps_") else idx
            supertype_list.append(row.get("display_supertype", "Unknown"))
            has_ephys_list.append(cid in ps_in_ephys)
            # customdata: [cell_id, trace_svg, fi_svg, specimen_id,
            #              subclass, dataset, supertype, morph_svg]
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

    # ── Ephys UMAP traces (one per display_subclass) ─────────────
    ephys_traces = []
    ephys_meta = []

    for sc_name in sorted(ephys_obs["display_subclass"].dropna().unique()):
        mask = ephys_obs["display_subclass"] == sc_name
        subset = ephys_obs.loc[mask]
        color = subclass_colors.get(sc_name, "#333333")

        hover_texts = []
        custom_data_rows = []
        supertype_list = []

        for idx, row in subset.iterrows():
            hover_texts.append(_build_hover_text_patchseq(row, source="ephys"))
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

    n_ref = int(is_ref.sum())
    n_ps = int(is_ps.sum())
    n_ephys = len(ephys_obs)

    return {
        "ref_trace": ref_trace,
        "ref_subclass_colors": ref_subclass_colors_list,
        "ref_supertype_colors": ref_supertype_colors_list,
        "expr_ps_traces": expr_ps_traces,
        "expr_ps_meta": expr_ps_meta,
        "ephys_traces": ephys_traces,
        "ephys_meta": ephys_meta,
        "n_ref": n_ref,
        "n_ps": n_ps,
        "n_ephys": n_ephys,
    }


# ============================================================================
# Main entry point
# ============================================================================

def build_viewer_html(output_path=None):
    """Build the interactive UMAP viewer HTML.

    1. Loads data via data_prep.load_viewer_data()
    2. Prepares Plotly trace data structures
    3. Renders Jinja2 template with data
    4. Saves HTML file

    Parameters
    ----------
    output_path : str or Path, optional
        Where to write the HTML file. Defaults to ``config.VIEWER_HTML``.
    """
    if output_path is None:
        output_path = VIEWER_HTML
    output_path = Path(output_path)

    print("=" * 60)
    print("Building Linked Interactive UMAP Viewer")
    print("=" * 60)
    t0 = time.time()

    # 1. Load all data
    data = load_viewer_data()

    # 2. Build Plotly traces
    print("\nBuilding Plotly traces...")
    traces = _build_plotly_traces(
        data["expr_obs"],
        data["ephys_obs"],
        data["subclass_colors"],
        data["supertype_colors"],
    )

    # 3. Render Jinja2 template
    print("Rendering HTML template...")
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        # Keep Jinja2 defaults for {{ }} delimiters.
        # The template uses {% raw %} blocks to protect JavaScript curly braces.
    )
    template = env.get_template("template.html")

    # Format counts with commas for display
    n_ref_fmt = f"{traces['n_ref']:,}"
    n_ps_fmt = f"{traces['n_ps']:,}"
    n_ephys_fmt = f"{traces['n_ephys']:,}"

    html = template.render(
        # Data payloads (pre-serialised to JSON for injection into <script>)
        ref_trace=json.dumps(traces["ref_trace"]),
        ref_subclass_colors=json.dumps(traces["ref_subclass_colors"]),
        ref_supertype_colors=json.dumps(traces["ref_supertype_colors"]),
        expr_ps_traces=json.dumps(traces["expr_ps_traces"]),
        expr_ps_meta=json.dumps(traces["expr_ps_meta"]),
        ephys_traces=json.dumps(traces["ephys_traces"]),
        ephys_meta=json.dumps(traces["ephys_meta"]),
        subclass_colors=json.dumps(data["subclass_colors"]),
        supertype_colors=json.dumps(data["supertype_colors"]),
        # Formatted counts for display in header and JS
        n_ref=n_ref_fmt,
        n_ps=n_ps_fmt,
        n_ephys=n_ephys_fmt,
    )

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"\nOpen in browser:")
    print(f"  open {output_path}")

    return str(output_path)


if __name__ == "__main__":
    build_viewer_html()
