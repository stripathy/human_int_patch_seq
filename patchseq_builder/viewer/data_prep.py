"""
data_prep.py -- Load and prepare all data for the interactive UMAP viewer.

Loads expression/ephys UMAP coordinates, builds consensus labels (scANVI
preferred, kNN fallback), attaches SVG trace/morphology paths, and returns
everything the viewer needs in a single dict.

Usage:
    from patchseq_builder.viewer.data_prep import load_viewer_data
    data = load_viewer_data()
"""

import numpy as np
import pandas as pd
import anndata as ad

from patchseq_builder.config import (
    PATCHSEQ_H5AD,
    REFERENCE_COMBINED_H5AD,
    TRACE_SVG_MAP_CSV,
    MORPHOLOGY_SVG_MAP_CSV,
    PROJECT_ROOT,
)
from patchseq_builder.naming import display_subclass
from patchseq_builder.reference.colors import load_allen_colors


def _build_consensus_labels(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Add display_subclass and display_supertype columns.

    Consensus logic: prefer scANVI labels, fall back to kNN labels.
    Normalises scANVI subclass names via ``display_subclass()``.
    """
    # Subclass: scANVI preferred, kNN fallback
    display_sub = (
        obs_df.get("subclass_scANVI", pd.Series(dtype=str))
        .astype(str)
        .copy()
    )
    display_sub = display_sub.replace("nan", np.nan)
    display_sub = display_sub.map(
        lambda x: display_subclass(x) if pd.notna(x) else x
    )
    knn_sub = (
        obs_df.get("knn_subclass", pd.Series(dtype=str))
        .astype(str)
        .replace("nan", np.nan)
    )
    missing = display_sub.isna()
    display_sub.loc[missing] = knn_sub.loc[missing]
    obs_df["display_subclass"] = display_sub

    # Supertype: scANVI preferred, kNN fallback
    display_sup = (
        obs_df.get("supertype_scANVI", pd.Series(dtype=str))
        .astype(str)
        .copy()
    )
    display_sup = display_sup.replace("nan", np.nan)
    knn_sup = (
        obs_df.get("knn_supertype", pd.Series(dtype=str))
        .astype(str)
        .replace("nan", np.nan)
    )
    missing = display_sup.isna()
    display_sup.loc[missing] = knn_sup.loc[missing]
    obs_df["display_supertype"] = display_sup

    return obs_df


def _load_svg_maps() -> tuple[dict, dict]:
    """Load electrophysiology trace and morphology SVG path mappings.

    Returns
    -------
    trace_map : dict
        specimen_id (int) -> {"trace_svg": str, "fi_svg": str}
    morphology_map : dict
        specimen_id (int) -> morphology_svg_path (str)
    """
    # -- Trace SVGs (voltage trace + FI curve) --
    trace_map: dict = {}
    if TRACE_SVG_MAP_CSV.exists():
        svg_df = pd.read_csv(TRACE_SVG_MAP_CSV)
        for _, srow in svg_df.iterrows():
            sid = int(srow["specimen_id"])
            trace_svg = str(srow.get("trace_svg", ""))
            fi_svg = str(srow.get("fi_svg", ""))
            # Convert paths to relative paths under traces/ for HTTP serving
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
            trace_map[sid] = {"trace_svg": trace_svg, "fi_svg": fi_svg}
        print(f"  SVG trace mapping: {len(trace_map)} cells")
    else:
        print(f"  WARNING: SVG map not found at {TRACE_SVG_MAP_CSV}")

    # -- Morphology SVGs --
    morphology_map: dict = {}
    if MORPHOLOGY_SVG_MAP_CSV.exists():
        morph_df = pd.read_csv(MORPHOLOGY_SVG_MAP_CSV)
        for _, mrow in morph_df.iterrows():
            if mrow.get("status") != "ok":
                continue
            mid = int(mrow["specimen_id"])
            morphology_map[mid] = str(mrow.get("svg_path", ""))
        print(f"  Morphology SVG mapping: {len(morphology_map)} cells")
    else:
        print(f"  WARNING: Morphology map not found at {MORPHOLOGY_SVG_MAP_CSV}")

    return trace_map, morphology_map


def _attach_svg_paths(
    obs_df: pd.DataFrame,
    trace_map: dict,
    morphology_map: dict,
) -> pd.DataFrame:
    """Add trace_svg, fi_svg, and morph_svg columns to an obs DataFrame."""
    trace_svgs = []
    fi_svgs = []
    morph_svgs = []

    for _, row in obs_df.iterrows():
        sid = row.get("specimen_id", np.nan)
        try:
            sid_int = int(float(sid))
        except (ValueError, TypeError):
            sid_int = None

        if sid_int and sid_int in trace_map:
            trace_svgs.append(trace_map[sid_int]["trace_svg"])
            fi_svgs.append(trace_map[sid_int]["fi_svg"])
        else:
            trace_svgs.append("")
            fi_svgs.append("")
        morph_svgs.append(morphology_map.get(sid_int, "") if sid_int else "")

    obs_df["trace_svg"] = trace_svgs
    obs_df["fi_svg"] = fi_svgs
    obs_df["morph_svg"] = morph_svgs
    return obs_df


def load_viewer_data() -> dict:
    """Load all data needed for the interactive viewer.

    Returns a dict with:
      - expr_obs: DataFrame of expression UMAP cells (42K reference + 1.1K patchseq)
      - ephys_obs: DataFrame of ephys UMAP cells (~911 patchseq)
      - subclass_colors: dict of subclass -> hex color
      - supertype_colors: dict of supertype -> hex color
      - trace_map: dict of specimen_id -> {trace_svg, fi_svg}
      - morphology_map: dict of specimen_id -> morphology_svg_path
    """
    # -- Load AnnData objects --
    print("Loading cached combined AnnData (expression UMAP)...")
    combined = ad.read_h5ad(str(REFERENCE_COMBINED_H5AD))
    print(f"  {combined.shape[0]:,} cells")

    print("Loading patchseq_combined.h5ad (ephys UMAP)...")
    ps = ad.read_h5ad(str(PATCHSEQ_H5AD))
    print(f"  {ps.shape[0]:,} cells")

    # -- Expression UMAP coordinates --
    expr_umap = combined.obsm["X_umap"]
    expr_obs = combined.obs.copy()
    expr_obs["umap_1"] = expr_umap[:, 0]
    expr_obs["umap_2"] = expr_umap[:, 1]

    # -- Ephys UMAP coordinates (only valid rows) --
    ephys_umap = np.array(ps.obsm["X_umap_ephys"])
    valid_ephys = ~np.isnan(ephys_umap).any(axis=1)
    ps_ephys = ps[valid_ephys].copy()
    ephys_umap_valid = ephys_umap[valid_ephys]

    ephys_obs = ps_ephys.obs.copy()
    ephys_obs["umap_1"] = ephys_umap_valid[:, 0]
    ephys_obs["umap_2"] = ephys_umap_valid[:, 1]

    # -- Morphology UMAP coordinates (only valid rows) --
    if "X_umap_morph" in ps.obsm:
        morph_umap = np.array(ps.obsm["X_umap_morph"])
        valid_morph = ~np.isnan(morph_umap).any(axis=1)
        ps_morph = ps[valid_morph].copy()
        morph_umap_valid = morph_umap[valid_morph]

        morph_obs = ps_morph.obs.copy()
        morph_obs["umap_1"] = morph_umap_valid[:, 0]
        morph_obs["umap_2"] = morph_umap_valid[:, 1]
        print(f"  Morphology UMAP: {len(morph_obs)} cells with valid coordinates")
    else:
        morph_obs = pd.DataFrame()
        print("  WARNING: X_umap_morph not found in h5ad")

    # -- Carry kNN labels from combined into ephys_obs and morph_obs --
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

    if not morph_obs.empty:
        ps_to_combined_morph = {name: f"ps_{name}" for name in morph_obs.index}
        for col in ["knn_subclass", "knn_subclass_conf", "knn_supertype", "knn_supertype_conf"]:
            if col in combined.obs.columns:
                vals = []
                for name in morph_obs.index:
                    cname = ps_to_combined_morph[name]
                    if cname in combined.obs.index:
                        vals.append(combined.obs.loc[cname, col])
                    else:
                        vals.append(np.nan)
                morph_obs[col] = vals

    # -- Carry transcriptomic type columns from patchseq -> expression obs --
    for col in [
        "transcriptomic_type_original", "l1_ttype",
        "subclass_scANVI", "supertype_scANVI",
        "subclass_conf_scANVI", "supertype_conf_scANVI",
    ]:
        if col in ps.obs.columns and col not in expr_obs.columns:
            ps_vals = ps.obs[col]
            expr_obs[col] = np.nan
            for ps_name in ps_vals.index:
                cname = f"ps_{ps_name}"
                if cname in expr_obs.index:
                    expr_obs.loc[cname, col] = ps_vals[ps_name]

    # -- Carry scANVI labels to ephys_obs and morph_obs --
    for target_obs in [ephys_obs] + ([morph_obs] if not morph_obs.empty else []):
        for col in [
            "subclass_scANVI", "supertype_scANVI",
            "subclass_conf_scANVI", "supertype_conf_scANVI",
        ]:
            if col in ps.obs.columns and col not in target_obs.columns:
                vals = []
                for name in target_obs.index:
                    if name in ps.obs.index:
                        vals.append(ps.obs.at[name, col])
                    else:
                        vals.append(np.nan)
                target_obs[col] = vals

    # -- Build consensus labels (scANVI preferred, kNN fallback) --
    expr_obs = _build_consensus_labels(expr_obs)
    ephys_obs = _build_consensus_labels(ephys_obs)
    if not morph_obs.empty:
        morph_obs = _build_consensus_labels(morph_obs)

    n_scanvi_expr = expr_obs.loc[
        expr_obs.get("batch", "") == "patch-seq", "subclass_scANVI"
    ].notna().sum()
    n_scanvi_ephys = ephys_obs["subclass_scANVI"].notna().sum()
    print(f"  Ephys UMAP: {len(ephys_obs)} cells with valid coordinates")
    print(f"  Consensus labels (scANVI->kNN): expr={n_scanvi_expr} scANVI, ephys={n_scanvi_ephys} scANVI")

    # -- Load SVG mappings and attach to obs DataFrames --
    trace_map, morphology_map = _load_svg_maps()
    expr_obs = _attach_svg_paths(expr_obs, trace_map, morphology_map)
    ephys_obs = _attach_svg_paths(ephys_obs, trace_map, morphology_map)
    if not morph_obs.empty:
        morph_obs = _attach_svg_paths(morph_obs, trace_map, morphology_map)

    if "batch" in expr_obs.columns:
        n_with_svg = (expr_obs.loc[expr_obs["batch"] == "patch-seq", "trace_svg"] != "").sum()
        n_with_morph = (expr_obs.loc[expr_obs["batch"] == "patch-seq", "morph_svg"] != "").sum()
    else:
        n_with_svg = (expr_obs["trace_svg"] != "").sum()
        n_with_morph = (expr_obs["morph_svg"] != "").sum()
    print(f"  Patch-seq cells with trace SVGs: {n_with_svg}")
    print(f"  Patch-seq cells with morphology SVGs: {n_with_morph}")

    # -- Load Allen colors --
    subclass_colors, supertype_colors = load_allen_colors()
    print(f"Loaded {len(subclass_colors)} subclass colors, {len(supertype_colors)} supertype colors")

    return {
        "expr_obs": expr_obs,
        "ephys_obs": ephys_obs,
        "morph_obs": morph_obs,
        "subclass_colors": subclass_colors,
        "supertype_colors": supertype_colors,
        "trace_map": trace_map,
        "morphology_map": morphology_map,
    }
