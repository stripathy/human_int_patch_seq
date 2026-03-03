#!/usr/bin/env python
"""
build_patchseq_in_reference_umap.py — Joint UMAP of patch-seq cells with
the GABAergic interneuron subset of the SEA-AD snRNA-seq reference.

Strategy:
  1. Subset SEA-AD to GABAergic interneurons (41K cells)
  2. Subset to shared genes between reference and patch-seq
  3. Normalize patch-seq FPKM into the same log-count space
  4. Concatenate into one AnnData (42K cells total)
  5. Compute HVGs on GABAergic reference only
  6. PCA → Harmony integration (correcting snRNA-seq vs patch-seq batch) → UMAP
  7. kNN label transfer: assign subclass & supertype to each patch-seq cell
  8. Compare kNN labels with scANVI labels
  9. Plot: snRNA-seq as small dots, patch-seq as large circles, Allen standard colors

Output:
  results/figures/patchseq_in_reference_umap.png
  results/figures/patchseq_in_reference_umap_supertype.png
  results/figures/patchseq_in_reference_umap_modality.png
  results/figures/knn_vs_scanvi_comparison.png
  results/tables/patchseq_knn_labels.csv
  results/tables/knn_vs_scanvi_confusion.csv
"""
import json
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path("/Users/shreejoy/Github/patch_seq_lee")
SEAAD_H5AD = PROJECT_ROOT / "nicole_sea_ad_snrnaseq_reference.h5ad"
PATCHSEQ_H5AD = PROJECT_ROOT / "data" / "patchseq" / "patchseq_combined.h5ad"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Allen Institute standard colors from SCZ_Xenium deploy index
COLORS_JSON = Path("/Users/shreejoy/Github/SCZ_Xenium/output/deploy/index.json")

N_PCS = 30
N_HVGS = 3000
KNN_K = 15  # k for kNN label transfer


def load_allen_colors():
    """Load standard Allen Institute subclass and supertype color maps."""
    with open(COLORS_JSON) as f:
        data = json.load(f)
    return data["subclass_colors"], data["supertype_colors"]


def load_gabaergic_reference(shared_genes):
    """Load SEA-AD, subset to GABAergic interneurons and shared genes."""
    t0 = time.time()
    print("1. Loading SEA-AD reference...")
    ref = sc.read_h5ad(str(SEAAD_H5AD))
    print(f"   {ref.shape[0]:,} cells x {ref.shape[1]:,} genes ({time.time()-t0:.0f}s)")

    # Subset to GABAergic
    print("2. Subsetting to GABAergic interneurons...")
    gaba_mask = ref.obs["Class"] == "Neuronal: GABAergic"
    ref = ref[gaba_mask].copy()
    print(f"   {ref.shape[0]:,} GABAergic cells")

    # Subset to shared genes
    ref = ref[:, shared_genes].copy()
    print(f"   {ref.shape[1]:,} shared genes")

    # Normalize
    print("3. Normalizing reference...")
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    # Add metadata
    ref.obs["modality"] = "snRNA-seq"
    ref.obs["subclass_for_plot"] = ref.obs["Subclass"].values
    ref.obs["supertype_for_plot"] = ref.obs["Supertype"].values

    return ref


def load_patchseq(shared_genes):
    """Load patch-seq, convert to log-count space, subset to shared genes."""
    print("\n4. Loading patch-seq data...")
    ps = sc.read_h5ad(str(PATCHSEQ_H5AD))
    print(f"   {ps.shape[0]:,} cells x {ps.shape[1]:,} genes")

    # Subset to shared genes
    ps = ps[:, shared_genes].copy()

    # Convert log2(FPKM+1) → FPKM → normalize to 10K → log1p
    # This puts patch-seq in the same scale as the normalized reference
    print("5. Converting patch-seq to reference expression space...")
    X_fpkm = np.power(2, ps.X) - 1
    X_fpkm = np.maximum(X_fpkm, 0)
    row_sums = X_fpkm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ps.X = np.log1p(X_fpkm / row_sums * 1e4).astype(np.float32)

    # Add metadata
    ps.obs["modality"] = "patch-seq"

    # Build subclass label for plotting
    if "subclass_label" in ps.obs.columns:
        ps_subclass = ps.obs["subclass_label"].astype(str).copy()
    else:
        ps_subclass = pd.Series(["Unknown"] * len(ps), index=ps.obs_names)

    name_map = {
        "Lamp5/Pax6": "Lamp5",
        "LAMP5/PAX6/Other": "Lamp5",
        "SST": "Sst",
        "PVALB": "Pvalb",
        "VIP": "Vip",
    }
    ps.obs["subclass_for_plot"] = ps_subclass.map(
        lambda x: name_map.get(x, x)
    ).values

    # Supertype label (scANVI — only available for ~365 cells)
    if "supertype_scANVI" in ps.obs.columns:
        ps.obs["supertype_for_plot"] = ps.obs["supertype_scANVI"].astype(str).values
    else:
        ps.obs["supertype_for_plot"] = "Unknown"

    return ps


def concatenate_and_integrate(ref, ps):
    """Concatenate, compute HVGs on reference, PCA, Harmony, UMAP."""

    # ── Concatenate ──────────────────────────────────────────────────
    print("\n6. Concatenating datasets...")
    # Make sure X is not sparse for patch-seq (it's dense float32)
    if issparse(ps.X):
        ps.X = ps.X.toarray()

    # Keep only shared obs columns
    shared_obs = ["modality", "subclass_for_plot", "supertype_for_plot"]
    ref_obs = ref.obs[shared_obs].copy()
    ps_obs = ps.obs[shared_obs].copy()

    # Add dataset label for Harmony
    ref_obs["batch"] = "snRNA-seq"
    ps_obs["batch"] = "patch-seq"

    # Carry over reference Subclass and Supertype for kNN transfer
    ref_obs["ref_subclass"] = ref.obs["Subclass"].values
    ref_obs["ref_supertype"] = ref.obs["Supertype"].values
    ps_obs["ref_subclass"] = np.nan
    ps_obs["ref_supertype"] = np.nan

    # Carry over scANVI labels and specimen_id for patch-seq cells
    for col in ["subclass_scANVI", "supertype_scANVI", "specimen_id",
                "subclass_label", "dataset",
                "transcriptomic_type_original", "l1_ttype"]:
        if col in ps.obs.columns:
            ps_obs[col] = ps.obs[col].values
            ref_obs[col] = np.nan
        else:
            ps_obs[col] = np.nan
            ref_obs[col] = np.nan

    ref_slim = sc.AnnData(
        X=ref.X,
        obs=ref_obs,
        var=ref.var[[]].copy(),
    )
    ref_slim.obs_names = ref.obs_names
    ps_slim = sc.AnnData(
        X=csr_matrix(ps.X) if not issparse(ps.X) else ps.X,
        obs=ps_obs,
        var=ps.var[[]].copy(),
    )
    ps_slim.obs_names = [f"ps_{n}" for n in ps.obs_names]

    combined = sc.concat([ref_slim, ps_slim], join="inner")
    print(f"   Combined: {combined.shape[0]:,} cells x {combined.shape[1]:,} genes")

    # Free memory
    del ref_slim, ps_slim, ref, ps
    gc.collect()

    # ── HVGs on reference cells only ─────────────────────────────────
    print("\n7. Computing HVGs on GABAergic reference subset...")
    ref_mask = combined.obs["batch"] == "snRNA-seq"
    ref_subset = combined[ref_mask].copy()
    sc.pp.highly_variable_genes(ref_subset, n_top_genes=N_HVGS, flavor="seurat")
    hvg_genes = ref_subset.var_names[ref_subset.var["highly_variable"]].tolist()
    combined.var["highly_variable"] = combined.var_names.isin(hvg_genes)
    print(f"   {len(hvg_genes)} HVGs from GABAergic reference")
    del ref_subset
    gc.collect()

    # ── Scale + PCA ──────────────────────────────────────────────────
    print("\n8. Scaling and PCA...")
    combined_hvg = combined[:, hvg_genes].copy()

    # Densify for scaling (3K genes is manageable)
    if issparse(combined_hvg.X):
        combined_hvg.X = combined_hvg.X.toarray()

    sc.pp.scale(combined_hvg, max_value=10)
    sc.tl.pca(combined_hvg, n_comps=N_PCS)

    # Copy PCA back
    combined.obsm["X_pca"] = combined_hvg.obsm["X_pca"].copy()
    print(f"   PCA: {combined.obsm['X_pca'].shape}")

    del combined_hvg
    gc.collect()

    # ── Harmony ──────────────────────────────────────────────────────
    print("\n9. Running Harmony integration...")
    ho = hm.run_harmony(
        combined.obsm["X_pca"],
        combined.obs,
        "batch",
        max_iter_harmony=20,
    )
    Z = ho.Z_corr
    # Handle both numpy and torch outputs; ensure shape is (n_cells, n_pcs)
    if hasattr(Z, "numpy"):
        Z = Z.numpy()
    Z = np.asarray(Z, dtype=np.float32)
    if Z.shape[0] == N_PCS and Z.shape[1] != N_PCS:
        Z = Z.T
    # If Z is 1-D (edge case), reshape
    if Z.ndim == 1:
        Z = Z.reshape(-1, N_PCS)
    combined.obsm["X_pca_harmony"] = Z
    print(f"   Harmony corrected PCA: {combined.obsm['X_pca_harmony'].shape}")

    # ── Neighbors + UMAP on Harmony-corrected space ──────────────────
    print("\n10. Computing UMAP on Harmony space...")
    sc.pp.neighbors(combined, use_rep="X_pca_harmony", n_neighbors=15, random_state=42)
    sc.tl.umap(combined, random_state=42)
    print(f"   UMAP: {combined.obsm['X_umap'].shape}")

    return combined


def knn_label_transfer(combined, k=KNN_K):
    """
    Transfer subclass and supertype labels from reference to patch-seq cells
    using kNN in Harmony-corrected PCA space.

    For each patch-seq cell, finds k nearest reference (snRNA-seq) neighbors
    and assigns the majority-vote label.
    """
    print(f"\n11. kNN label transfer (k={k})...")

    is_ref = (combined.obs["batch"] == "snRNA-seq").values
    is_ps = ~is_ref

    pca_harmony = combined.obsm["X_pca_harmony"]

    ref_pca = pca_harmony[is_ref]
    ps_pca = pca_harmony[is_ps]

    ref_subclass = combined.obs.loc[is_ref, "ref_subclass"].values
    ref_supertype = combined.obs.loc[is_ref, "ref_supertype"].values

    print(f"   Reference: {ref_pca.shape[0]:,} cells")
    print(f"   Patch-seq: {ps_pca.shape[0]:,} cells")

    # Fit kNN on reference
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(ref_pca)

    # Find neighbors for each patch-seq cell
    distances, indices = nn.kneighbors(ps_pca)

    # Majority vote for subclass
    knn_subclass = []
    knn_subclass_conf = []
    for i in range(len(ps_pca)):
        neighbor_labels = ref_subclass[indices[i]]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        winner_idx = np.argmax(counts)
        knn_subclass.append(labels[winner_idx])
        knn_subclass_conf.append(counts[winner_idx] / k)

    # Majority vote for supertype
    knn_supertype = []
    knn_supertype_conf = []
    for i in range(len(ps_pca)):
        neighbor_labels = ref_supertype[indices[i]]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        winner_idx = np.argmax(counts)
        knn_supertype.append(labels[winner_idx])
        knn_supertype_conf.append(counts[winner_idx] / k)

    # Mean distance to k neighbors (quality metric)
    mean_dist = distances.mean(axis=1)

    # Store results
    ps_idx = combined.obs.index[is_ps]
    combined.obs.loc[ps_idx, "knn_subclass"] = knn_subclass
    combined.obs.loc[ps_idx, "knn_subclass_conf"] = knn_subclass_conf
    combined.obs.loc[ps_idx, "knn_supertype"] = knn_supertype
    combined.obs.loc[ps_idx, "knn_supertype_conf"] = knn_supertype_conf
    combined.obs.loc[ps_idx, "knn_mean_dist"] = mean_dist

    # Summary
    print(f"\n   kNN subclass distribution:")
    subclass_counts = pd.Series(knn_subclass).value_counts()
    for label, count in subclass_counts.items():
        print(f"     {label}: {count}")

    print(f"\n   kNN supertype: {len(set(knn_supertype))} unique types assigned")
    print(f"   Mean subclass confidence: {np.mean(knn_subclass_conf):.3f}")
    print(f"   Mean supertype confidence: {np.mean(knn_supertype_conf):.3f}")

    return combined


def compare_knn_vs_scanvi(combined):
    """
    Compare kNN-transferred labels with scANVI labels for cells that have both.
    Returns a summary DataFrame.
    """
    print("\n12. Comparing kNN labels vs scANVI labels...")

    is_ps = (combined.obs["batch"] == "patch-seq").values
    ps_obs = combined.obs.loc[is_ps].copy()

    # Load original patch-seq data to get scANVI labels
    ps_orig = sc.read_h5ad(str(PATCHSEQ_H5AD), backed="r")

    # Map combined obs_names (prefixed with ps_) back to original
    ps_specimen_ids = [n.replace("ps_", "") for n in ps_obs.index]

    # Build mapping from original obs
    orig_obs = ps_orig.obs.copy()

    # Get scANVI labels for these cells
    scanvi_subclass = []
    scanvi_supertype = []
    for sid in ps_specimen_ids:
        if sid in orig_obs.index:
            sc_sub = orig_obs.loc[sid, "subclass_scANVI"] if "subclass_scANVI" in orig_obs.columns else np.nan
            sc_sup = orig_obs.loc[sid, "supertype_scANVI"] if "supertype_scANVI" in orig_obs.columns else np.nan
            scanvi_subclass.append(sc_sub if pd.notna(sc_sub) else np.nan)
            scanvi_supertype.append(sc_sup if pd.notna(sc_sup) else np.nan)
        else:
            scanvi_subclass.append(np.nan)
            scanvi_supertype.append(np.nan)

    ps_obs["scanvi_subclass"] = scanvi_subclass
    ps_obs["scanvi_supertype"] = scanvi_supertype

    # Also get original subclass_label
    orig_subclass = []
    for sid in ps_specimen_ids:
        if sid in orig_obs.index:
            orig_subclass.append(orig_obs.loc[sid, "subclass_label"] if "subclass_label" in orig_obs.columns else np.nan)
        else:
            orig_subclass.append(np.nan)
    ps_obs["original_subclass"] = orig_subclass

    del ps_orig

    # ── Subclass comparison ──────────────────────────────────────────
    has_scanvi_sub = ps_obs["scanvi_subclass"].notna()
    n_scanvi = has_scanvi_sub.sum()
    print(f"\n   Cells with scANVI subclass labels: {n_scanvi}")

    if n_scanvi > 0:
        # Harmonize Lamp5_Lhx6 → Lamp5 Lhx6 for fair comparison
        knn_sub = ps_obs.loc[has_scanvi_sub, "knn_subclass"].astype(str).values
        scanvi_sub = ps_obs.loc[has_scanvi_sub, "scanvi_subclass"].astype(str).values

        # Normalize naming
        scanvi_sub_norm = np.array([s.replace("_", " ") for s in scanvi_sub])
        knn_sub_norm = np.array([s.replace("_", " ") for s in knn_sub])

        subclass_agree = (knn_sub_norm == scanvi_sub_norm).sum()
        subclass_total = len(knn_sub_norm)
        print(f"   Subclass agreement: {subclass_agree}/{subclass_total} ({100*subclass_agree/subclass_total:.1f}%)")

        # Per-subclass breakdown
        print(f"\n   Per-subclass agreement:")
        for sc_name in sorted(set(scanvi_sub_norm)):
            mask = scanvi_sub_norm == sc_name
            agree = (knn_sub_norm[mask] == scanvi_sub_norm[mask]).sum()
            total = mask.sum()
            print(f"     {sc_name}: {agree}/{total} ({100*agree/total:.1f}%)")

    # ── Supertype comparison ─────────────────────────────────────────
    has_scanvi_sup = ps_obs["scanvi_supertype"].notna()
    n_scanvi_sup = has_scanvi_sup.sum()
    print(f"\n   Cells with scANVI supertype labels: {n_scanvi_sup}")

    if n_scanvi_sup > 0:
        knn_sup = ps_obs.loc[has_scanvi_sup, "knn_supertype"].astype(str).values
        scanvi_sup = ps_obs.loc[has_scanvi_sup, "scanvi_supertype"].astype(str).values

        supertype_agree = (knn_sup == scanvi_sup).sum()
        supertype_total = len(knn_sup)
        print(f"   Supertype agreement: {supertype_agree}/{supertype_total} ({100*supertype_agree/supertype_total:.1f}%)")

        # Per-supertype concordance
        all_sup = sorted(set(list(knn_sup) + list(scanvi_sup)))
        cm = confusion_matrix(scanvi_sup, knn_sup, labels=all_sup)
        cm_df = pd.DataFrame(cm, index=all_sup, columns=all_sup)
        cm_df.index.name = "scANVI_supertype"
        cm_df.columns.name = "kNN_supertype"

        # Save confusion matrix
        cm_path = str(TABLES_DIR / "knn_vs_scanvi_confusion.csv")
        cm_df.to_csv(cm_path)
        print(f"   Saved confusion matrix: {cm_path}")

    # ── Save full label table ────────────────────────────────────────
    label_df = ps_obs[["knn_subclass", "knn_subclass_conf",
                       "knn_supertype", "knn_supertype_conf",
                       "knn_mean_dist",
                       "scanvi_subclass", "scanvi_supertype",
                       "original_subclass",
                       "subclass_for_plot"]].copy()
    # Remove ps_ prefix from index
    label_df.index = [idx.replace("ps_", "") for idx in label_df.index]
    label_df.index.name = "specimen_id"

    label_path = str(TABLES_DIR / "patchseq_knn_labels.csv")
    label_df.to_csv(label_path)
    print(f"   Saved kNN labels: {label_path}")
    print(f"   {len(label_df)} cells total")

    return ps_obs, label_df


def plot_knn_vs_scanvi(label_df, supertype_colors):
    """Generate comparison figures for kNN vs scANVI labels."""
    print("\n13. Plotting kNN vs scANVI comparison...")

    has_scanvi = label_df["scanvi_subclass"].notna()
    if has_scanvi.sum() == 0:
        print("   No cells with scANVI labels to compare — skipping.")
        return

    df = label_df.loc[has_scanvi].copy()

    # ── Figure: Confusion matrix heatmap (subclass level) ────────────
    knn_sub = df["knn_subclass"].astype(str).str.replace("_", " ").values
    scanvi_sub = df["scanvi_subclass"].astype(str).str.replace("_", " ").values

    all_sub = sorted(set(list(knn_sub) + list(scanvi_sub)))
    cm_sub = confusion_matrix(scanvi_sub, knn_sub, labels=all_sub)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Subclass confusion matrix
    ax = axes[0]
    im = ax.imshow(cm_sub, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(all_sub)))
    ax.set_xticklabels(all_sub, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(all_sub)))
    ax.set_yticklabels(all_sub, fontsize=12)
    ax.set_xlabel("kNN Subclass", fontsize=14)
    ax.set_ylabel("scANVI Subclass", fontsize=14)
    ax.set_title("Subclass: kNN vs scANVI", fontsize=16)

    # Annotate cells
    for i in range(len(all_sub)):
        for j in range(len(all_sub)):
            if cm_sub[i, j] > 0:
                ax.text(j, i, str(cm_sub[i, j]),
                        ha="center", va="center", fontsize=11,
                        color="white" if cm_sub[i, j] > cm_sub.max() / 2 else "black")

    agree = (knn_sub == scanvi_sub).sum()
    total = len(knn_sub)
    ax.text(0.02, 0.98, f"Agreement: {agree}/{total} ({100*agree/total:.1f}%)",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Supertype confusion matrix (only show supertypes with ≥3 cells)
    ax = axes[1]
    knn_sup = df["knn_supertype"].astype(str).values
    scanvi_sup = df["scanvi_supertype"].astype(str).values

    all_sup = sorted(set(list(knn_sup) + list(scanvi_sup)))

    # Filter to supertypes with ≥2 cells in either set
    sup_counts_scanvi = pd.Series(scanvi_sup).value_counts()
    sup_counts_knn = pd.Series(knn_sup).value_counts()
    keep_sup = sorted(set(
        sup_counts_scanvi[sup_counts_scanvi >= 2].index.tolist() +
        sup_counts_knn[sup_counts_knn >= 2].index.tolist()
    ))

    if len(keep_sup) > 0:
        cm_sup = confusion_matrix(scanvi_sup, knn_sup, labels=keep_sup)
        im2 = ax.imshow(cm_sup, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(keep_sup)))
        ax.set_xticklabels(keep_sup, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(keep_sup)))
        ax.set_yticklabels(keep_sup, fontsize=9)
        ax.set_xlabel("kNN Supertype", fontsize=14)
        ax.set_ylabel("scANVI Supertype", fontsize=14)
        ax.set_title("Supertype: kNN vs scANVI", fontsize=16)

        for i in range(len(keep_sup)):
            for j in range(len(keep_sup)):
                if cm_sup[i, j] > 0:
                    ax.text(j, i, str(cm_sup[i, j]),
                            ha="center", va="center", fontsize=8,
                            color="white" if cm_sup[i, j] > cm_sup.max() / 2 else "black")

        agree_sup = (knn_sup == scanvi_sup).sum()
        ax.text(0.02, 0.98, f"Agreement: {agree_sup}/{total} ({100*agree_sup/total:.1f}%)",
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("kNN Label Transfer vs scANVI (cells with both labels)",
                 fontsize=18, y=1.02)
    plt.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "knn_vs_scanvi_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"   Saved: knn_vs_scanvi_comparison.png")


def plot_results(combined, subclass_colors, supertype_colors):
    """Generate UMAP plots with Allen standard colors."""
    print("\n14. Generating UMAP plots with Allen colors...")

    umap = combined.obsm["X_umap"]
    is_ref = (combined.obs["batch"] == "snRNA-seq").values
    is_ps = ~is_ref

    # Use kNN-transferred labels for patch-seq cells if available
    subclass = combined.obs["subclass_for_plot"].values.copy()
    if "knn_subclass" in combined.obs.columns:
        ps_idx = combined.obs.index[is_ps]
        knn_sub = combined.obs.loc[ps_idx, "knn_subclass"].values
        # Use kNN subclass for patch-seq cells
        subclass_arr = subclass.copy()
        subclass_arr[is_ps] = knn_sub
    else:
        subclass_arr = subclass

    # Supertype: use kNN for patch-seq
    if "knn_supertype" in combined.obs.columns:
        supertype = combined.obs["supertype_for_plot"].values.copy()
        ps_idx = combined.obs.index[is_ps]
        knn_sup = combined.obs.loc[ps_idx, "knn_supertype"].values
        supertype_arr = supertype.copy()
        supertype_arr[is_ps] = knn_sup
    else:
        supertype_arr = combined.obs["supertype_for_plot"].values

    # ── Figure 1: Subclass coloring ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    # Reference as small dots
    ref_subclasses = sorted(set(subclass_arr[is_ref]))
    for sc_name in ref_subclasses:
        mask = is_ref & (subclass_arr == sc_name)
        color = subclass_colors.get(sc_name, "#CCCCCC")
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color, s=0.5, alpha=0.1, rasterized=True,
        )

    # Patch-seq as large circles with black edge
    ps_subclasses = sorted(set(subclass_arr[is_ps]))
    for sc_name in ps_subclasses:
        mask = is_ps & (subclass_arr == sc_name)
        if mask.sum() == 0:
            continue
        color = subclass_colors.get(sc_name, "#333333")
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color, s=45, alpha=0.85,
            edgecolors="black", linewidths=0.3,
            label=f"{sc_name} (n={mask.sum()})",
            zorder=5,
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title(
        "Patch-Seq in GABAergic SEA-AD Reference\n(Harmony integrated, kNN subclass labels, Allen colors)",
        fontsize=16,
    )
    ax.legend(
        fontsize=11, markerscale=1.5, frameon=True,
        title="Patch-seq subclass (kNN)", title_fontsize=13,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "patchseq_in_reference_umap.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"   Saved: patchseq_in_reference_umap.png")

    # ── Figure 2: Supertype coloring (patch-seq colored by kNN supertype) ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))

    # Reference background (light, by subclass)
    for sc_name in ref_subclasses:
        mask = is_ref & (subclass_arr == sc_name)
        color = subclass_colors.get(sc_name, "#CCCCCC")
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color, s=0.5, alpha=0.08, rasterized=True,
        )

    # Patch-seq colored by supertype using Allen colors
    ps_supertypes = supertype_arr[is_ps]
    unique_st = sorted(set(ps_supertypes) - {"nan", "Unknown", "None"})

    for st in unique_st:
        mask = is_ps & (supertype_arr == st)
        if mask.sum() == 0:
            continue
        color = supertype_colors.get(st, "#333333")
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color, s=45, alpha=0.85,
            edgecolors="black", linewidths=0.3,
            label=f"{st} (n={mask.sum()})",
            zorder=5,
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title(
        "Patch-Seq Supertypes in GABAergic Reference\n(Harmony integrated, kNN supertype labels, Allen colors)",
        fontsize=16,
    )
    ax.legend(
        fontsize=7, markerscale=1.2, frameon=True, ncol=2,
        title="Patch-seq supertype (kNN)", title_fontsize=11,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "patchseq_in_reference_umap_supertype.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"   Saved: patchseq_in_reference_umap_supertype.png")

    # ── Figure 3: Dataset of origin ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.scatter(
        umap[is_ref, 0], umap[is_ref, 1],
        c="#B0B0B0", s=0.5, alpha=0.1, rasterized=True,
        label=f"snRNA-seq (n={is_ref.sum():,})",
    )
    ax.scatter(
        umap[is_ps, 0], umap[is_ps, 1],
        c="#E74C3C", s=35, alpha=0.8,
        edgecolors="black", linewidths=0.3,
        label=f"patch-seq (n={is_ps.sum():,})",
        zorder=5,
    )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.set_title("Modality: snRNA-seq vs Patch-Seq (Harmony Integrated)", fontsize=18)
    ax.legend(fontsize=13, markerscale=2, frameon=True, loc="best")
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "patchseq_in_reference_umap_modality.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"   Saved: patchseq_in_reference_umap_modality.png")

    # ── Figure 4: kNN confidence ─────────────────────────────────────
    if "knn_subclass_conf" in combined.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ps_obs = combined.obs.loc[is_ps]

        # Subclass confidence histogram
        ax = axes[0]
        conf = ps_obs["knn_subclass_conf"].values.astype(float)
        ax.hist(conf, bins=30, color="#4ECDC4", edgecolor="black", alpha=0.8)
        ax.axvline(np.median(conf), color="red", linestyle="--", linewidth=2,
                   label=f"Median: {np.median(conf):.2f}")
        ax.set_xlabel("kNN Subclass Confidence", fontsize=14)
        ax.set_ylabel("Number of cells", fontsize=14)
        ax.set_title("kNN Subclass Confidence Distribution", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)

        # Supertype confidence histogram
        ax = axes[1]
        conf = ps_obs["knn_supertype_conf"].values.astype(float)
        ax.hist(conf, bins=30, color="#FF6B6B", edgecolor="black", alpha=0.8)
        ax.axvline(np.median(conf), color="red", linestyle="--", linewidth=2,
                   label=f"Median: {np.median(conf):.2f}")
        ax.set_xlabel("kNN Supertype Confidence", fontsize=14)
        ax.set_ylabel("Number of cells", fontsize=14)
        ax.set_title("kNN Supertype Confidence Distribution", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)

        plt.suptitle(f"kNN Label Transfer Confidence (k={KNN_K}, n={is_ps.sum()} cells)",
                     fontsize=18, y=1.02)
        plt.tight_layout()
        fig.savefig(
            str(FIGURES_DIR / "knn_confidence_distributions.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"   Saved: knn_confidence_distributions.png")


def main():
    print("=" * 60)
    print("Patch-Seq in GABAergic SEA-AD Reference (Harmony + kNN)")
    print("=" * 60)

    t0 = time.time()

    # Load Allen colors
    subclass_colors, supertype_colors = load_allen_colors()
    print(f"Loaded {len(subclass_colors)} subclass colors, {len(supertype_colors)} supertype colors")

    # Find shared genes first (without loading full matrices)
    print("\n0. Finding shared genes...")
    ref_tmp = sc.read_h5ad(str(SEAAD_H5AD), backed="r")
    ps_tmp = sc.read_h5ad(str(PATCHSEQ_H5AD), backed="r")
    shared_genes = sorted(set(ref_tmp.var_names) & set(ps_tmp.var_names))
    print(f"   {len(shared_genes)} shared genes")
    del ref_tmp, ps_tmp

    ref = load_gabaergic_reference(shared_genes)
    ps = load_patchseq(shared_genes)

    combined = concatenate_and_integrate(ref, ps)

    # kNN label transfer
    combined = knn_label_transfer(combined)

    # Compare with scANVI
    ps_obs, label_df = compare_knn_vs_scanvi(combined)

    # Plot comparison
    plot_knn_vs_scanvi(label_df, supertype_colors)

    # Generate UMAP plots with Allen colors
    plot_results(combined, subclass_colors, supertype_colors)

    # Save combined AnnData cache for interactive viewer
    # Drop expression matrix to keep file small — only need obs + obsm
    INTERMEDIATES_DIR = PROJECT_ROOT / "results" / "intermediates"
    INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = INTERMEDIATES_DIR / "patchseq_reference_combined.h5ad"
    print(f"\n15. Saving combined AnnData cache...")
    combined_slim = sc.AnnData(
        obs=combined.obs.copy(),
        obsm={"X_umap": combined.obsm["X_umap"].copy()},
    )
    combined_slim.write_h5ad(str(cache_path))
    print(f"   Saved: {cache_path} ({combined_slim.shape[0]:,} cells)")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
