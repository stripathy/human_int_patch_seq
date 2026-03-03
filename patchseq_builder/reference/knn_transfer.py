"""
knn_transfer.py -- kNN label transfer and comparison with scANVI labels.

Transfers subclass and supertype labels from reference (snRNA-seq) cells
to patch-seq cells using k-nearest neighbors in Harmony-corrected PCA space.
Also provides comparison against scANVI-transferred labels.

Usage:
    from patchseq_builder.reference.knn_transfer import knn_label_transfer, compare_knn_vs_scanvi
    knn_labels = knn_label_transfer(combined, k=15)
    confusion_df = compare_knn_vs_scanvi(combined, knn_labels)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

from patchseq_builder.config import KNN_K, PATCHSEQ_H5AD


def knn_label_transfer(combined: sc.AnnData, k: int = KNN_K) -> pd.DataFrame:
    """Transfer subclass/supertype labels from reference to patch-seq cells via kNN.

    For each patch-seq cell, finds the *k* nearest reference (snRNA-seq)
    neighbors in Harmony-corrected PCA space and assigns labels by majority vote.

    Parameters
    ----------
    combined : sc.AnnData
        Combined AnnData from ``integrate_patchseq_with_reference``. Must have
        ``.obsm["X_pca_harmony"]`` and obs columns ``batch``, ``ref_subclass``,
        ``ref_supertype``.
    k : int, optional
        Number of nearest neighbors. Default is ``config.KNN_K`` (15).

    Returns
    -------
    pd.DataFrame
        One row per patch-seq cell, indexed by specimen_id (with ``ps_`` prefix
        stripped). Columns:
        - ``knn_subclass``: majority-vote subclass label
        - ``knn_subclass_conf``: fraction of *k* neighbors with winning subclass
        - ``knn_supertype``: majority-vote supertype label
        - ``knn_supertype_conf``: fraction of *k* neighbors with winning supertype
        - ``knn_mean_dist``: mean Euclidean distance to *k* neighbors

    Notes
    -----
    The function also stores labels directly into ``combined.obs`` for the
    patch-seq cells (in-place modification), so downstream plotting code
    can access them.
    """
    print(f"kNN label transfer (k={k})...")

    is_ref = (combined.obs["batch"] == "snRNA-seq").values
    is_ps = ~is_ref

    pca_harmony = combined.obsm["X_pca_harmony"]
    ref_pca = pca_harmony[is_ref]
    ps_pca = pca_harmony[is_ps]

    ref_subclass = combined.obs.loc[is_ref, "ref_subclass"].values
    ref_supertype = combined.obs.loc[is_ref, "ref_supertype"].values

    print(f"  Reference: {ref_pca.shape[0]:,} cells")
    print(f"  Patch-seq: {ps_pca.shape[0]:,} cells")

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

    # Store results into combined.obs for downstream use
    ps_idx = combined.obs.index[is_ps]
    combined.obs.loc[ps_idx, "knn_subclass"] = knn_subclass
    combined.obs.loc[ps_idx, "knn_subclass_conf"] = knn_subclass_conf
    combined.obs.loc[ps_idx, "knn_supertype"] = knn_supertype
    combined.obs.loc[ps_idx, "knn_supertype_conf"] = knn_supertype_conf
    combined.obs.loc[ps_idx, "knn_mean_dist"] = mean_dist

    # Build result DataFrame
    result = pd.DataFrame({
        "knn_subclass": knn_subclass,
        "knn_subclass_conf": knn_subclass_conf,
        "knn_supertype": knn_supertype,
        "knn_supertype_conf": knn_supertype_conf,
        "knn_mean_dist": mean_dist,
    }, index=[idx.replace("ps_", "") for idx in ps_idx])
    result.index.name = "specimen_id"

    # Summary
    print(f"\n  kNN subclass distribution:")
    subclass_counts = pd.Series(knn_subclass).value_counts()
    for label, count in subclass_counts.items():
        print(f"    {label}: {count}")

    print(f"\n  kNN supertype: {len(set(knn_supertype))} unique types assigned")
    print(f"  Mean subclass confidence: {np.mean(knn_subclass_conf):.3f}")
    print(f"  Mean supertype confidence: {np.mean(knn_supertype_conf):.3f}")

    return result


def compare_knn_vs_scanvi(combined: sc.AnnData, knn_labels: pd.DataFrame) -> pd.DataFrame:
    """Compare kNN-transferred labels vs scANVI labels.

    Loads the original patch-seq h5ad to retrieve scANVI subclass and
    supertype labels, then computes agreement rates and a confusion matrix
    at the supertype level.

    Parameters
    ----------
    combined : sc.AnnData
        Combined AnnData (with kNN labels already stored in obs).
    knn_labels : pd.DataFrame
        Output of ``knn_label_transfer``, indexed by specimen_id.

    Returns
    -------
    pd.DataFrame
        Supertype-level confusion matrix (rows = scANVI, columns = kNN).
        Returns an empty DataFrame if no cells have both kNN and scANVI labels.
    """
    print("Comparing kNN labels vs scANVI labels...")

    # Load original patch-seq data to get scANVI labels
    ps_orig = sc.read_h5ad(str(PATCHSEQ_H5AD), backed="r")
    orig_obs = ps_orig.obs.copy()
    del ps_orig

    # Match kNN labels to scANVI labels via specimen_id
    specimen_ids = knn_labels.index.tolist()

    scanvi_subclass = []
    scanvi_supertype = []
    original_subclass = []

    for sid in specimen_ids:
        if sid in orig_obs.index:
            row = orig_obs.loc[sid]
            sc_sub = row.get("subclass_scANVI", np.nan) if "subclass_scANVI" in orig_obs.columns else np.nan
            sc_sup = row.get("supertype_scANVI", np.nan) if "supertype_scANVI" in orig_obs.columns else np.nan
            orig_sub = row.get("subclass_label", np.nan) if "subclass_label" in orig_obs.columns else np.nan
            scanvi_subclass.append(sc_sub if pd.notna(sc_sub) else np.nan)
            scanvi_supertype.append(sc_sup if pd.notna(sc_sup) else np.nan)
            original_subclass.append(orig_sub if pd.notna(orig_sub) else np.nan)
        else:
            scanvi_subclass.append(np.nan)
            scanvi_supertype.append(np.nan)
            original_subclass.append(np.nan)

    knn_labels = knn_labels.copy()
    knn_labels["scanvi_subclass"] = scanvi_subclass
    knn_labels["scanvi_supertype"] = scanvi_supertype
    knn_labels["original_subclass"] = original_subclass

    # -- Subclass comparison --------------------------------------------------
    has_scanvi_sub = knn_labels["scanvi_subclass"].notna()
    n_scanvi = has_scanvi_sub.sum()
    print(f"\n  Cells with scANVI subclass labels: {n_scanvi}")

    if n_scanvi > 0:
        knn_sub = knn_labels.loc[has_scanvi_sub, "knn_subclass"].astype(str).values
        scanvi_sub = knn_labels.loc[has_scanvi_sub, "scanvi_subclass"].astype(str).values

        # Normalize naming: underscores -> spaces for fair comparison
        knn_sub_norm = np.array([s.replace("_", " ") for s in knn_sub])
        scanvi_sub_norm = np.array([s.replace("_", " ") for s in scanvi_sub])

        subclass_agree = (knn_sub_norm == scanvi_sub_norm).sum()
        subclass_total = len(knn_sub_norm)
        print(f"  Subclass agreement: {subclass_agree}/{subclass_total} "
              f"({100*subclass_agree/subclass_total:.1f}%)")

        # Per-subclass breakdown
        print(f"\n  Per-subclass agreement:")
        for sc_name in sorted(set(scanvi_sub_norm)):
            mask = scanvi_sub_norm == sc_name
            agree = (knn_sub_norm[mask] == scanvi_sub_norm[mask]).sum()
            total = mask.sum()
            print(f"    {sc_name}: {agree}/{total} ({100*agree/total:.1f}%)")

    # -- Supertype comparison -------------------------------------------------
    has_scanvi_sup = knn_labels["scanvi_supertype"].notna()
    n_scanvi_sup = has_scanvi_sup.sum()
    print(f"\n  Cells with scANVI supertype labels: {n_scanvi_sup}")

    cm_df = pd.DataFrame()

    if n_scanvi_sup > 0:
        knn_sup = knn_labels.loc[has_scanvi_sup, "knn_supertype"].astype(str).values
        scanvi_sup = knn_labels.loc[has_scanvi_sup, "scanvi_supertype"].astype(str).values

        supertype_agree = (knn_sup == scanvi_sup).sum()
        supertype_total = len(knn_sup)
        print(f"  Supertype agreement: {supertype_agree}/{supertype_total} "
              f"({100*supertype_agree/supertype_total:.1f}%)")

        # Confusion matrix
        all_sup = sorted(set(list(knn_sup) + list(scanvi_sup)))
        cm = confusion_matrix(scanvi_sup, knn_sup, labels=all_sup)
        cm_df = pd.DataFrame(cm, index=all_sup, columns=all_sup)
        cm_df.index.name = "scANVI_supertype"
        cm_df.columns.name = "kNN_supertype"

    return cm_df
