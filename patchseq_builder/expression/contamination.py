"""
contamination.py -- Identify off-target contamination genes in patch-seq data.

Patch-seq cells inevitably pick up transcripts from surrounding non-neuronal
tissue (microglia, astrocytes, oligodendrocytes) and from nearby pyramidal
neurons. These off-target transcripts dilute genuine interneuron marker gene
expression, causing contaminated cells to cluster in the center of the
expression UMAP rather than with their true subclass.

Strategy:
  1. Use the SEA-AD snRNA-seq reference to identify genes that are specific
     to each off-target cell class (Microglia, Astrocyte, Oligo/OPC,
     Glutamatergic) versus GABAergic interneurons.
  2. Build a combined "blacklist" of off-target genes.
  3. Compute per-cell contamination scores (mean expression of each
     off-target gene set).

Usage:
    from patchseq_builder.expression.contamination import (
        build_contamination_blacklist,
        score_contamination,
    )
"""

import numpy as np
import pandas as pd
import scanpy as sc

from patchseq_builder.config import (
    SEAAD_H5AD,
    INTERMEDIATES_DIR,
)


# Default parameters for off-target gene identification
CONTAM_CLASSES = {
    "Microglia": {"class": "Non-neuronal and Non-neural", "subclass": "Microglia-PVM"},
    "Astrocyte": {"class": "Non-neuronal and Non-neural", "subclass": "Astrocyte"},
    "Oligo_OPC": {"class": "Non-neuronal and Non-neural", "subclass": ["Oligodendrocyte", "OPC"]},
    "Glutamatergic": {"class": "Neuronal: Glutamatergic", "subclass": None},  # all excitatory
}
DEFAULT_TOP_N = 200
DEFAULT_LOGFC_MIN = 1.0


def build_contamination_blacklist(
    top_n: int = DEFAULT_TOP_N,
    logfc_min: float = DEFAULT_LOGFC_MIN,
    cache: bool = True,
) -> dict[str, list[str]]:
    """Identify off-target genes for each contamination class using the SEA-AD reference.

    For each contamination class, runs a 1-vs-GABAergic differential expression
    test and takes the top marker genes.

    Parameters
    ----------
    top_n : int
        Maximum number of marker genes per contamination class.
    logfc_min : float
        Minimum log2 fold-change threshold for a gene to be considered.
    cache : bool
        If True, save/load results from intermediates directory.

    Returns
    -------
    dict
        Keys are contamination class names, values are lists of gene names.
        Also includes a "combined" key with all unique off-target genes.
    """
    cache_path = INTERMEDIATES_DIR / "contamination_blacklist.csv"
    if cache and cache_path.exists():
        print(f"  Loading cached contamination blacklist from {cache_path}")
        df = pd.read_csv(cache_path)
        result = {}
        for cls_name in df["contam_class"].unique():
            result[cls_name] = df.loc[df["contam_class"] == cls_name, "gene"].tolist()
        result["combined"] = df["gene"].unique().tolist()
        print(f"  {len(result['combined'])} total off-target genes across {len(result) - 1} classes")
        return result

    print("  Loading SEA-AD reference for DE analysis...")
    ref = sc.read_h5ad(str(SEAAD_H5AD))

    # Subset to GABAergic interneurons (target) and all other classes
    gaba_mask = ref.obs["Class"] == "Neuronal: GABAergic"
    print(f"  Reference: {ref.n_obs:,} cells, {gaba_mask.sum():,} GABAergic")

    result = {}
    records = []

    for cls_name, cls_spec in CONTAM_CLASSES.items():
        print(f"\n  Computing DE: {cls_name} vs GABAergic...")

        # Select cells for this contamination class
        if cls_spec["subclass"] is not None:
            if isinstance(cls_spec["subclass"], list):
                contam_mask = ref.obs["Subclass"].isin(cls_spec["subclass"])
            else:
                contam_mask = ref.obs["Subclass"] == cls_spec["subclass"]
        else:
            contam_mask = ref.obs["Class"] == cls_spec["class"]

        n_contam = contam_mask.sum()
        print(f"    {n_contam:,} {cls_name} cells vs {gaba_mask.sum():,} GABAergic cells")

        # Create a binary grouping for DE
        subset = ref[contam_mask | gaba_mask].copy()
        subset.obs["de_group"] = "GABAergic"
        subset.obs.loc[contam_mask[contam_mask | gaba_mask], "de_group"] = cls_name

        # Normalize for DE (if not already)
        sc.pp.normalize_total(subset, target_sum=1e4)
        sc.pp.log1p(subset)

        # Run DE: contamination class vs GABAergic
        sc.tl.rank_genes_groups(
            subset,
            groupby="de_group",
            groups=[cls_name],
            reference="GABAergic",
            method="wilcoxon",
            n_genes=top_n * 2,  # get more than needed, filter by logfc
        )

        # Extract results
        de_df = sc.get.rank_genes_groups_df(subset, group=cls_name)
        # Filter by logfc and adjusted p-value
        sig = de_df[(de_df["logfoldchanges"] >= logfc_min) & (de_df["pvals_adj"] < 0.05)]
        top_genes = sig.head(top_n)["names"].tolist()

        result[cls_name] = top_genes
        print(f"    {len(top_genes)} genes (logFC >= {logfc_min}, padj < 0.05)")

        for gene in top_genes:
            row = de_df[de_df["names"] == gene].iloc[0]
            records.append({
                "contam_class": cls_name,
                "gene": gene,
                "logfoldchange": row["logfoldchanges"],
                "pval_adj": row["pvals_adj"],
            })

    # Combine all off-target genes
    all_genes = set()
    for genes in result.values():
        all_genes.update(genes)
    result["combined"] = sorted(all_genes)
    print(f"\n  Total unique off-target genes: {len(result['combined'])}")

    # Cache results
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    return result


def score_contamination(
    adata: sc.AnnData,
    blacklist: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute per-cell contamination scores for each off-target class.

    For each contamination class, computes the mean expression (in the
    original log2(FPKM+1) space) of that class's marker genes.

    Parameters
    ----------
    adata : sc.AnnData
        Patch-seq AnnData with X in log2(FPKM+1) space.
    blacklist : dict
        Output of ``build_contamination_blacklist()``.

    Returns
    -------
    pd.DataFrame
        Columns: one per contamination class + "total_contam_score".
        Index matches adata.obs.index.
    """
    scores = pd.DataFrame(index=adata.obs.index)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    for cls_name, genes in blacklist.items():
        if cls_name == "combined":
            continue
        # Find genes present in this adata
        valid_idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not valid_idx:
            scores[f"contam_{cls_name}"] = 0.0
            continue

        col_name = f"contam_{cls_name}"
        scores[col_name] = np.mean(X[:, valid_idx], axis=1)
        n_found = len(valid_idx)
        n_total = len(genes)
        print(f"  {cls_name}: {n_found}/{n_total} genes found, "
              f"mean score = {scores[col_name].mean():.3f}")

    # Total contamination: sum across all classes
    contam_cols = [c for c in scores.columns if c.startswith("contam_")]
    scores["total_contam_score"] = scores[contam_cols].sum(axis=1)

    return scores
