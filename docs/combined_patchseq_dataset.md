# Combined Human Interneuron Patch-Seq Dataset

## Overview

This document describes a harmonized dataset combining two human patch-seq studies into a single resource of **1,139 unique interneurons** with simultaneous transcriptomic and electrophysiological measurements.

**Source datasets:**
- **Lee & Dalley (LeeDalley)** — 779 human MTG interneurons recorded across cortical layers 1-6, with multi-subclass coverage (Pvalb, Sst, Vip, Lamp5/Pax6). Includes both acute and cultured slices from epilepsy and tumor resection tissue.
- **Lee et al. Science 2023 (L1)** — 419 human Layer 1 interneurons (Lee et al., [Science 2023](https://www.science.org/doi/10.1126/science.adf0805)). Focused on superficial interneuron types: Lamp5, Sncg, Pax6, Vip subclasses. All acute recordings from neurosurgical tissue.

The two datasets share 43 cells in common and 72 overlapping donors.

---

## Dataset Summary

| | LeeDalley-only | L1-only | Overlap | **Total** |
|---|---|---|---|---|
| Cells | 736 | 376 | 43 | **1,155** |
| With expression data | 735 | 361 | 43 | **1,139** |
| With electrophysiology | 642 | 211 | 38 | **891** |
| With scANVI supertype labels | 0 | 349 | 34 | **383** |
| With ephys + scANVI | 0 | 196 | 34 | **230** |
| Unique donors | — | — | — | **221** |
| Brain regions | — | — | — | **24** |

### Subclass distribution (all cells with expression)

| Subclass | Count | Source |
|---|---|---|
| Pvalb | 327 | LeeDalley only |
| Sst | 247 | LeeDalley only |
| Lamp5 | 161 | Mostly L1 |
| Vip | 131 | Both |
| Sncg | 107 | L1 only |
| Lamp5/Pax6 (unresolved) | 88 | LeeDalley cells without scANVI |
| Pax6 | 65 | L1 only |
| L4 IT | 6 | L1 (likely misclassified) |

The LeeDalley dataset provides the vast majority of Pvalb and Sst cells, while the L1 dataset adds Lamp5, Sncg, and Pax6 subtypes that are rare or absent in the LeeDalley data.

---

## Output Files

All files are in `data/patchseq/`:

### CSV Tables

| File | Shape | Description |
|---|---|---|
| `patchseq_metadata_joined.csv` | 1,155 x 34 | Full harmonized metadata for every cell in either dataset |
| `patchseq_ephys_joined.csv` | 929 x 125 | All electrophysiology features (82 shared + 11 LD-only + 30 L1-only + dataset column + specimen_id) |
| `patchseq_combined.csv` | 1,155 x 50 | Metadata + 16 key ephys features + scANVI labels in one table |

### H5AD (AnnData)

| File | Shape | Description |
|---|---|---|
| `patchseq_combined.h5ad` | 1,139 x 50,281 | Gene expression (log2(FPKM+1)) with metadata in `.obs`, UMAP embeddings in `.obsm` |

The h5ad contains:
- **X**: log2(FPKM+1) expression, 50,281 genes
- **obs**: cell metadata (specimen_id, cell_name, donor, brain_region, cortical_layer, dataset, subclass_label, supertype_scANVI, sag, tau, input_resistance, etc.)
- **obsm["X_umap"]**: UMAP from gene expression (3,000 HVGs, 30 PCs)
- **obsm["X_umap_ephys"]**: UMAP from 6 electrophysiology features (874 cells; NaN for cells without ephys)
- **obsm["X_pca"]**: PCA coordinates (30 components)
- **var**: gene names as index

---

## Data Harmonization Details

### Expression

The LeeDalley dataset stores expression as **log2(FPKM+1)**, while the L1 dataset stores **raw FPKM**. We verified that for the 43 overlap cells, `log2(L1_FPKM + 1)` matches the LeeDalley values exactly (correlation = 1.0000, max difference = 0.0). The combined h5ad stores all expression as log2(FPKM+1).

Both datasets share the same 50,281 gene set with identical gene ordering.

### Cell Identifiers

Cells are identified by three different ID systems across the datasets:

| ID Type | Format | Used by |
|---|---|---|
| `specimen_id` | Integer (e.g., `569871062`) | Both datasets — **primary join key** |
| `cell_name` | String (e.g., `H17.03.002.11.09.08`) | Both datasets |
| `patched_cell_container` | String (e.g., `P1S4_170214_008_A01`) | LeeDalley expression matrix column names |
| `exp_component_name` | String (e.g., `SM-GE4QD_S054_E1-50`) | L1 metadata; scANVI index |

The expression matrices use `patched_cell_container` as column names (in LeeDalley) and `sample_id` (in L1, same format). The `specimen_id` integer links to the metadata CSVs.

### scANVI Labels

scANVI cell type labels (mapped to SEA-AD supertypes) are available only for 383 cells, primarily from the L1 dataset. The scANVI results file uses `exp_component_name` as its index, which matches L1 but not LeeDalley cells. LeeDalley cells retain their original transcriptomic type labels but lack fine-grained SEA-AD supertype assignments.

### Electrophysiology

82 electrophysiology features are shared between both datasets with identical column names and extraction pipelines. 11 features are LeeDalley-only and 30 are L1-only (including QC flag columns). For the 38 overlap cells with ephys, 80 of 82 shared features match exactly; 2 features have minor rounding differences.

### Overlap Cells

43 cells appear in both datasets. For these cells:
- Expression is identical (after log2 transform)
- Metadata is merged (LeeDalley fields + L1 fields combined)
- `dataset` column is set to `"both"`

---

## UMAP Visualizations

### Gene Expression UMAP

![Expression UMAP](../results/figures/patchseq_umap_expression.png)

Computed from 3,000 highly variable genes, 30 PCs, 15 nearest neighbors. Cell types form distinct clusters: Pvalb and Sst separate clearly on the right side, while Lamp5, Vip, and Sncg types cluster together on the left. This is consistent with the known transcriptomic hierarchy where GABAergic interneurons split first by major subclass.

### Electrophysiology UMAP

![Ephys UMAP](../results/figures/patchseq_umap_ephys.png)

Computed from 6 core electrophysiology features (sag, tau, input resistance, rheobase, FI slope, resting Vm). 874 cells with complete ephys data are shown. Subclass structure is visible but less discrete than in the expression UMAP, reflecting the higher variability and overlap of electrophysiological properties across interneuron types.

### Dataset of Origin

![Dataset UMAP](../results/figures/patchseq_umap_dataset.png)

The two datasets intermingle in expression space, indicating no major batch effect between the LeeDalley and L1 cohorts. The L1 cells (blue) cluster predominantly in the Lamp5/Sncg/Pax6 region (left), reflecting their Layer 1 origin, while LeeDalley cells (red) span the full subclass range.

---

## Key Considerations

1. **scANVI coverage is partial.** Only L1 cells have scANVI supertype labels. LeeDalley cells have original cluster labels (`Transcriptomic_type`) but these use a different taxonomy than the SEA-AD supertypes. Extending scANVI label transfer to LeeDalley cells would require re-running the scANVI model with the `patched_cell_container` → `exp_component_name` mapping.

2. **Subclass composition differs by dataset.** LeeDalley provides deep coverage of Pvalb (327) and Sst (247), while L1 adds Lamp5 (161), Sncg (107), and Pax6 (65). Analyses spanning all subclasses benefit from the combined dataset, but within-subclass analyses should consider dataset of origin as a potential covariate.

3. **Expression is log2(FPKM+1), not counts.** The data cannot be directly used with count-based methods (e.g., scVI, negative binomial models). For those, the original FPKM values can be recovered via `2^X - 1`, but note that FPKM is itself a derived measure.

4. **Disease context.** Most cells come from epilepsy (470 LeeDalley + 126 L1) or tumor (261 LeeDalley + 20 L1) resection tissue. This is not healthy control tissue, and disease effects on gene expression or electrophysiology should be considered.

5. **Culture effects.** LeeDalley includes both acute (311) and cultured (468) slices. The L1 dataset is predominantly acute (all Patch-seq Production). The `condition` column in the metadata tracks this.

---

## Reproducing the Dataset

```bash
# Step 1: Build joined CSV tables
python scripts/build_joined_patchseq_tables.py

# Step 2: Build h5ad with expression, metadata, and UMAPs
python scripts/build_combined_patchseq_h5ad.py
```

### Requirements

- Python 3.10+
- `pyreadr` (for loading .RData files)
- `scanpy`, `anndata`
- `umap-learn`
- `sklearn`
- `matplotlib`, `pandas`, `numpy`

### Input Dependencies

- `complete_patchseq_data_sets.RData` (28 MB) — LeeDalley expression + annotations
- `patchseq_human_L1-main/data/ps_human.RData` (12 MB) — L1 expression + annotations
- `data/patchseq/LeeDalley_manuscript_metadata_v2.csv` — LeeDalley metadata
- `data/patchseq/LeeDalley_ephys_fx.csv` — LeeDalley electrophysiology
- `patchseq_human_L1-main/data/human_l1_dataset_2023_02_06.csv` — L1 metadata
- `patchseq_human_L1-main/data/aibs_features_E.csv` — L1 electrophysiology
- `data/patchseq/iterative_scANVI_results_patchseq_only.2022-11-22.csv` — scANVI labels
