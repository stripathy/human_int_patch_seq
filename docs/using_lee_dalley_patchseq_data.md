# Using the Lee & Dalley Patch-Seq Data for Gene-Electrophysiology Analyses

## Overview

The Lee & Dalley human patch-seq dataset provides simultaneous transcriptomic and electrophysiological recordings from individual neurons in human middle temporal gyrus (MTG). This document describes how to use these data for integrative gene expression and electrophysiology correlation analyses, as implemented in this project's pipeline (Step 06).

---

## Available Data Files

All files are in `data/patchseq/`:

| File | Rows | Description |
|------|------|-------------|
| `LeeDalley_manuscript_metadata_v2.csv` | 779 cells | Cell-level metadata: donor, brain region, cortical layer, disease category, transcriptomic type, subclass labels |
| `LeeDalley_ephys_fx.csv` | 704 cells | Extracted electrophysiology features per cell (~90 features): sag, tau, input resistance, firing rate, spike shape, etc. |
| `iterative_scANVI_results_patchseq_only.2022-11-22.csv` | 4,549 cells | scANVI cell type label transfer from SEA-AD reference: supertype and subclass assignments with confidence scores |

### Key columns

**Metadata** (`LeeDalley_manuscript_metadata_v2.csv`):
- `specimen_id_x` — unique cell identifier (integer), links to ephys
- `cell_name` — human-readable cell name (e.g., `H17.06.003.11.06.02`)
- `Donor` — donor ID
- `Cortical_layer` — layer of recording (1-6)
- `subclass_label`, `Revised_subclass_label` — original transcriptomic subclass
- `Transcriptomic_type` — original t-type classification
- `patched_cell_container` — links to scANVI results via `exp_component_name`

**Electrophysiology** (`LeeDalley_ephys_fx.csv`):
- `specimen_id` — join key to metadata
- `sag` — voltage sag ratio during hyperpolarization (HCN channel activity)
- `tau` — membrane time constant (ms)
- `input_resistance` — input resistance (MOhm)
- `rheobase_i` — minimum current for spiking
- `fi_fit_slope` — firing rate gain
- ~85 additional spike shape, firing pattern, and subthreshold features

**scANVI labels** (`iterative_scANVI_results_patchseq_only.2022-11-22.csv`):
- Index is `exp_component_name` (e.g., `SM-GE4W8_S016_E1-50`)
- `subclass_scANVI` — mapped SEA-AD subclass (e.g., Sst, Pvalb, Vip)
- `supertype_scANVI` — mapped SEA-AD supertype (e.g., Sst_1, Pvalb_15)
- `subclass_conf_scANVI`, `supertype_conf_scANVI` — confidence scores (0-1)

---

## Core Analysis: Supertype-Level Gene-Ephys Correlations

The primary analysis in this project correlates gene expression specificity (from the SEA-AD snRNA-seq reference) with mean electrophysiology features (from patch-seq recordings) across SST interneuron supertypes. This is implemented in `scripts/06_gene_ephys_correlations.py` and the library module `scz_celltype_enrichment/integration/gene_ephys.py`.

### Step 1: Prepare mean ephys per supertype

The file `sst_supertype_layer_info.csv` (in the project root) contains pre-computed mean SAG and TAU per SST supertype, derived from patch-seq cells that were assigned SST supertype labels via scANVI:

```python
import pandas as pd

sst_layer_info = pd.read_csv("sst_supertype_layer_info.csv")
# Columns: supertype_scANVI, mean_layer, median_layer, n_cells, mean_sag, mean_tau, layer_group
```

This table was built by:
1. Merging metadata + ephys on `specimen_id`
2. Joining scANVI supertype labels via `patched_cell_container` / `exp_component_name`
3. Filtering to SST-labeled cells
4. Computing mean SAG and TAU per supertype

### Step 2: Correlate specificity with ephys across supertypes

Each data point is one SST supertype. For each gene, we compute the Spearman correlation between its expression specificity across SST supertypes and the supertype's mean ephys value:

```python
from scz_celltype_enrichment.integration.gene_ephys import compute_supertype_correlations

# specificity: genes x supertypes matrix (from SEA-AD, Step 01)
# ephys_by_supertype: Series with mean_sag or mean_tau, indexed by supertype name

sag_corr = compute_supertype_correlations(
    specificity[sst_types],
    ephys_by_supertype["mean_sag"],
    feature_name="SAG"
)
# Returns: gene, spearman_rho, pval, pval_fdr, abs_rho
```

This asks: **which genes are more specifically expressed in SST supertypes that have higher (or lower) sag?**

### Step 3: Find discordant genes (sag+/tau-)

Genes positively correlated with sag but negatively correlated with tau identify candidates related to HCN channel biology. HCN1 is the prototypical example: it increases sag (by conducting Ih current) while decreasing tau (by lowering effective membrane resistance).

```python
from scz_celltype_enrichment.integration.gene_ephys import find_discordant_genes

discordant = find_discordant_genes(
    sag_corr, tau_corr,
    sag_direction="+", tau_direction="-",
    p_threshold=0.05
)
# Returns genes ranked by combined_score = |rho_sag| + |rho_tau|
```

### Step 4: Annotate with GWAS

Flag which of these electrophysiology-associated genes also fall in SCZ GWAS risk loci:

```python
from scz_celltype_enrichment.integration.gene_ephys import annotate_with_gwas
from scz_celltype_enrichment.enrichment.gwas import load_scz_gwas_gene_set

scz_genes = load_scz_gwas_gene_set("scz_gwas_gene_set.csv")
discordant = annotate_with_gwas(discordant, scz_genes)
# Adds 'is_scz' boolean column
```

---

## Alternative: Cell-Level Correlations

The pipeline also supports direct cell-level correlations using patch-seq gene expression and ephys from the same cell (rather than the supertype-level aggregation above):

```python
from scz_celltype_enrichment.integration.gene_ephys import compute_gene_ephys_correlations

# gene_values: (n_cells, n_genes) array of expression
# ephys_values: (n_cells,) array of sag/tau
cell_corr = compute_gene_ephys_correlations(
    gene_values, ephys_values,
    gene_names=gene_names,
    min_expressing=10  # require >= 10 cells expressing the gene
)
```

### Controlling for cortical depth

SAG varies systematically with cortical layer, so correlations may be confounded by depth. Use residualized correlations to control for this:

```python
from scz_celltype_enrichment.integration.gene_ephys import compute_residualized_correlations

# Regresses ephys ~ depth, then correlates residuals with gene expression
resid_corr = compute_residualized_correlations(
    gene_values, ephys_values, depth_values,
    gene_names=gene_names
)
```

---

## Relationship to the patchseq_human_L1-main Repository

The `patchseq_human_L1-main/` directory in this project contains the original Lee et al. analysis codebase and data (accompanying the Science paper: https://www.science.org/doi/10.1126/science.adf0805). It is **not imported** by this project's enrichment pipeline.

However, it contains additional data that could extend the analyses:

| File | Description |
|------|-------------|
| `data/human_l1_dataset_2023_02_06.csv` | 419 human L1 interneurons with metadata, t-type labels, depth, morphology flags |
| `data/human_l1_dataset_strict.csv` | 272-cell subset with stricter QC |
| `data/aibs_features_E.csv` | 263 cells with full ephys feature extraction (same features as LeeDalley_ephys_fx.csv) |
| `data/RawFeatureWide_human+derivatives.csv` | 86 cells with morphological features |
| `patchseq_utils/` | Python library for patch-seq analysis (d-prime, clustering, classification) |

**Overlap**: 43 cells appear in both the L1 dataset and the LeeDalley dataset. The L1 dataset adds 376 cells that are exclusively Layer 1 interneurons (mostly LAMP5/PAX6/Other and VIP subclasses). Of these, 242 have both ephys features and scANVI supertype labels.

The L1 cells map to different SEA-AD supertypes than the SST-focused analyses here — primarily Lamp5 (114), Sncg (65), Pax6 (41), and Vip (17) subclasses. This means they could be used to extend the gene-ephys correlation framework to non-SST interneuron classes, particularly L1-specific types like rosehip cells (LAMP5 LCP2) and LAMP5 NMBR cells.

---

## Quick Start: Running the Analysis

```bash
# Prerequisites: run Steps 01-02 to generate specificity matrix and enrichment results
python scripts/01_compute_specificity.py
python scripts/02_magma_enrichment.py

# Run gene-ephys correlations
python scripts/06_gene_ephys_correlations.py
```

### Output files (in `results/tables/`):
- `sag_gene_correlations_seaad_supertypes.csv` — all genes ranked by sag correlation
- `tau_gene_correlations_seaad_supertypes.csv` — all genes ranked by tau correlation
- `sag_pos_tau_neg_genes.csv` — discordant genes with GWAS annotation
- `sst_supertype_ephys_summary.csv` — mean ephys per SST supertype

### Output figures (in `results/figures/`):
- `hcn1_vs_sag.png` — HCN1 specificity vs mean sag across SST supertypes
- `sst_layer_tau_sag_relationships.png` — layer depth vs sag/tau relationships

---

## Key Considerations

1. **Sample sizes are small at the supertype level.** The supertype-level correlations use ~8-12 SST supertypes as data points. Spearman correlations with n < 10 should be interpreted cautiously and treated as hypothesis-generating rather than confirmatory.

2. **scANVI label confidence varies.** Not all patch-seq cells map cleanly to SEA-AD supertypes. Filter on `supertype_conf_scANVI` if tighter assignments are needed.

3. **Specificity comes from a different dataset.** Gene expression specificity is computed from SEA-AD snRNA-seq (post-mortem aging brain), while ephys comes from patch-seq (acute neurosurgical tissue, often epilepsy patients). This cross-dataset design is a strength (independent measurements) but also means the correlation is ecological — it assumes that expression patterns are conserved between the two cohorts.

4. **Depth is a confound.** SST subtypes occupy different cortical layers, and both gene expression and ephys features vary with depth. The spatial analysis (Step 05) and residualized correlations address this, but it should always be kept in mind.
