# Patch-seq Human GABAergic Cell Viewer

Interactive multi-modal explorer for ~1,139 human cortical GABAergic interneurons profiled with Patch-seq. The viewer links three UMAP projections (gene expression, morphology, electrophysiology) with hover-triggered display of morphology reconstructions and electrophysiology recordings.

## Source Datasets

### Patch-seq Data

- **Lee, Dalley et al. (2023)** — 779 cells, multi-layer MTG Patch-seq
  [doi:10.1126/science.adf6484](https://doi.org/10.1126/science.adf6484) |
  [GitHub](https://github.com/AllenInstitute/human_patchseq_gaba)

- **Chartrand et al. (2023)** — 419 cells, human L1 interneurons
  [doi:10.1126/science.adf0805](https://doi.org/10.1126/science.adf0805) |
  [GitHub](https://github.com/AllenInstitute/patchseq_human_L1)

### Reference & Cell Type Labels

- **SEA-AD (Gabitto et al. 2024)** — snRNA-seq reference for Harmony integration and scANVI label transfer
  [doi:10.1038/s41593-024-01774-5](https://doi.org/10.1038/s41593-024-01774-5)

### Electrophysiology Recordings (DANDI)

- [DANDI:000636](https://dandiarchive.org/dandiset/000636) — Lee, Dalley / AIBS (691 NWBs)
- [DANDI:000630](https://dandiarchive.org/dandiset/000630) — L1 / AIBS (210 NWBs)
- [DANDI:000228](https://dandiarchive.org/dandiset/000228) — Lee-Bhatt (91 NWBs)
- [DANDI:000337](https://dandiarchive.org/dandiset/000337) — (21 NWBs)

Trace SVGs generated with [pyAPisolation](https://github.com/smestern/pyAPisolation) via the [intraDANDI explorer](https://github.com/smestern/intraDANDI_explorer) (Sam Mestern).

### Morphology Reconstructions (Brain Image Library)

SWC files downloaded from [BIL](https://www.brainimagelibrary.org/) across 7 submission directories.

## Repository Structure

```
patchseq_builder/              # Python package
  config.py                    # All paths, parameters, URLs
  naming.py                    # Cell type name normalization
  validate.py                  # Pipeline stage validation
  metadata/                    # Metadata harmonization, ephys features, scANVI labels
  expression/                  # Expression loading, normalization, contamination, UMAP
  reference/                   # Harmony integration, kNN label transfer, colors
  morphology/                  # SWC download, orientation, SVG rendering
  traces/                      # NWB processing, trace SVG generation, symlinks
  viewer/                      # Data prep, Plotly HTML builder, Jinja2 template
scripts/
  build_patchseq_viewer.py     # Main 7-stage pipeline
  deploy.py                    # Build deployment bundle for Netlify
data/                          # Input data (mostly gitignored)
results/                       # Pipeline outputs (mostly gitignored)
```

## Replication Guide

### Prerequisites

- Python 3.10+
- Key dependencies: `scanpy`, `anndata`, `harmonypy`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `plotly`, `jinja2`, `pyreadr`
- Optional: `pyAPisolation` (for generating trace SVGs from NWB files)

```bash
pip install scanpy anndata harmonypy pandas numpy scipy scikit-learn matplotlib plotly jinja2 pyreadr
# Optional: pip install git+https://github.com/smestern/pyAPisolation.git
```

### Step 1: Clone upstream data repos

Clone into the project root directory:

```bash
cd human_int_patch_seq/

git clone https://github.com/AllenInstitute/human_patchseq_gaba.git
git clone https://github.com/AllenInstitute/patchseq_human_L1.git

# Pre-generated trace SVGs from intraDANDI explorer
git clone https://github.com/smestern/intraDANDI_explorer.git
mv intraDANDI_explorer intraDANDI_explorer-master
```

### Step 2: Obtain the SEA-AD snRNA-seq reference

Download `SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad` from the [SEA-AD data portal](https://portal.brain-map.org/). Place or symlink it at:

```
data/seaad_reference.h5ad
```

### Step 3: Obtain scANVI labels

The scANVI label transfer results CSV should be placed at:

```
data/patchseq/iterative_scANVI_results_patchseq_only.2022-11-22.csv
```

This file contains scANVI-transferred subclass and supertype labels for 4,549 cells.

### Step 4: Run the pipeline

```bash
python3 scripts/build_patchseq_viewer.py
```

The pipeline runs 7 stages in order:

| Stage | Description | Key Outputs |
|-------|-------------|-------------|
| 1 | Metadata harmonization | `patchseq_combined.csv` |
| 2 | Expression h5ad + UMAPs (expression, ephys, morphology) | `patchseq_combined.h5ad` |
| 3 | Harmony integration + kNN label transfer | `patchseq_reference_combined.h5ad` |
| 4 | Morphology SWC download + SVG rendering | `morphology_svgs/*.svg` |
| 5 | Trace SVG mapping + symlinks | `traces/` symlinks |
| 6 | Interactive viewer HTML | `patchseq_umap_interactive.html` |
| 7 | Comprehensive accounting | `data_source_accounting.txt` |

Useful flags:
- `--only N` — run only stage N
- `--force` — regenerate all cached outputs
- `--from N` — restart from stage N

Stage 4 (morphology download + rendering) takes ~20 minutes on first run. Other stages complete in seconds to minutes.

### Step 5: Deploy (optional)

Build a lightweight deployment bundle and deploy to Netlify:

```bash
python3 scripts/deploy.py
netlify deploy --dir=deploy --prod
```

The deploy script parses the viewer HTML to identify only the SVGs actually referenced, producing a self-contained ~515 MB bundle (vs ~4.7 GB if all SVGs were included).

## Credits

Developed by [Shreejoy Tripathy](https://triplab.org/) with the assistance of [Claude Code](https://claude.ai/claude-code).

Patch-seq cells were collected from resected neocortical human neurosurgical tissues. The snRNA-seq reference was collected from post-mortem donors. Data from the Allen Institute for Brain Science, SEA-AD, DANDI Archive, and Brain Image Library.
