# Triple-modal cells for biophysical / computational modeling

These two CSVs together identify the 208 human cortical interneurons in this
project that have **upstream-QC-passing data in all three modalities**
(electrophysiology, morphology, transcriptomics) and point to the original
files for each modality. They are intended for colleagues building
single-neuron biophysical models who need raw NWB recordings + SWC
reconstructions, with cell-type labels for stratification.

| File | Rows × Cols | What it gives you |
|------|-------------|--------------------|
| `triple_modal_cell_metadata.csv` | 208 × 79 | Cell-level metadata: identity, donor demographics, anatomy, all transcriptomic labels (original + scANVI + kNN), 16 key ephys features, 17 key morphology features, UMAP coords. |
| `triple_modal_data_manifest.csv` | 208 × 24 | Per-cell pointers to ephys NWB (DANDI URL + dandiset path), morphology SWC (BIL URL + local path), and transcriptomic source (h5ad path + obs index, or upstream RData). |

## Cell selection criteria

A cell appears in both files iff it is in `patchseq_all_features.csv` with:

- `has_ephys == True` (upstream QC pass — see below)
- `has_morphology == True` (upstream QC pass — see below)
- has a transcriptomic type label (all 208 cells have `transcriptomic_type_original`
  populated and `genes_detected ≥ 2,461`)

Breakdown by source dataset:

| Dataset | Count |
|---------|-------|
| Lee, Dalley et al. 2023 | 124 |
| Chartrand et al. 2023 (L1) | 72 |
| Both studies (overlap) | 12 |
| **Total** | **208** |

`has_ephys` and `has_morphology` come directly from the upstream studies'
metadata:

- Lee, Dalley: `Has_ephys`, `Has_morphology` columns in
  `human_patchseq_gaba/data/LeeDalley_manuscript_metadata_v2.csv` (these reflect
  the authors' QC for inclusion in their published analyses).
- L1 / Chartrand: `has_ephys`, `has_morph` columns in
  `patchseq_human_L1/data/human_l1_dataset_2023_02_06.csv`.

## Subclass distribution (scANVI)

| scANVI subclass | n |
|-----------------|---|
| Pvalb | 59 |
| Lamp5 | 32 |
| Sst | 30 |
| Pax6 | 28 |
| Vip | 25 |
| Sncg | 21 |
| Lamp5_Lhx6 | 4 |
| Chandelier | 2 |
| L4 IT | 2 |
| Sst Chodl | 2 |
| (unlabeled) | 3 |

(See also `subclass_label_original`, `transcriptomic_type_original`,
`knn_subclass`, etc. for alternative label schemes.)

## How to fetch the underlying data

### Electrophysiology (NWB)

Each row gives a DANDI dandiset, the relative path within it, and an
`ephys_dandi_download_url` that resolves to the file. The URL pattern is the
DANDI asset download endpoint, which 302-redirects to an S3 blob:

```
https://api.dandiarchive.org/api/assets/<asset-id>/download/
```

`ephys_status` flags how the cell was matched:

| Status | n | Meaning |
|--------|---|---------|
| `dandi_mapped` | 182 | Matched via the existing `specimen_to_dandi_map.csv` (DANDI:000630 / 000636). URL resolves directly. |
| `dandi_228_lab28` | 12 | Matched to DANDI:000228 (Lee-Bhatt) via the lab28 cell↔NWB matching. URL resolves directly. |
| `not_in_dandi_map` | 14 | Cell is QC-flagged for ephys upstream but no DANDI asset has been linked yet. The recording files exist on internal Allen systems; ask the upstream authors if you need them. |

For the 14 `not_in_dandi_map` cells, the original ephys feature values are still
accessible via the upstream features CSV (column
`ephys_features_csv_upstream`), but the raw NWB traces are not currently
exposed publicly.

If you prefer the DANDI Python client:

```python
from dandi.dandiapi import DandiAPIClient
with DandiAPIClient() as client:
    dandiset = client.get_dandiset("000636", "draft")
    asset = dandiset.get_asset_by_path("sub-X/sub-X_ses-Y_icephys.nwb")
    asset.download("local.nwb")
```

### Morphology (SWC)

Each row gives a `morphology_swc_url` pointing to the Brain Image Library
submission that hosts the file. All 208 SWCs are also already downloaded into
this repo at `morphology_local_swc_path` (typically
`data/morphology/swc/<specimen_id>_upright.swc`) so you can use them
without hitting the network.

`morphology_bil_source` records which BIL submission directory was used:

| BIL submission | n |
|----------------|---|
| `69fe931fee2b2215` (manual reconstructions, `*_m.swc`) | 71 |
| `group/20230426/swc` | 56 |
| `241a10cde842c99b` (transformed) | 54 |
| `d833ba8bd931f23f` | 15 |
| `49e6114ba67eda01` | 8 |
| `efb9b12ba2fab63d` | 3 |
| `85f4b93699151f1c` | 1 |

Note that `69fe…` cells use non-standard filenames like
`H17.03.002.11.09.08_<id>_m.swc` rather than `<sid>_upright.swc`; the URLs in
the manifest are the resolved final filenames, not the directory listing.

Both BIL URLs and DANDI URLs were spot-checked at manifest build time. Files
move occasionally; if a URL fails, fall back to the local SWC.

### Transcriptomics

`expression_status` indicates where to find expression for the cell:

| Status | n | Where |
|--------|---|-------|
| `in_h5ad_export` | 190 | Use `expression_h5ad_path` (one of the two h5ads in this directory). Look up the cell by `expression_h5ad_obs_index` (the obs name) or by `specimen_id` in `.obs`. |
| `raw_rdata_only` | 18 | Cell is in the upstream RData but was dropped during the harmonized export (12 "both" cells + 6 L1 cells). Use `expression_raw_source` — load `complete_patchseq_data_sets.RData` (LeeDalley `datPatch` object) or `patchseq_human_L1/data/ps_human.RData` (`datPS` object) via `pyreadr` or R. |

```python
import anndata
adata = anndata.read_h5ad("data/patchseq/exports/patchseq_leedalley.h5ad")
cell = adata[adata.obs["specimen_id"].astype(str) == "643397120"]
```

## Manifest columns

| Column | Description |
|--------|-------------|
| `specimen_id` | Allen specimen ID (primary key). |
| `cell_name` | Allen cell name (e.g. `H17.03.002.11.09.08`). |
| `dataset` | `LeeDalley`, `L1`, or `both`. |
| `donor`, `lab` | Donor ID and originating lab. |
| `ephys_status` | See "Electrophysiology" above. |
| `ephys_dandiset` | Six-digit DANDI ID (`000630`, `000636`, `000228`). |
| `ephys_dandi_path` | Asset path inside the dandiset (relative). |
| `ephys_dandi_asset_id` | DANDI asset UUID. Empty if not in DANDI. |
| `ephys_dandi_download_url` | Direct download URL (302-redirects to S3). |
| `ephys_local_nwb_path` | Path inside this repo if the NWB was cached locally (sparse — only ~4 cells). |
| `ephys_features_csv_upstream` | Path to the upstream ephys features CSV containing this cell's row. |
| `morphology_swc_url` | Direct download URL for the SWC at BIL. |
| `morphology_bil_source` | Which BIL submission directory served the SWC. |
| `morphology_local_swc_path` | Path to the SWC inside this repo. |
| `morphology_features_csv_upstream` | Path to upstream morphology features CSV. |
| `expression_status`, `expression_h5ad_path`, `expression_h5ad_obs_index`, `expression_raw_source` | See "Transcriptomics" above. |
| `morphology_svg_repo`, `ephys_trace_svg_repo`, `ephys_fi_curve_svg_repo` | Pre-rendered figures inside this repo (handy for QC). |
| `upstream_metadata_csv` | Path to the upstream raw metadata CSV. |

## Build provenance

These files were generated from `patchseq_all_features.csv` plus the helper
maps under `data/patchseq/`:

- `specimen_to_dandi_map.csv` — DANDI 630/636 mapping (existing)
- `dandi_630_asset_map.csv`, `dandi_636_asset_map.csv` — asset UUIDs (existing)
- `dandi_228_asset_map.csv` — fetched from DANDI API at build time
- `lab28_cell_vs_nwb_matching.csv` — Lab28 ↔ DANDI 228 mapping (existing)
- `specimen_to_bil_url.csv` — generated by HEAD-probing the BIL direct URL
  patterns in `patchseq_builder/config.py` against each specimen, and resolving
  the `69fe…` directory listings
- `patchseq_leedalley.h5ad`, `patchseq_l1.h5ad` — checked for `specimen_id`
  presence to determine `expression_status`
