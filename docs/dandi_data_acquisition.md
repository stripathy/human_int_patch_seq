# Downloading NWB Files from DANDI and Generating Electrophysiology Trace SVGs

A standalone reference for acquiring patch-clamp recording data from the DANDI Archive
and rendering voltage-trace SVGs for integration into the interactive UMAP viewer.

---

## 1. Overview

### What is DANDI?

The **DANDI Archive** (Distributed Archives for Neurophysiology Data Integration,
[dandiarchive.org](https://dandiarchive.org/)) is an open-access repository for
neurophysiology datasets. Data are organized into **dandisets**, each identified by
a six-digit code (e.g., `000636`). Individual recording files within a dandiset are
stored as **NWB (Neurodata Without Borders)** files, a standardized HDF5-based format
that packages raw electrophysiology sweeps, stimulus waveforms, and metadata into a
single file per recording session.

### What NWB files contain (for patch-clamp)

Each NWB file in these dandisets typically contains:

- **CurrentClampSeries**: voltage recordings under injected-current stimuli
- **CurrentClampStimulusSeries**: the corresponding current injection waveforms
- **Sweep metadata**: protocol names (e.g., "Long square", "Ramp", "Short square"),
  stimulus amplitudes, sampling rates
- **Subject metadata**: species, age, sex, donor ID
- **Session metadata**: session ID, recording date, lab/institution

### Dandisets containing our patch-seq data

| Dandiset | Name / Lab | NWB Count | URL |
|----------|-----------|-----------|-----|
| **000636** | LeeDalley (AIBS) | ~691 | [dandiarchive.org/dandiset/000636](https://dandiarchive.org/dandiset/000636) |
| **000630** | L1 cells (AIBS) | ~210 | [dandiarchive.org/dandiset/000630](https://dandiarchive.org/dandiset/000630) |
| **000228** | Lab-28 (Lee-Bhatt) | ~91 | [dandiarchive.org/dandiset/000228](https://dandiarchive.org/dandiset/000228) |
| **000337** | Lab-29 | ~21 | [dandiarchive.org/dandiset/000337](https://dandiarchive.org/dandiset/000337) |

**000636** is the primary dandiset for the LeeDalley manuscript. NWB files use
AIBS numeric IDs for subjects and sessions:
`sub-{subject_id}/sub-{subject_id}_ses-{session_id}_icephys.nwb`.

**000630** contains L1 interneuron recordings. Same AIBS naming convention.

**000228** uses human-readable donor codes:
`sub-{donor}/sub-{donor}_obj-{hash}_icephys.nwb` (e.g., `sub-H18-28-025_obj-4ubkfz_icephys.nwb`).

**000337** uses AIBS numeric IDs, similar to 000636.

### Current coverage

- **968 cells** have trace SVGs mapped in `data/patchseq/specimen_to_svg_map.csv`
- **84 cells** with `has_ephys=True` in the metadata are **missing** trace SVGs
  (documented in `data/patchseq/missing_ephys_accounting.csv`)

---

## 2. Prerequisites

### Python packages

```bash
# Core DANDI tools
pip install dandi pynwb

# Sam Mestern's analysis/plotting library (includes NWB loading + trace rendering)
pip install git+https://github.com/smestern/pyAPisolation.git

# Allen Institute feature extraction (installed as dependency, but for clarity)
pip install ipfx --no-deps

# Standard scientific stack
pip install matplotlib numpy pandas
```

### Verification

```python
from dandi.dandiapi import DandiAPIClient
from pyAPisolation.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from pyAPisolation.database.build_database import plot_data, build_dataset_traces
import pynwb
print("All imports successful")
```

---

## 3. How Sam's intraDANDI Explorer Works

The `intraDANDI_explorer-master/` directory contains Sam Mestern's tool for scraping,
analyzing, and visualizing intracellular electrophysiology data across all of DANDI.
The relevant code lives in `intraDANDI_explorer-master/dandi_scraper/dandi_scraper.py`.

### Architecture

```
dandi_scraper.py
  |
  |-- download_dandiset()     # Downloads NWBs via dandi API
  |-- analyze_dandiset()      # Extracts ephys features via pyAPisolation
  |-- run_plot_dandiset()     # Renders trace + FI SVGs for all cells
  |-- sort_plot_dandiset()    # Moves SVGs into data/traces/{dandiset}/{subject}/
  |-- build_server()          # Launches interactive web viewer
```

### Key function: `plot_data()`

Imported from `pyAPisolation.database.build_database`:

```python
from pyAPisolation.database.build_database import plot_data
```

What it does:

1. **Loads the NWB file** using `loadNWB()` from pyAPisolation
2. **Filters for Long Square current-clamp sweeps** using `GLOBAL_STIM_NAMES`
   - Looks for protocol names containing "Long square" (or variants)
   - Falls back to broader stimulus name matching via `stim_inc` configuration
3. **Selects representative sweeps** at target current amplitudes:
   `[-100, -20, 20, 100, 150, 250, 500, 1000]` pA (default)
   - Finds the sweep closest to each target amplitude
4. **Renders two SVG figures**:
   - **Voltage trace SVG** (4x3 inches): overlaid voltage traces from selected
     sweeps, black lines with alpha=0.5
   - **FI curve SVG** (3x3 inches): firing rate vs. injected current

**Function signature** (simplified):

```python
result = plot_data(
    index=0,                    # Index (unused when processing single files)
    files=["/path/to/file.nwb"],
    target_amps=[-100, -20, 20, 100, 150, 250, 500, 1000],
    overwrite=True,
    save=True,                  # If True, saves SVGs next to the NWB file
    stim_override='',           # Override stimulus name filter ('' = use default)
)
```

**Returns** a dict containing:
- `fig_trace`: matplotlib Figure for the voltage trace
- `fi_fig`: matplotlib Figure for the FI curve
- `sweep_xs`, `sweep_ys`: raw sweep time and voltage arrays
- SVG files saved as `{nwb_filename}.svg` and `{nwb_filename}_FI.svg`

### The `build_dataset_traces()` function

Batch-processes all NWB files in a directory:

```python
from pyAPisolation.database.build_database import build_dataset_traces

build_dataset_traces(
    folder="/path/to/dandiset/000636",
    ids=list_of_relative_paths,  # e.g., ["sub-636948822/sub-636948822_ses-636982248_icephys.nwb"]
    parallel=True                # Uses joblib for parallel processing
)
```

### The `sort_plot_dandiset()` function

After SVGs are generated next to NWB files on the download drive, this function
relocates them into the organized structure:

```
data/traces/{dandiset}/{subject}/{nwb_filename}.svg
data/traces/{dandiset}/{subject}/{nwb_filename}_FI.svg
```

For example:
```
data/traces/000636/sub-636948822/sub-636948822_ses-636982248_icephys.nwb.svg
data/traces/000636/sub-636948822/sub-636948822_ses-636982248_icephys.nwb_FI.svg
```

---

## 4. Step-by-Step: Downloading NWBs from DANDI

### Option A: Download an entire dandiset (CLI)

```bash
# Download all NWBs from dandiset 000636
dandi download DANDI:000636

# Download to a specific directory
dandi download DANDI:000636 --output-dir /path/to/storage/
```

This will create a directory structure:
```
000636/
  sub-636948822/
    sub-636948822_ses-636982248_icephys.nwb
    sub-636948822_ses-636991746_icephys.nwb
  sub-642843699/
    ...
```

**Warning**: Dandiset 000636 is several hundred GB. Consider downloading
individual files instead if you only need a subset.

### Option B: Download an entire dandiset (Python API)

```python
from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download

client = DandiAPIClient()
dandiset = client.get_dandiset("000636")
dandi_download(dandiset.api_url, "/path/to/storage/")
```

### Option C: Download individual NWBs via S3 URL (fastest for specific cells)

Each NWB has a direct S3 download URL. These are stored in our mapping files.

```python
import requests
from pathlib import Path

# Look up the S3 URL from specimen_to_svg_map.csv or the DANDI API
s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/4e2/df3/4e2df3b5-64f1-4c1b-90b9-5b2eaf6aceb4"

# Download
resp = requests.get(s3_url, stream=True)
output_path = Path("000636/sub-636948822/sub-636948822_ses-636982248_icephys.nwb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)
```

### Option D: Get S3 URL programmatically from DANDI API

```python
from dandi.dandiapi import DandiAPIClient

with DandiAPIClient() as client:
    dandiset = client.get_dandiset("000636", "draft")
    asset = dandiset.get_asset_by_path(
        "sub-636948822/sub-636948822_ses-636982248_icephys.nwb"
    )
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    print(s3_url)
```

### Option E: Stream NWB without downloading (for inspection only)

```python
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
import remfile
import h5py

with DandiAPIClient() as client:
    dandiset = client.get_dandiset("000636", "draft")
    asset = dandiset.get_asset_by_path(
        "sub-636948822/sub-636948822_ses-636982248_icephys.nwb"
    )
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# Stream via remfile (no full download)
rfile = remfile.File(s3_url)
h5 = h5py.File(rfile, "r")
io = NWBHDF5IO(file=h5, load_namespaces=True)
nwb = io.read()
print(nwb.session_id, nwb.subject.subject_id)
```

### Where the current data lives

The pre-generated SVGs are stored at:
```
intraDANDI_explorer-master/data/traces/
  000228/sub-H18-28-011/...
  000337/sub-1047717119/...
  000570/sub-907501068/...     (AIBS mouse data, not part of our study)
  000630/sub-1001372857/...
  000636/sub-1000181910/...
```

These are symlinked into the viewer's serving directory:
```
results/figures/traces/
  000228 -> intraDANDI_explorer-master/data/traces/000228
  000337 -> intraDANDI_explorer-master/data/traces/000337
  000630 -> intraDANDI_explorer-master/data/traces/000630
  000636 -> intraDANDI_explorer-master/data/traces/000636
```

---

## 5. Step-by-Step: Generating Trace SVGs

### Complete Python script for generating SVGs from NWB files

```python
#!/usr/bin/env python
"""
generate_trace_svgs.py -- Generate voltage-trace and FI-curve SVGs from NWB files.

Usage:
    python generate_trace_svgs.py /path/to/nwb_files/ /path/to/output_dir/

Requirements:
    pip install git+https://github.com/smestern/pyAPisolation.git
    pip install ipfx --no-deps
    pip install pynwb matplotlib numpy
"""
import os
import sys
import glob
import traceback
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for SVG generation
import matplotlib.pyplot as plt

# Force install ipfx without deps (it has strict numpy version requirements)
import subprocess
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "ipfx", "--no-deps"],
    stdout=subprocess.DEVNULL
)

from pyAPisolation.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from pyAPisolation.database.build_database import plot_data, build_dataset_traces


# ── Configuration ──────────────────────────────────────────────────────

# GLOBAL_STIM_NAMES controls which stimulus protocols are recognized.
# The default configuration looks for "Long square" in the protocol name.
# For AIBS NWB files, this works out of the box.
# If your NWB files use different protocol names, override:
#   GLOBAL_STIM_NAMES.stim_inc = ['']  # accept all protocol names

# Target current injection amplitudes (pA) for selecting representative sweeps.
# plot_data will find the recorded sweep closest to each target.
TARGET_AMPS = [-100, -20, 20, 100, 150, 250, 500, 1000]


def generate_svg_for_nwb(nwb_path, output_dir=None, overwrite=False):
    """
    Generate trace SVG and FI curve SVG for a single NWB file.

    Parameters
    ----------
    nwb_path : str
        Full path to the .nwb file.
    output_dir : str or None
        Where to save SVGs. If None, saves next to the NWB file.
    overwrite : bool
        If True, regenerate even if SVGs already exist.

    Returns
    -------
    dict with keys 'trace_svg' and 'fi_svg' (paths), or None on failure.
    """
    nwb_path = os.path.abspath(nwb_path)
    basename = os.path.basename(nwb_path)

    if output_dir is None:
        output_dir = os.path.dirname(nwb_path)

    trace_svg = os.path.join(output_dir, basename + ".svg")
    fi_svg = os.path.join(output_dir, basename + "_FI.svg")

    if not overwrite and os.path.exists(trace_svg) and os.path.exists(fi_svg):
        print(f"  [skip] SVGs already exist for {basename}")
        return {"trace_svg": trace_svg, "fi_svg": fi_svg}

    try:
        result = plot_data(
            0,
            [nwb_path],
            target_amps=TARGET_AMPS,
            overwrite=True,
            save=False,        # We will save manually to control output path
            stim_override='',  # Accept all stimulus names
        )

        if result is None or "fig_trace" not in result:
            print(f"  [warn] No Long Square sweeps found in {basename}")
            return None

        # Save trace SVG
        os.makedirs(output_dir, exist_ok=True)
        result["fig_trace"].savefig(trace_svg, format="svg", bbox_inches="tight")
        plt.close(result["fig_trace"])

        # Save FI curve SVG
        if "fi_fig" in result and result["fi_fig"] is not None:
            result["fi_fig"].savefig(fi_svg, format="svg", bbox_inches="tight")
            plt.close(result["fi_fig"])
        else:
            fi_svg = None

        print(f"  [ok] {basename}")
        return {"trace_svg": trace_svg, "fi_svg": fi_svg}

    except Exception as e:
        print(f"  [error] {basename}: {e}")
        traceback.print_exc()
        return None


def batch_generate(nwb_dir, output_dir, overwrite=False):
    """
    Generate SVGs for all NWB files in a directory tree.

    Parameters
    ----------
    nwb_dir : str
        Root directory containing NWB files (searched recursively).
    output_dir : str
        Root output directory. SVGs are placed in a mirror of the input tree.
    overwrite : bool
        If True, regenerate existing SVGs.
    """
    nwb_files = glob.glob(os.path.join(nwb_dir, "**/*.nwb"), recursive=True)
    print(f"Found {len(nwb_files)} NWB files in {nwb_dir}")

    results = []
    for i, nwb_path in enumerate(sorted(nwb_files)):
        rel_path = os.path.relpath(nwb_path, nwb_dir)
        out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        print(f"[{i+1}/{len(nwb_files)}] {rel_path}")
        result = generate_svg_for_nwb(nwb_path, output_dir=out_subdir, overwrite=overwrite)
        if result:
            results.append({"nwb": rel_path, **result})

    print(f"\nDone: {len(results)}/{len(nwb_files)} files processed successfully")
    return results


def batch_generate_parallel(nwb_dir, output_dir, overwrite=False):
    """
    Same as batch_generate but uses build_dataset_traces for parallel processing.
    This is how the intraDANDI explorer does it.
    """
    nwb_files = glob.glob(os.path.join(nwb_dir, "**/*.nwb"), recursive=True)
    ids = [os.path.relpath(f, nwb_dir) for f in nwb_files]
    print(f"Processing {len(ids)} NWB files in parallel...")
    build_dataset_traces(nwb_dir, ids, parallel=True)
    print("Done. SVGs saved next to NWB files.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_trace_svgs.py <nwb_dir> <output_dir> [--overwrite]")
        sys.exit(1)

    nwb_dir = sys.argv[1]
    output_dir = sys.argv[2]
    overwrite = "--overwrite" in sys.argv

    batch_generate(nwb_dir, output_dir, overwrite=overwrite)
```

### Generating SVGs for a single cell (interactive/notebook use)

```python
from pyAPisolation.database.build_database import plot_data
import matplotlib.pyplot as plt

# Load and render
result = plot_data(
    0,
    ["/path/to/sub-636948822/sub-636948822_ses-636982248_icephys.nwb"],
    target_amps=[-100, -20, 20, 100, 150, 250, 500, 1000],
    overwrite=True,
    save=False,
    stim_override='',
)

# Display in notebook
if result and "fig_trace" in result:
    display(result["fig_trace"])
    display(result["fi_fig"])

    # Or save manually
    result["fig_trace"].savefig("my_cell_trace.svg", format="svg")
    result["fi_fig"].savefig("my_cell_fi.svg", format="svg")
    plt.close("all")
```

### Handling NWBs without Long Square protocol

Some NWB files may not contain the Long Square protocol needed for standard trace
rendering. The `plot_data()` function will return `None` or a dict without
`fig_trace` in this case. The `dandi_000228_nwb_metadata.csv` file documents which
NWBs in dandiset 000228 have Long Square protocols:

```python
import pandas as pd
meta = pd.read_csv("data/patchseq/dandi_000228_nwb_metadata.csv")
has_ls = meta[meta["has_long_square"] == True]
print(f"{len(has_ls)}/{len(meta)} NWBs have Long Square protocol")
```

### GLOBAL_STIM_NAMES configuration

The `GLOBAL_STIM_NAMES` object from pyAPisolation controls which NWB sweep
protocol names are recognized as current-clamp stimuli:

```python
from pyAPisolation.loadFile.loadNWB import GLOBAL_STIM_NAMES

# Default: looks for protocols containing "Long square"
# For broader matching (e.g., custom protocol names), set:
GLOBAL_STIM_NAMES.stim_inc = ['']  # Accept all protocol names

# To be more specific:
GLOBAL_STIM_NAMES.stim_inc = ['Long square', 'Long Square']
```

---

## 6. Integrating New SVGs into the Viewer

Once new SVGs are generated, three steps are needed to make them visible in the
interactive UMAP viewer.

### Step 1: Organize SVGs into the trace directory

SVGs must follow the path convention:
```
intraDANDI_explorer-master/data/traces/{dandiset}/{subject}/{nwb_filename}.svg
intraDANDI_explorer-master/data/traces/{dandiset}/{subject}/{nwb_filename}_FI.svg
```

If SVGs were generated elsewhere, move them:
```python
import shutil, os

src_svg = "/path/to/generated/sub-636948822_ses-636982248_icephys.nwb.svg"
dst_dir = "intraDANDI_explorer-master/data/traces/000636/sub-636948822/"
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src_svg, dst_dir)
```

### Step 2: Ensure symlinks exist in `results/figures/traces/`

The interactive HTML viewer serves SVGs from `results/figures/traces/`. These are
symlinks to the intraDANDI explorer data:

```bash
cd results/figures/traces/

# These should already exist, but if not:
ln -sf /Users/shreejoy/Github/patch_seq_lee/intraDANDI_explorer-master/data/traces/000636 000636
ln -sf /Users/shreejoy/Github/patch_seq_lee/intraDANDI_explorer-master/data/traces/000630 000630
ln -sf /Users/shreejoy/Github/patch_seq_lee/intraDANDI_explorer-master/data/traces/000228 000228
ln -sf /Users/shreejoy/Github/patch_seq_lee/intraDANDI_explorer-master/data/traces/000337 000337
```

### Step 3: Update `data/patchseq/specimen_to_svg_map.csv`

This CSV maps each specimen to its SVG paths. The viewer reads this file at build
time.

**Columns:**
| Column | Description |
|--------|-------------|
| `specimen_id` | Integer specimen ID from patchseq metadata |
| `ses_id` | DANDI session ID (float, from NWB filename) |
| `dandiset` | Dandiset number (integer, e.g., 636) |
| `dandi_path` | Relative NWB path within dandiset |
| `trace_svg` | Path to voltage trace SVG |
| `fi_svg` | Path to FI curve SVG |
| `file_link` | S3 download URL for the NWB file |

**Example row:**
```csv
636982263,636982248.0,636,sub-636948822/sub-636948822_ses-636982248_icephys.nwb,intraDANDI_explorer-master/data/traces/000636/sub-636948822/sub-636948822_ses-636982248_icephys.nwb.svg,intraDANDI_explorer-master/data/traces/000636/sub-636948822/sub-636948822_ses-636982248_icephys.nwb_FI.svg,https://dandiarchive.s3.amazonaws.com/blobs/4e2/df3/4e2df3b5-64f1-4c1b-90b9-5b2eaf6aceb4
```

To add new entries:
```python
import pandas as pd

svg_map = pd.read_csv("data/patchseq/specimen_to_svg_map.csv")

new_row = {
    "specimen_id": 756894558,
    "ses_id": None,  # Fill if known
    "dandiset": 228,
    "dandi_path": "sub-H18-28-025/sub-H18-28-025_obj-jr1eyz_icephys.nwb",
    "trace_svg": "intraDANDI_explorer-master/data/traces/000228/sub-H18-28-025/sub-H18-28-025_obj-jr1eyz_icephys.nwb.svg",
    "fi_svg": "intraDANDI_explorer-master/data/traces/000228/sub-H18-28-025/sub-H18-28-025_obj-jr1eyz_icephys.nwb_FI.svg",
    "file_link": "",  # Fill with S3 URL if available
}
svg_map = pd.concat([svg_map, pd.DataFrame([new_row])], ignore_index=True)
svg_map.to_csv("data/patchseq/specimen_to_svg_map.csv", index=False)
```

### Step 4: Rebuild the interactive HTML

```bash
python3 scripts/build_interactive_umap.py
```

This reads `specimen_to_svg_map.csv`, matches specimen IDs to cells in the UMAP,
and embeds SVG paths into the HTML. The viewer converts paths like
`intraDANDI_explorer-master/data/traces/000636/...` to
`traces/000636/...` for HTTP serving.

**SVG path convention in the HTML**: `traces/{dandiset}/{subject}/{nwb_filename}.svg`

---

## 7. Current Gaps and Action Items

### Reference file

`data/patchseq/missing_ephys_accounting.csv` documents all 84 cells that have
`has_ephys=True` in the metadata but are missing trace SVGs.

### Breakdown by lab

| Lab Code | Count | Expected Dandiset | Status |
|----------|-------|-------------------|--------|
| lab-29 | 33 | 000337 | `No_NWB_found` -- none of these cells have matching NWBs in DANDI |
| lab-3 | 26 | 000630 or 000636 | `No_NWB_found` -- 16 in L1 dataset, 10 in LeeDalley |
| lab-28 | 13 | 000228 | 3 have donor matches (different slice); 10 have no donor match |
| lab-6 | 10 | 000630 or 000636 | `No_NWB_found` -- 1 in L1 dataset, 9 in LeeDalley |
| lab-26 | 2 | 000630 | `No_NWB_found` |

### The 3 partial matches in DANDI 000228

Three lab-28 cells have donors that exist in dandiset 000228, but the specific
recording session was not found. This means a different slice from the same donor
was uploaded, but not the slice containing our patched cell:

| specimen_id | cell_name | donor | NWB status |
|-------------|-----------|-------|------------|
| 756894558 | H18.28.025.11.07.02 | H18.28.025 | Donor in 228, specific NWB not found |
| 828758585 | H19.28.005.11.05.01 | H19.28.005 | Donor in 228, specific NWB not found |
| 1038298499 | H20.28.023.11.05.02 | H20.28.023 | Donor in 228, specific NWB not found |

The matching analysis for 000228 is documented in
`data/patchseq/lab28_cell_vs_nwb_matching.csv`, which cross-references our cell
names against the NWB files available in the dandiset by donor ID and dat-file
prefix.

### Key observation

All 84 missing cells have **zero computed ephys features** in the joined dataset.
This strongly suggests that the original recordings were either:
1. Never converted to NWB format and uploaded to DANDI
2. Failed quality control and were excluded from the archive
3. Exist under different identifiers that we have not matched

**Recommended action**: Contact the original lab PIs (especially for lab-29 and lab-3)
to determine whether NWB files for these recordings exist outside of DANDI.

---

## 8. Specimen-to-NWB Matching

### How specimen IDs map to NWB files

The matching logic differs by dandiset because each uses different naming
conventions.

### Dandiset 000636 (LeeDalley, AIBS)

**Convention**: NWB files use numeric AIBS session IDs.

```
sub-{subject_id}/sub-{subject_id}_ses-{session_id}_icephys.nwb
```

**Matching**: The `ses_id` in the NWB filename corresponds to the
`patched_cell_container` session ID in the AIBS LIMS system. Specimen IDs map
to session IDs via a direct lookup: `specimen_id` -> `session_id` is a 1:1
relationship in the AIBS database. The session_id appears in the NWB filename.

**Example**:
- Specimen `636982263` -> Session `636982248` -> NWB: `sub-636948822_ses-636982248_icephys.nwb`

**Mapping file**: `data/patchseq/dandi_636_asset_map.csv`
- Columns: `nwb_id` (session ID from filename), `path`, `size_mb`, `identifier` (DANDI UUID)

### Dandiset 000630 (L1 cells, AIBS)

**Convention**: Identical to 000636 -- numeric AIBS IDs.

**Matching**: Same session-ID-based matching as 000636. Many L1 cells also appear
in 000636 (the L1 dataset is a subset).

**Mapping file**: `data/patchseq/dandi_630_asset_map.csv`
- Columns: `ses_id`, `path`, `size_mb`, `identifier`

### Dandiset 000228 (Lab-28, Lee-Bhatt)

**Convention**: NWB files use human-readable donor codes and object hashes.

```
sub-{donor_id}/sub-{donor_id}_obj-{hash}_icephys.nwb
```

**Matching**: This is the most complex dandiset to match because:
1. Our cell names encode the donor and dat-file prefix (e.g., `H18.28.025.11.07.02`)
2. The NWB filename contains the donor (`H18-28-025`) but uses an opaque object hash
3. Each donor may have multiple NWB files (one per recorded cell)
4. The `dat_file` field inside the NWB metadata links to the original recording file

The matching was done by:
1. Listing all NWB files per donor from DANDI
2. Reading the `dat_file` field from each NWB's metadata
3. Matching the dat-file prefix (e.g., `H18.28.025.11.07`) to our cell name
4. Documenting the `group_label` (patched cell container) for disambiguation

**Mapping file**: `data/patchseq/dandi_000228_nwb_metadata.csv`
- Columns: `nwb_fname`, `donor`, `dat_file`, `group_label`, `has_long_square`,
  `n_sweeps`, `protocols`

**Cross-reference file**: `data/patchseq/lab28_cell_vs_nwb_matching.csv`
- Columns: `donor`, `our_specimen_id`, `our_cell_name`, `our_dataset`,
  `nwb_filename`, `n_our_cells`, `n_nwb_files`

### Dandiset 000337 (Lab-29)

**Convention**: Numeric AIBS-style IDs.

**Matching**: Similar to 000636/000630 but the dandiset contains far fewer files
(~21 NWBs). None of the 33 missing lab-29 cells were found.

### Master mapping file

`data/patchseq/specimen_to_dandi_map.csv` is the consolidated mapping from
specimen IDs to DANDI paths:
- Columns: `specimen_id`, `dandi_path`, `dandiset`
- 896 rows (one per successfully matched cell)
- This was used to build `specimen_to_svg_map.csv`

---

## Appendix: File Reference

| File | Location | Description |
|------|----------|-------------|
| `specimen_to_svg_map.csv` | `data/patchseq/` | Maps 968 specimens to trace + FI SVG paths |
| `specimen_to_dandi_map.csv` | `data/patchseq/` | Maps 896 specimens to DANDI NWB paths |
| `missing_ephys_accounting.csv` | `data/patchseq/` | Documents 84 missing-SVG cells |
| `dandi_636_asset_map.csv` | `data/patchseq/` | All NWB assets in dandiset 000636 |
| `dandi_630_asset_map.csv` | `data/patchseq/` | All NWB assets in dandiset 000630 |
| `dandi_000228_nwb_metadata.csv` | `data/patchseq/` | NWB metadata for dandiset 000228 |
| `lab28_cell_vs_nwb_matching.csv` | `data/patchseq/` | Lab-28 cell-to-NWB cross-reference |
| `dandi_scraper.py` | `intraDANDI_explorer-master/dandi_scraper/` | Sam's scraper/analyzer/plotter |
| `build_interactive_umap.py` | `scripts/` | Builds the interactive HTML viewer |
| `traces/` | `intraDANDI_explorer-master/data/` | Pre-generated SVG files |
| `traces/` | `results/figures/` | Symlinks to the above for HTTP serving |
