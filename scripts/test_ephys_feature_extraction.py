"""
Test script: Verify that ipfx feature extraction from NWB files reproduces
the values in the existing LeeDalley_ephys_fx.csv.

For each cached NWB file that has a matching specimen in the CSV, this script:
1. Loads the NWB file using ipfx's create_ephys_data_set
2. Runs extract_data_set_features to get cell-level features
3. Compares key features against the CSV values

Key findings:
- tau: ipfx returns ms, CSV stores seconds (x 1/1000 conversion needed)
- latency: ipfx cell_record 'latency' = latency_hero (NOT latency_rheo)
- short_square features: may differ slightly due to sweep selection (mean vs single)
- Core subthreshold features (sag, tau, ri, vrest, fi_fit_slope, vm_for_sag,
  rheobase_i) all reproduce exactly within floating-point precision.
"""

import os
import warnings
import numpy as np
import pandas as pd
from ipfx.dataset.create import create_ephys_data_set
from ipfx.data_set_features import extract_data_set_features

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE = "/Users/shreejoy/Github/patch_seq_lee"
NWB_CACHE = os.path.join(BASE, "data/nwb_cache/000636")
EPHYS_CSV = os.path.join(BASE, "data/patchseq/LeeDalley_ephys_fx.csv")
MAPPING_CSV = os.path.join(BASE, "data/patchseq/specimen_to_dandi_map.csv")

# -------------------------------------------------------------------
# Feature name mapping: ipfx cell_record key -> CSV column name
# -------------------------------------------------------------------
# Core features (these should match exactly)
CORE_FEATURE_MAP = {
    "sag":                          "sag",
    "tau":                          "tau",           # NOTE: ipfx=ms, CSV=seconds
    "ri":                           "input_resistance",
    "vrest":                        "v_baseline",
    "f_i_curve_slope":              "fi_fit_slope",
    "vm_for_sag":                   "vm_for_sag",
    "threshold_i_long_square":      "rheobase_i",
}

# Secondary features (may differ due to sweep selection / averaging differences)
SECONDARY_FEATURE_MAP = {
    "upstroke_downstroke_ratio_ramp": "upstroke_downstroke_ratio_ramp",
    "threshold_v_ramp":             "threshold_v_ramp",
    "threshold_v_long_square":      "threshold_v_rheo",
    "latency":                      "latency_hero",  # ipfx 'latency' = hero sweep latency
    "adaptation":                   "adapt_hero",
}

# All features combined
FEATURE_MAP = {**CORE_FEATURE_MAP, **SECONDARY_FEATURE_MAP}

# Features that need unit conversion
UNIT_CONVERSIONS = {
    "tau": 1.0 / 1000.0,  # ipfx returns ms, CSV stores seconds
}


def find_matching_specimens():
    """Find specimens that have both a cached NWB file and an entry in the ephys CSV."""
    mapping = pd.read_csv(MAPPING_CSV)
    ephys = pd.read_csv(EPHYS_CSV)

    # Build a set of actually cached NWB filenames
    cached_nwbs = set()
    for subject_dir in os.listdir(NWB_CACHE):
        subject_path = os.path.join(NWB_CACHE, subject_dir)
        if os.path.isdir(subject_path):
            for f in os.listdir(subject_path):
                if f.endswith("_icephys.nwb"):
                    cached_nwbs.add(f)

    # Add NWB filename to mapping
    mapping["nwb_filename"] = mapping["dandi_path"].apply(os.path.basename)
    mapping["is_cached"] = mapping["nwb_filename"].isin(cached_nwbs)

    # Filter to cached specimens that are also in the ephys CSV
    cached_mapping = mapping[mapping["is_cached"]]
    ephys_ids = set(ephys["specimen_id"].values)
    matched = cached_mapping[cached_mapping["specimen_id"].isin(ephys_ids)].copy()

    # Build full path to NWB
    matched["nwb_path"] = matched["dandi_path"].apply(
        lambda p: os.path.join(NWB_CACHE, p)
    )

    return matched, ephys


def extract_and_compare(specimen_id, nwb_path, ephys_row):
    """Extract features from NWB and compare to CSV values."""
    try:
        ds = create_ephys_data_set(nwb_path)
        result = extract_data_set_features(ds)
        cell_record = result[2]
    except Exception as e:
        return {"specimen_id": specimen_id, "error": str(e)}

    comparisons = []
    for ipfx_key, csv_key in FEATURE_MAP.items():
        ipfx_val = cell_record.get(ipfx_key)
        csv_val = ephys_row.get(csv_key)

        if ipfx_val is None or pd.isna(csv_val):
            comparisons.append({
                "feature": csv_key,
                "ipfx_key": ipfx_key,
                "ipfx_val": ipfx_val,
                "csv_val": csv_val if not pd.isna(csv_val) else None,
                "match": "skip (missing)",
            })
            continue

        # Apply unit conversion if needed
        conversion = UNIT_CONVERSIONS.get(ipfx_key, 1.0)
        ipfx_converted = float(ipfx_val) * conversion
        csv_float = float(csv_val)

        # Check relative match (1% tolerance)
        denom = max(abs(csv_float), abs(ipfx_converted), 1e-10)
        rel_diff = abs(ipfx_converted - csv_float) / denom

        comparisons.append({
            "feature": csv_key,
            "ipfx_key": ipfx_key,
            "ipfx_val": ipfx_converted,
            "csv_val": csv_float,
            "rel_diff": rel_diff,
            "match": rel_diff < 0.01,
        })

    return {
        "specimen_id": specimen_id,
        "comparisons": comparisons,
        "error": None,
    }


def main():
    print("=" * 90)
    print("EPHYS FEATURE EXTRACTION REPRODUCIBILITY TEST")
    print("=" * 90)

    matched, ephys = find_matching_specimens()
    print(f"\nFound {len(matched)} specimens with both cached NWB and ephys CSV entry")

    # Test a subset (first N specimens)
    N_TEST = 8
    test_specimens = matched.head(N_TEST)
    print(f"Testing {N_TEST} specimens...\n")

    all_results = []
    for _, row in test_specimens.iterrows():
        specimen_id = row["specimen_id"]
        nwb_path = row["nwb_path"]
        ephys_row = ephys[ephys["specimen_id"] == specimen_id].iloc[0]

        print(f"\n{'─' * 90}")
        print(f"Specimen: {specimen_id}")
        print(f"NWB: {os.path.basename(nwb_path)}")
        print(f"{'─' * 90}")

        result = extract_and_compare(specimen_id, nwb_path, ephys_row)

        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            all_results.append(result)
            continue

        print(f"  {'Feature':<40} {'ipfx value':<20} {'CSV value':<20} {'Rel Diff':<12} {'Match'}")
        print(f"  {'-' * 100}")

        n_match = 0
        n_skip = 0
        n_fail = 0

        for comp in result["comparisons"]:
            feature = comp["feature"]
            ipfx_val = comp["ipfx_val"]
            csv_val = comp["csv_val"]
            match = comp["match"]

            if match == "skip (missing)":
                ipfx_str = str(ipfx_val) if ipfx_val is not None else "N/A"
                csv_str = str(csv_val) if csv_val is not None else "N/A"
                print(f"  {feature:<40} {ipfx_str:<20} {csv_str:<20} {'---':<12} SKIP")
                n_skip += 1
            else:
                rel_diff = comp["rel_diff"]
                status = "OK" if match else "MISMATCH"
                if not match:
                    n_fail += 1
                else:
                    n_match += 1
                print(f"  {feature:<40} {ipfx_val:<20.8g} {csv_val:<20.8g} {rel_diff:<12.2e} {status}")

        print(f"\n  Summary: {n_match} matched, {n_fail} mismatched, {n_skip} skipped")
        all_results.append(result)

    # -------------------------------------------------------------------
    # Overall summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 90}")

    total_match = 0
    total_fail = 0
    total_skip = 0
    total_error = 0

    for res in all_results:
        if res.get("error"):
            total_error += 1
            continue
        for comp in res["comparisons"]:
            if comp["match"] == "skip (missing)":
                total_skip += 1
            elif comp["match"]:
                total_match += 1
            else:
                total_fail += 1

    print(f"  Specimens tested: {len(all_results)}")
    print(f"  Specimens with errors: {total_error}")
    print(f"  Feature comparisons matched: {total_match}")
    print(f"  Feature comparisons mismatched: {total_fail}")
    print(f"  Feature comparisons skipped (missing data): {total_skip}")

    if total_fail == 0 and total_error == 0:
        print("\n  RESULT: All features reproduced successfully from NWB files!")
    else:
        print(f"\n  RESULT: {total_fail} mismatches and {total_error} errors found.")

    print()


if __name__ == "__main__":
    main()
