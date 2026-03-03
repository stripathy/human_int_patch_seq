"""
naming.py — Single source of truth for cell type name normalization.

All subclass/supertype name mappings are defined here. Other modules import
from this file instead of defining inline name maps.

Name conventions:
  - "canonical": Allen Institute SEA-AD naming (e.g. "Sst", "Pvalb", "Lamp5")
  - "display": Human-readable for plots (e.g. "Lamp5 Lhx6" not "Lamp5_Lhx6")
  - "original": Names as they appear in raw data sources (vary by dataset)
"""

# ============================================================================
# Subclass name normalization: various source formats → canonical Allen names
# ============================================================================

# Original label formats found in different data sources
SUBCLASS_TO_CANONICAL = {
    # ALL-CAPS (L1 dataset, scANVI raw)
    "SST": "Sst",
    "PVALB": "Pvalb",
    "VIP": "Vip",
    "LAMP5": "Lamp5",
    "PAX6": "Pax6",
    "SNCG": "Sncg",
    # Mixed variants
    "Lamp5/Pax6": "Lamp5",
    "LAMP5/PAX6/Other": "Lamp5",
    # Underscore-joined (scANVI output)
    "Lamp5_Lhx6": "Lamp5 Lhx6",
    "Sst Chodl": "Sst Chodl",
    # Already canonical — pass through
    "Sst": "Sst",
    "Pvalb": "Pvalb",
    "Vip": "Vip",
    "Lamp5": "Lamp5",
    "Pax6": "Pax6",
    "Sncg": "Sncg",
    "Chandelier": "Chandelier",
    "Lamp5 Lhx6": "Lamp5 Lhx6",
    "L2/3 IT": "L2/3 IT",
    "L4 IT": "L4 IT",
    "L5 IT": "L5 IT",
    "L5 ET": "L5 ET",
    "L6 IT": "L6 IT",
    "L6 CT": "L6 CT",
    "L6b": "L6b",
}

# For display in figures: canonical → display-friendly
CANONICAL_TO_DISPLAY = {
    "Lamp5_Lhx6": "Lamp5 Lhx6",
}


def normalize_subclass(name):
    """Convert any subclass name variant to canonical Allen form.

    Examples:
        normalize_subclass("SST") → "Sst"
        normalize_subclass("LAMP5/PAX6/Other") → "Lamp5"
        normalize_subclass("Lamp5_Lhx6") → "Lamp5 Lhx6"
        normalize_subclass("Sst") → "Sst" (no change)
    """
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return None
    name = str(name).strip()
    return SUBCLASS_TO_CANONICAL.get(name, name)


def display_subclass(name):
    """Convert canonical subclass name to display format.

    Mainly fixes underscore-joined names like Lamp5_Lhx6 → Lamp5 Lhx6.
    """
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return None
    name = str(name).strip()
    return CANONICAL_TO_DISPLAY.get(name, name)


def normalize_series(series):
    """Normalize a pandas Series of subclass names to canonical form."""
    import pandas as pd
    return series.map(lambda x: normalize_subclass(x) if pd.notna(x) else x)


# ============================================================================
# Dendrite type normalization (L1 dataset)
# ============================================================================

DENDRITE_TYPE_PREFIX = "dendrite type - "


def normalize_dendrite_type(name):
    """Remove the 'dendrite type - ' prefix from L1 dendrite types."""
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return None
    name = str(name).strip()
    if name.startswith(DENDRITE_TYPE_PREFIX):
        return name[len(DENDRITE_TYPE_PREFIX):]
    return name


# ============================================================================
# Dataset identifiers
# ============================================================================

DATASET_LEEDALLEY = "LeeDalley"
DATASET_L1 = "L1"
DATASET_BOTH = "both"  # cells present in both datasets (43 cells)
