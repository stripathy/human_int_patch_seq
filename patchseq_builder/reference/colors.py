"""
colors.py -- Allen Institute color scheme loading for subclass and supertype.

Loads standardized colors from the Allen Institute deployment JSON
(SCZ_Xenium index.json). Falls back to hardcoded GABAergic interneuron
colors if the external repo is not available.

Usage:
    from patchseq_builder.reference.colors import load_allen_colors
    subclass_colors, supertype_colors = load_allen_colors()
"""

import json
import warnings

from patchseq_builder.config import ALLEN_COLORS_JSON

# ============================================================================
# Fallback colors for GABAergic interneuron subclasses
# These are the Allen Institute standard colors, hardcoded so the pipeline
# works even without the SCZ_Xenium repo cloned locally.
# ============================================================================

FALLBACK_SUBCLASS_COLORS = {
    "Chandelier": "#F641A8",
    "Lamp5": "#DA808C",
    "Lamp5 Lhx6": "#935F50",
    "Pax6": "#71238C",
    "Pvalb": "#D93137",
    "Sncg": "#DF70FF",
    "Sst": "#FF9900",
    "Sst Chodl": "#B1B10C",
    "Vip": "#A45FBF",
}

# Supertype fallback: subclass color with slight variation per supertype.
# Only includes the most common GABAergic supertypes seen in patch-seq data.
FALLBACK_SUPERTYPE_COLORS = {
    "Pvalb_1": "#D93137",
    "Pvalb_2": "#C42D33",
    "Pvalb_3": "#E0484E",
    "Sst_1": "#FF9900",
    "Sst_2": "#E68A00",
    "Sst_3": "#FFB347",
    "Vip_1": "#A45FBF",
    "Vip_2": "#9350B0",
    "Vip_3": "#B570CE",
    "Lamp5_1": "#DA808C",
    "Lamp5_2": "#C8707C",
    "Lamp5_3": "#E8909C",
    "Sncg_1": "#DF70FF",
    "Sncg_2": "#D060EE",
    "Chandelier_1": "#F641A8",
    "Chandelier_2": "#E73599",
}


def load_allen_colors(json_path=None) -> tuple[dict, dict]:
    """Load Allen Institute color scheme for subclass and supertype.

    Reads the deployment index.json from the SCZ_Xenium repo, which contains
    ``subclass_colors`` and ``supertype_colors`` dicts mapping cell type names
    to hex color strings.

    Parameters
    ----------
    json_path : str or Path, optional
        Path to the Allen colors JSON file. Defaults to
        ``config.ALLEN_COLORS_JSON``.

    Returns
    -------
    subclass_colors : dict
        Mapping of subclass name -> hex color string.
    supertype_colors : dict
        Mapping of supertype name -> hex color string.

    Notes
    -----
    If the JSON file is not found, falls back to hardcoded GABAergic
    interneuron colors and issues a warning. The fallback colors cover the
    subclasses and common supertypes relevant to patch-seq analysis but do
    not include non-neuronal or excitatory types.
    """
    if json_path is None:
        json_path = ALLEN_COLORS_JSON

    try:
        with open(json_path) as f:
            data = json.load(f)
        subclass_colors = data["subclass_colors"]
        supertype_colors = data["supertype_colors"]
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(
            f"Could not load Allen colors from {json_path}: {e}. "
            f"Using fallback GABAergic interneuron colors.",
            stacklevel=2,
        )
        subclass_colors = FALLBACK_SUBCLASS_COLORS.copy()
        supertype_colors = FALLBACK_SUPERTYPE_COLORS.copy()

    return subclass_colors, supertype_colors
