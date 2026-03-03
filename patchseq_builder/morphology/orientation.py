"""
orientation.py -- Detect and fix inverted morphology orientations.

Some SWC files downloaded from BIL have their y-axis inverted. We apply
flipping *conservatively*, only to Sst and Sst Chodl cells where the
dendrite-up / axon-down rule is unambiguous. Lamp5 and Pax6 (Layer 1)
interneurons are excluded because they normally have ascending axons that
project toward pia, so the heuristic would incorrectly flag their
biologically correct orientation.

This module provides:
  - A curated list of inverted specimen IDs (Sst / Sst Chodl only)
  - A detection function that scans an SWC directory for inverted cells
  - A flip function that mirrors y-coordinates around the soma
"""

from pathlib import Path

from patchseq_builder.morphology.download import parse_swc


# ── Known inverted specimen IDs (Sst / Sst Chodl only) ──────────────────
# Only Sst and Sst Chodl cells are flipped. These cell types have
# descending axons, so the dendrite-up / axon-down rule is reliable.
#
# Lamp5 and Pax6 cells were previously on this list but are excluded
# because L1 interneurons normally project axons toward pia (ascending),
# making the heuristic unreliable for those types.
#
# Excluded Lamp5/Pax6 specimens (NOT flipped):
#   548480353  (Pax6, PAX6 CDH12)
#   653810044  (Lamp5, SST NMBR/ADARB2+)
#   707795387  (Pax6, PAX6 CDH12)
#   737549661  (Lamp5, LAMP5 LCP2)
#   756894558  (Lamp5, SST NMBR/ADARB2+)
#   811932153  (Lamp5, SST NMBR/ADARB2+)
#   811953283  (Pax6, PAX6 CDH12)
#   1009830894 (Lamp5, SST NMBR/ADARB2+)

INVERTED_SPECIMEN_IDS = [
    855783147,   # Sst (Sst_25), LeeDalley, SST CALB1
    893647190,   # Sst (Sst_25), LeeDalley — visually confirmed inverted
    941707648,   # Sst (Sst_25), LeeDalley — visually confirmed inverted
    941862585,   # Sst Chodl (Sst Chodl_2), LeeDalley, SST CALB1
    966905488,   # Sst (Sst_25), LeeDalley, SST CALB1
    1002962250,  # Sst (Sst_25), LeeDalley, SST CALB1
]


def detect_inverted_cells(swc_dir) -> list:
    """Detect cells with inverted y-orientation in a directory of SWC files.

    A cortical neuron is considered inverted if the majority of dendrite
    length (type 3 = basal dendrite) extends below the soma y-coordinate
    and the majority of axon length (type 2 = axon) extends above the
    soma y-coordinate. This is the opposite of normal cortical orientation
    where dendrites project toward pia (up/positive y) and axon descends
    toward white matter (down/negative y).

    Parameters
    ----------
    swc_dir : str or Path
        Directory containing SWC files named {specimen_id}_upright.swc.

    Returns
    -------
    list
        List of specimen_ids (int) where the morphology appears inverted.
    """
    swc_dir = Path(swc_dir)
    inverted = []

    for swc_path in sorted(swc_dir.glob("*_upright.swc")):
        # Extract specimen_id from filename
        stem = swc_path.stem  # e.g. "548480353_upright"
        try:
            specimen_id = int(stem.split("_")[0])
        except ValueError:
            continue

        nodes = parse_swc(swc_path)
        if not nodes:
            continue

        # Find soma y
        soma_y = None
        for nid, node in nodes.items():
            if node["type"] == 1:
                soma_y = node["y"]
                break
        if soma_y is None:
            continue

        # Compute fraction of dendrite nodes above vs below soma
        dendrite_above = 0
        dendrite_below = 0
        axon_above = 0
        axon_below = 0

        for nid, node in nodes.items():
            if node["type"] == 3:  # basal dendrite
                if node["y"] > soma_y:
                    dendrite_above += 1
                else:
                    dendrite_below += 1
            elif node["type"] == 2:  # axon
                if node["y"] > soma_y:
                    axon_above += 1
                else:
                    axon_below += 1

        # Normal orientation: dendrites mostly above soma, axon mostly below.
        # Inverted: dendrites mostly below soma AND axon mostly above.
        total_dendrite = dendrite_above + dendrite_below
        total_axon = axon_above + axon_below

        if total_dendrite == 0 and total_axon == 0:
            continue

        dendrite_frac_below = (dendrite_below / total_dendrite
                               if total_dendrite > 0 else 0)
        axon_frac_above = (axon_above / total_axon
                           if total_axon > 0 else 0)

        # Flag as inverted if dendrites are mostly below AND axon is mostly above
        # (or if there's only one compartment and it's in the wrong direction)
        is_inverted = False
        if total_dendrite > 0 and total_axon > 0:
            is_inverted = dendrite_frac_below > 0.6 and axon_frac_above > 0.6
        elif total_dendrite > 0:
            is_inverted = dendrite_frac_below > 0.7
        elif total_axon > 0:
            is_inverted = axon_frac_above > 0.7

        if is_inverted:
            inverted.append(specimen_id)

    return inverted


def flip_swc_y(nodes: dict) -> dict:
    """Mirror all y-coordinates around the soma y-coordinate.

    This corrects inverted morphologies by reflecting every node's
    y-coordinate across the soma position: y_new = 2 * soma_y - y_old.

    Parameters
    ----------
    nodes : dict
        SWC nodes as returned by parse_swc(): node_id -> {type, x, y, z,
        radius, parent}.

    Returns
    -------
    dict
        New nodes dict with corrected y-coordinates. The original dict
        is not modified.

    Raises
    ------
    ValueError
        If no soma node (type 1) is found.
    """
    # Find soma y
    soma_y = None
    for nid, node in nodes.items():
        if node["type"] == 1:
            soma_y = node["y"]
            break
    if soma_y is None:
        raise ValueError("No soma node (type 1) found in SWC data")

    # Mirror all y-coordinates around soma
    flipped = {}
    for nid, node in nodes.items():
        flipped[nid] = {
            "type": node["type"],
            "x": node["x"],
            "y": 2.0 * soma_y - node["y"],
            "z": node["z"],
            "radius": node["radius"],
            "parent": node["parent"],
        }
    return flipped
