"""
orientation.py -- Detect and fix inverted morphology orientations.

In the SWC files downloaded from BIL, 12 cells have their y-axis inverted:
dendrites extend below the soma and the axon extends above, which is the
opposite of normal cortical neuron orientation (dendrites toward pia = up,
axon toward white matter = down).

This module provides:
  - A hardcoded list of the 12 known inverted specimen IDs
  - A detection function that scans an SWC directory for inverted cells
  - A flip function that mirrors y-coordinates around the soma
"""

from pathlib import Path

from patchseq_builder.morphology.download import parse_swc


# ── Known inverted specimen IDs ──────────────────────────────────────────
# These 12 cells were identified by manual inspection: their dendrites
# (type 3, basal) are mostly above the soma y and axon (type 3/4) is
# below, or vice versa, inconsistent with the expected pia-up orientation.

INVERTED_SPECIMEN_IDS = [
    1002962250,
    1009830894,
    548480353,
    653810044,
    707795387,
    737549661,
    756894558,
    811932153,
    811953283,
    855783147,
    941862585,
    966905488,
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
