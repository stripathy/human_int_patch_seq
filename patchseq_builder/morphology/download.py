"""
download.py -- Download SWC morphology files from the Brain Image Library (BIL).

Tries multiple BIL submission directories in order:
  1. Direct URL patterns (6 known submissions with predictable filenames)
  2. 69fe directory listing (non-standard H*_m.swc filenames)

URL templates and directory paths are imported from config.
"""

import html.parser
import urllib.request
import urllib.error
from pathlib import Path

from patchseq_builder.config import BIL_DIRECT_URLS, BIL_69FE_DIR


# ---------------------------------------------------------------------------
# HTML directory listing parser (for 69fe submission)
# ---------------------------------------------------------------------------

class DirListingParser(html.parser.HTMLParser):
    """Extract href links from a BIL directory listing HTML page."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val:
                    self.links.append(val)


# ---------------------------------------------------------------------------
# SWC parsing and validation
# ---------------------------------------------------------------------------

def parse_swc(swc_path) -> dict:
    """Parse SWC file into dict of nodes.

    Parameters
    ----------
    swc_path : str or Path
        Path to the SWC file.

    Returns
    -------
    dict
        Mapping node_id -> {type, x, y, z, radius, parent}.
    """
    nodes = {}
    with open(swc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            nodes[nid] = {
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            }
    return nodes


def is_valid_swc(path) -> bool:
    """Check that a file is a valid SWC (not an HTML 404 page).

    Returns False if the file is missing, too small (<200 bytes),
    or starts with <html>.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 200:
        return False
    with open(path) as f:
        first_line = f.readline()
    return "<html>" not in first_line.lower()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_url(url, out_path, timeout=15) -> bool:
    """Download a URL to a file. Returns True on success (valid SWC)."""
    try:
        urllib.request.urlretrieve(url, out_path)
        return is_valid_swc(out_path)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        out_path = Path(out_path)
        if out_path.exists():
            out_path.unlink()
        return False


def download_swc_direct(specimen_id, out_path):
    """Try downloading SWC from known direct URL patterns.

    Iterates through ``config.BIL_DIRECT_URLS`` in order.

    Returns
    -------
    tuple
        (url, source_name) on success, or (None, None).
    """
    sid = str(specimen_id)
    out_path = Path(out_path)
    for name, url_template in BIL_DIRECT_URLS:
        url = url_template.format(sid=sid)
        if download_url(url, out_path):
            return url, name
        # Clean up failed download
        if out_path.exists():
            out_path.unlink()
    return None, None


def download_swc_69fe(specimen_id, out_path):
    """Download SWC from the 69fe submission (non-standard filenames).

    Lists the directory on BIL, finds the .swc file (preferring *_m.swc
    manual reconstructions), and downloads it.

    Returns
    -------
    tuple
        (url, source_name) on success, or (None, None).
    """
    sid = str(specimen_id)
    out_path = Path(out_path)
    dir_url = BIL_69FE_DIR.format(sid=sid)

    try:
        req = urllib.request.Request(dir_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            page_html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None, None

    # Parse directory listing for .swc files
    parser = DirListingParser()
    parser.feed(page_html)
    swc_files = [link for link in parser.links if link.endswith(".swc")]

    if not swc_files:
        return None, None

    # Prefer _m.swc (manual reconstruction) if available
    m_files = [f for f in swc_files if f.endswith("_m.swc")]
    target = m_files[0] if m_files else swc_files[0]

    file_url = dir_url + target
    if download_url(file_url, out_path):
        return file_url, "69fe_dirlist"
    if out_path.exists():
        out_path.unlink()
    return None, None


def download_swc(specimen_id, out_path) -> tuple:
    """Try all download strategies for a specimen's SWC file.

    Strategy order:
      1. Direct URL patterns (fast, predictable filenames)
      2. 69fe directory listing (slower, handles non-standard names)

    Parameters
    ----------
    specimen_id : int or str
        The specimen ID to download.
    out_path : str or Path
        Where to save the downloaded SWC file.

    Returns
    -------
    tuple
        (url, source) on success, or (None, None) if not found.
    """
    # Strategy 1: Direct URL patterns (fast)
    url, source = download_swc_direct(specimen_id, out_path)
    if url:
        return url, source

    # Strategy 2: 69fe directory listing (slower, but handles non-standard names)
    url, source = download_swc_69fe(specimen_id, out_path)
    if url:
        return url, source

    return None, None
