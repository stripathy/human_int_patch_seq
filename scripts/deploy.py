"""
deploy.py -- Build a lightweight deployment bundle for the patch-seq viewer.

Parses the built HTML to find only the SVG files actually referenced,
then copies them (resolving symlinks) into a self-contained deploy/ directory
suitable for Netlify or any static hosting.

Usage:
    python3 scripts/deploy.py
    python3 scripts/deploy.py --clean   # remove deploy/ first
"""

import argparse
import os
import re
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
HTML_SOURCE = FIGURES_DIR / "patchseq_umap_interactive.html"
DEPLOY_DIR = PROJECT_ROOT / "deploy"


def extract_referenced_svgs(html_text: str) -> tuple[set[str], set[str]]:
    """Extract SVG paths referenced in the HTML customdata arrays.

    Returns (trace_paths, morph_paths) -- sets of relative paths like
    'traces/000636/sub-.../....svg' and 'morphology_svgs/....svg'.
    """
    trace_paths = set(re.findall(r'traces/\d+/sub-[^"\\]+\.svg', html_text))
    morph_paths = set(re.findall(r'morphology_svgs/[^"\\]+\.svg', html_text))
    return trace_paths, morph_paths


def copy_svgs(svg_paths: set[str], src_root: Path, dst_root: Path) -> tuple[int, int]:
    """Copy SVG files from src_root to dst_root, preserving directory structure.

    Resolves symlinks so the deployment directory is self-contained.
    Returns (n_copied, n_missing).
    """
    n_copied = 0
    n_missing = 0
    for rel_path in sorted(svg_paths):
        src = src_root / rel_path
        dst = dst_root / rel_path
        # Resolve symlinks to get the actual file
        try:
            src_resolved = src.resolve(strict=True)
        except (OSError, FileNotFoundError):
            n_missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_resolved, dst)
        n_copied += 1
    return n_copied, n_missing


def dir_size_mb(path: Path) -> float:
    """Compute total size of a directory in MB."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


def main():
    parser = argparse.ArgumentParser(description="Build deployment bundle")
    parser.add_argument("--clean", action="store_true",
                        help="Remove deploy/ directory before building")
    args = parser.parse_args()

    t0 = time.time()

    if not HTML_SOURCE.exists():
        raise FileNotFoundError(
            f"Viewer HTML not found: {HTML_SOURCE}\n"
            f"Run the pipeline first: python3 scripts/build_patchseq_viewer.py"
        )

    # Clean if requested
    if args.clean and DEPLOY_DIR.exists():
        print(f"Cleaning {DEPLOY_DIR}...")
        shutil.rmtree(DEPLOY_DIR)

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Parse HTML for referenced SVGs
    print("Parsing HTML for referenced SVG paths...")
    html_text = HTML_SOURCE.read_text()
    trace_paths, morph_paths = extract_referenced_svgs(html_text)
    print(f"  Trace SVGs referenced: {len(trace_paths)}")
    print(f"  Morphology SVGs referenced: {len(morph_paths)}")

    # 2. Copy HTML as index.html
    print("\nCopying HTML...")
    shutil.copy2(HTML_SOURCE, DEPLOY_DIR / "index.html")

    # 3. Copy referenced SVGs
    print("Copying trace SVGs...")
    n_trace, n_trace_miss = copy_svgs(trace_paths, FIGURES_DIR, DEPLOY_DIR)
    print(f"  Copied: {n_trace}, missing: {n_trace_miss}")

    print("Copying morphology SVGs...")
    n_morph, n_morph_miss = copy_svgs(morph_paths, FIGURES_DIR, DEPLOY_DIR)
    print(f"  Copied: {n_morph}, missing: {n_morph_miss}")

    # 4. Write netlify.toml
    print("Writing netlify.toml...")
    netlify_toml = DEPLOY_DIR / "netlify.toml"
    netlify_toml.write_text("""\
[build]
  publish = "."

[[headers]]
  for = "/*.svg"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/morphology_svgs/*.svg"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/traces/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/index.html"
  [headers.values]
    Cache-Control = "public, max-age=3600"
""")

    # 5. Summary
    elapsed = time.time() - t0
    total_files = 1 + n_trace + n_morph + 1  # HTML + traces + morphs + netlify.toml
    total_mb = dir_size_mb(DEPLOY_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Deployment bundle ready: {DEPLOY_DIR}")
    print(f"{'=' * 60}")
    html_mb = (DEPLOY_DIR / "index.html").stat().st_size / 1e6
    print(f"  HTML:           1 file   ({html_mb:.1f} MB)")
    print(f"  Trace SVGs:     {n_trace} files ({dir_size_mb(DEPLOY_DIR / 'traces'):.1f} MB)")
    print(f"  Morphology SVGs:{n_morph} files ({dir_size_mb(DEPLOY_DIR / 'morphology_svgs'):.1f} MB)")
    print(f"  Config:         1 file")
    print(f"  ---")
    print(f"  Total:          {total_files} files ({total_mb:.1f} MB)")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"\nTo test locally:")
    print(f"  cd {DEPLOY_DIR} && python3 -m http.server 8765")
    print(f"\nTo deploy to Netlify:")
    print(f"  netlify deploy --dir={DEPLOY_DIR} --prod")


if __name__ == "__main__":
    main()
