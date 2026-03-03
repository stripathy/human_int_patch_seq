"""
symlinks.py -- Create symlinks from results/figures/traces/ to intraDANDI trace directories.

The interactive HTML viewer serves files via HTTP from results/figures/.
Trace SVGs live under intraDANDI_explorer-master/data/traces/{dandiset}/...
so we create symlinks like:

    results/figures/traces/000636 -> <project>/intraDANDI_explorer-master/data/traces/000636
    results/figures/traces/000630 -> <project>/intraDANDI_explorer-master/data/traces/000630
    etc.

This allows the viewer to reference trace SVGs as traces/000636/sub-.../file.svg
relative to the HTTP server root (results/figures/).
"""

from pathlib import Path

from patchseq_builder.config import (
    INTRADANDI_TRACES_DIR,
    DANDI_DANDISETS,
    TRACE_SYMLINK_DIR,
)


def ensure_trace_symlinks():
    """Create/verify symlinks from results/figures/traces/ to intraDANDI trace dirs.

    For each dandiset in config.DANDI_DANDISETS, creates a symlink at
    results/figures/traces/{dandiset_padded} pointing to the corresponding
    directory under intraDANDI_explorer-master/data/traces/.

    Skips dandisets that don't have trace data or where the symlink already
    exists and points to the correct target. Warns if a symlink exists but
    points to the wrong target.
    """
    TRACE_SYMLINK_DIR.mkdir(parents=True, exist_ok=True)
    n_created = 0
    n_existing = 0
    n_skipped = 0

    for dandiset_id in DANDI_DANDISETS:
        dandiset_str = f"000{dandiset_id}" if len(str(dandiset_id)) == 3 else str(dandiset_id)
        source_dir = INTRADANDI_TRACES_DIR / dandiset_str
        link_path = TRACE_SYMLINK_DIR / dandiset_str

        # Check if source directory exists and has SVG content
        if not source_dir.exists():
            print(f"  Skipping {dandiset_str}: source dir not found "
                  f"({source_dir})")
            n_skipped += 1
            continue

        # Check for any SVGs in the source directory
        has_svgs = any(source_dir.rglob("*.svg"))
        if not has_svgs:
            print(f"  Skipping {dandiset_str}: no SVGs found in {source_dir}")
            n_skipped += 1
            continue

        # Create or verify symlink
        if link_path.is_symlink():
            current_target = link_path.resolve()
            expected_target = source_dir.resolve()
            if current_target == expected_target:
                n_existing += 1
                continue
            else:
                print(f"  WARNING: {link_path} points to {current_target}, "
                      f"expected {expected_target}. Replacing.")
                link_path.unlink()

        if link_path.exists():
            print(f"  WARNING: {link_path} exists but is not a symlink. "
                  f"Skipping.")
            n_skipped += 1
            continue

        # Create absolute symlink
        link_path.symlink_to(source_dir.resolve())
        print(f"  Created symlink: {dandiset_str} -> {source_dir}")
        n_created += 1

    total = n_created + n_existing + n_skipped
    print(f"  Trace symlinks: {n_created} created, {n_existing} already ok, "
          f"{n_skipped} skipped ({total} total)")
