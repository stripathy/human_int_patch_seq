"""
validate.py — Validation utilities for pipeline stage outputs.

Each stage can call validation functions to verify its outputs before
the next stage runs. This catches data integrity issues early.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Summary of validation checks for a pipeline stage output."""
    stage: str
    file_path: str
    checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self):
        return len(self.errors) == 0

    def add_check(self, msg):
        self.checks.append(msg)

    def add_warning(self, msg):
        self.warnings.append(f"⚠️  {msg}")

    def add_error(self, msg):
        self.errors.append(f"❌ {msg}")

    def print_summary(self):
        status = "✅ PASS" if self.ok else "❌ FAIL"
        print(f"\n{'─' * 60}")
        print(f"Validation: {self.stage} [{status}]")
        print(f"  File: {self.file_path}")
        for c in self.checks:
            print(f"  ✓ {c}")
        for w in self.warnings:
            print(f"  {w}")
        for e in self.errors:
            print(f"  {e}")
        print(f"{'─' * 60}")

    def assert_ok(self):
        """Raise SystemExit if validation failed."""
        if not self.ok:
            self.print_summary()
            sys.exit(1)


def validate_metadata(csv_path, expected_min_cells=1100):
    """Validate the harmonized metadata CSV."""
    report = ValidationReport("Metadata", str(csv_path))

    if not Path(csv_path).exists():
        report.add_error(f"File not found: {csv_path}")
        return report

    df = pd.read_csv(csv_path)
    report.add_check(f"{len(df)} cells loaded")

    # Row count
    if len(df) < expected_min_cells:
        report.add_error(f"Only {len(df)} cells, expected >= {expected_min_cells}")

    # Required columns
    required = ["specimen_id", "cell_name", "donor", "dataset",
                 "has_ephys", "has_morphology"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        report.add_error(f"Missing columns: {missing_cols}")
    else:
        report.add_check(f"All required columns present")

    # No duplicate specimen_ids
    dups = df["specimen_id"].duplicated().sum()
    if dups > 0:
        report.add_error(f"{dups} duplicate specimen_ids")
    else:
        report.add_check(f"No duplicate specimen_ids")

    # scANVI coverage
    if "subclass_scANVI" in df.columns:
        n_scanvi = df["subclass_scANVI"].notna().sum()
        pct = 100 * n_scanvi / len(df)
        report.add_check(f"scANVI coverage: {n_scanvi}/{len(df)} ({pct:.1f}%)")
        if pct < 90:
            report.add_warning(f"scANVI coverage below 90% ({pct:.1f}%)")

    # Dataset distribution
    for ds in ["LeeDalley", "L1", "both"]:
        n = (df["dataset"] == ds).sum()
        report.add_check(f"  {ds}: {n} cells")

    return report


def validate_ephys(csv_path, expected_min_cells=900):
    """Validate the harmonized ephys features CSV."""
    report = ValidationReport("Ephys Features", str(csv_path))

    if not Path(csv_path).exists():
        report.add_error(f"File not found: {csv_path}")
        return report

    df = pd.read_csv(csv_path)
    report.add_check(f"{len(df)} cells with ephys")

    if len(df) < expected_min_cells:
        report.add_error(f"Only {len(df)} cells, expected >= {expected_min_cells}")

    # Check key features exist
    key_features = ["sag", "tau", "input_resistance"]
    for feat in key_features:
        if feat in df.columns:
            n_valid = df[feat].notna().sum()
            report.add_check(f"  {feat}: {n_valid} non-NaN")
        else:
            report.add_warning(f"  {feat}: column missing")

    return report


def validate_h5ad(h5ad_path, expected_min_cells=1100, expected_min_genes=50000):
    """Validate the combined h5ad file."""
    import scanpy as sc

    report = ValidationReport("Combined h5ad", str(h5ad_path))

    if not Path(h5ad_path).exists():
        report.add_error(f"File not found: {h5ad_path}")
        return report

    adata = sc.read_h5ad(h5ad_path)
    report.add_check(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if adata.shape[0] < expected_min_cells:
        report.add_error(f"Only {adata.shape[0]} cells, expected >= {expected_min_cells}")
    if adata.shape[1] < expected_min_genes:
        report.add_warning(f"Only {adata.shape[1]} genes, expected >= {expected_min_genes}")

    # Check expression range (should be log2(FPKM+1))
    x_min = float(adata.X.min())
    x_max = float(adata.X.max())
    report.add_check(f"Expression range: [{x_min:.2f}, {x_max:.2f}]")
    if x_min < -0.01:
        report.add_error(f"Negative expression values found (min={x_min})")
    if x_max > 25:
        report.add_warning(f"Unusually high max expression ({x_max}), expected <20 for log2(FPKM+1)")

    # Check obs columns
    expected_obs = ["specimen_id", "dataset", "subclass_scANVI"]
    for col in expected_obs:
        if col in adata.obs.columns:
            report.add_check(f"  obs.{col}: present")
        else:
            report.add_warning(f"  obs.{col}: missing")

    # Check UMAP coordinates
    for key in ["X_umap", "X_umap_expr", "X_umap_ephys"]:
        if key in adata.obsm:
            n_valid = (~np.isnan(adata.obsm[key])).all(axis=1).sum()
            report.add_check(f"  obsm.{key}: {n_valid} cells with coordinates")
        else:
            if key == "X_umap":
                pass  # not always expected
            else:
                report.add_warning(f"  obsm.{key}: missing")

    return report


def validate_viewer(html_path):
    """Validate the interactive viewer HTML."""
    report = ValidationReport("Interactive Viewer", str(html_path))

    if not Path(html_path).exists():
        report.add_error(f"File not found: {html_path}")
        return report

    size_mb = Path(html_path).stat().st_size / (1024 * 1024)
    report.add_check(f"File size: {size_mb:.1f} MB")

    if size_mb < 1.0:
        report.add_warning(f"Unexpectedly small ({size_mb:.1f} MB), may be incomplete")
    if size_mb > 20:
        report.add_warning(f"Very large ({size_mb:.1f} MB), consider optimization")

    # Check for key content markers
    content = Path(html_path).read_text()
    markers = ["plotly", "exprPlot", "ephysPlot", "morphology"]
    for marker in markers:
        if marker in content:
            report.add_check(f"  Contains '{marker}'")
        else:
            report.add_error(f"  Missing '{marker}' — viewer may be broken")

    return report


def validate_svg_map(csv_path, trace_dir=None):
    """Validate the trace SVG mapping CSV."""
    report = ValidationReport("Trace SVG Map", str(csv_path))

    if not Path(csv_path).exists():
        report.add_error(f"File not found: {csv_path}")
        return report

    df = pd.read_csv(csv_path)
    report.add_check(f"{len(df)} specimens mapped to SVGs")

    # Check for required columns
    for col in ["specimen_id", "trace_svg"]:
        if col not in df.columns:
            report.add_error(f"Missing column: {col}")

    # Spot-check a few SVG paths resolve
    if trace_dir and "trace_svg" in df.columns:
        sample = df.head(5)
        n_found = 0
        for _, row in sample.iterrows():
            svg_path = Path(trace_dir) / row["trace_svg"]
            if svg_path.exists():
                n_found += 1
        report.add_check(f"  SVG spot-check: {n_found}/5 files found")
        if n_found < 3:
            report.add_warning(f"Many SVGs not found at {trace_dir}")

    return report
