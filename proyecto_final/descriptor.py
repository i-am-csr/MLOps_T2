"""Lightweight dataset descriptor CLI.

This simplified script produces a Markdown profile of a dataset WITHOUT
performing any cleaning, outlier analysis, or plotting. It is intended
for quick inspection and manual review before any transformations.

Generated artifact (by default under reports/eda/):
  - <input_stem>_analysis.md : Human-readable Markdown report derived from the input filename.

Included in the Markdown:
  - Dataset path & shape
  - Memory usage
  - Column summary table with:
      name | dtype | non-null | missing (%) | unique | example values | min | max | mean | std | skew
    (numeric-only stats are blank for non-numeric columns)
  - Notes section placeholder for manual annotations.

Usage examples:
  python -m proyecto_final.descriptor describe
  python -m proyecto_final.descriptor describe --input-path data/raw/energy_efficiency_original.csv
  python -m proyecto_final.descriptor describe --output-dir reports/eda_original --max-example-values 8

Design choices:
  - Markdown only (CSV export removed for simplicity).
  - Minimal branching to keep cognitive complexity low.
  - Output filename now reflects the analyzed file: <stem>_analysis.md
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
import sys

from proyecto_final.config import RAW_DATA_DIR, REPORTS_DIR
from proyecto_final.data.data_loader import DataLoader

app = typer.Typer(help="Generate a lightweight Markdown dataset profile (markdown only).")


@app.callback()
def main():  # pragma: no cover
    """Root command. Use the 'describe' subcommand."""
    pass


def _human_bytes(num: int) -> str:
    if num is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    n = float(num)
    while n >= 1024 and idx < len(units)-1:
        n /= 1024.0
        idx += 1
    return f"{n:.2f} {units[idx]}"


def _example_values(series: pd.Series, max_values: int) -> str:
    if series.empty:
        return ""
    vals = series.dropna().unique()[:max_values]
    rendered = []
    for v in vals:
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        rendered.append(s)
    return ", ".join(rendered)


def _column_summary(df: pd.DataFrame, max_example_values: int, show_progress: bool = True, keep_progress: bool = True) -> pd.DataFrame:
    """Vectorized column summary for all columns with optional progress bar.

    Args:
        df: Input DataFrame
        max_example_values: Max examples per column
        show_progress: Whether to show tqdm bars
        keep_progress: Whether to keep the bar output after completion (leave=True)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "name", "dtype", "non_null", "missing", "missing_pct", "unique",
            "examples", "min", "max", "mean", "std", "skew"
        ])

    total_rows = len(df)
    non_null = df.notna().sum()
    missing = df.isna().sum()
    missing_pct = (missing / total_rows * 100).round(2)
    unique = df.nunique(dropna=True)
    dtypes = df.dtypes.astype(str)

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        desc = numeric.describe().T
        skew = numeric.skew()
    else:
        desc = pd.DataFrame()
        skew = pd.Series(dtype=float)

    examples = {}
    use_progress = show_progress and sys.stdout.isatty()
    iterable = tqdm(
        df.columns,
        desc="Profiling columns",
        leave=keep_progress,
        disable=not use_progress,
        ncols=80
    ) if use_progress else df.columns

    for col in iterable:
        examples[col] = _example_values(df[col], max_example_values)

    rows = []
    iterable2 = tqdm(
        df.columns,
        desc="Assembling rows",
        leave=keep_progress,
        disable=not use_progress,
        ncols=80
    ) if use_progress else df.columns

    for col in iterable2:
        is_num = col in numeric.columns
        def _fmt(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (int, float)):
                return f"{value:.3f}"
            return str(value)
        rows.append({
            "name": col,
            "dtype": dtypes[col],
            "non_null": int(non_null[col]),
            "missing": int(missing[col]),
            "missing_pct": f"{missing_pct[col]:.2f}",
            "unique": int(unique[col]),
            "examples": examples[col],
            "min": _fmt(desc.loc[col, "min"]) if is_num and col in desc.index else "",
            "max": _fmt(desc.loc[col, "max"]) if is_num and col in desc.index else "",
            "mean": _fmt(desc.loc[col, "mean"]) if is_num and col in desc.index else "",
            "std": _fmt(desc.loc[col, "std"]) if is_num and col in desc.index else "",
            "skew": _fmt(skew[col]) if is_num and col in skew.index else "",
        })

    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, max_width: int = 120) -> str:
    if df.empty:
        return "(No columns)"
    order = [
        "name", "dtype", "non_null", "missing", "missing_pct", "unique",
        "examples", "min", "max", "mean", "std", "skew"
    ]
    df = df[order]

    def trunc(val):
        if pd.isna(val):
            return ""
        s = str(val)
        return s if len(s) <= max_width else s[: max_width - 3] + "..."

    lines = [" | ".join(order), " | ".join(["---"] * len(order))]
    for _, row in df.iterrows():
        lines.append(" | ".join(trunc(row[col]) for col in order))
    return "\n".join(lines)


def _generate_markdown(input_path: Path, df: pd.DataFrame, column_df: pd.DataFrame, output_md: Path) -> Path:
    memory_bytes = int(df.memory_usage(deep=True).sum())
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    md_lines: List[str] = []
    md_lines.append("# Dataset Profile\n")
    md_lines.append(f"**Source file:** `{input_path}`  ")
    md_lines.append(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns  ")
    md_lines.append(f"**Memory usage:** {_human_bytes(memory_bytes)}  ")
    md_lines.append(f"**Numeric columns:** {len(numeric_cols)} | **Categorical columns:** {len(categorical_cols)}\n")

    if numeric_cols:
        md_lines.append(f"Numeric columns: {', '.join(numeric_cols)}\n")
    if categorical_cols:
        md_lines.append(f"Categorical columns: {', '.join(categorical_cols)}\n")

    md_lines.append("## Column Summary\n")
    md_lines.append(_markdown_table(column_df))

    output_md.write_text("\n".join(md_lines), encoding="utf-8")
    return output_md


def _validate_input_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return path


def _prepare_output_md(output_dir: Path, input_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_analysis.md"


@app.command("describe")
def command_markdown(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / "energy_efficiency_modified_dropped.csv",
        "--input-path",
        help="Path to input CSV file",
    ),
    output_dir: Path = typer.Option(
        REPORTS_DIR / "eda",
        "--output-dir",
        help="Directory where the markdown profile will be written",
    ),
    max_example_values: int = typer.Option(5, help="Max example distinct values to list per column"),
    show_progress: bool = typer.Option(True, help="Show tqdm progress bars while profiling", rich_help_panel="Display"),
    keep_progress: bool = typer.Option(True, help="Keep progress bars on screen after completion"),
) -> Path:
    """Generate a Markdown dataset profile with basic descriptive statistics (markdown only)."""
    input_path = _validate_input_path(input_path)
    output_md = _prepare_output_md(output_dir, input_path)

    logger.info(f"Loading dataset: {input_path}")
    df = DataLoader.load_csv(input_path)
    logger.info(f"Shape: {df.shape}")

    logger.info("Computing column summary")
    col_df = _column_summary(
        df,
        max_example_values=max_example_values,
        show_progress=show_progress,
        keep_progress=keep_progress,
    )

    logger.info("Writing markdown profile")
    _generate_markdown(input_path, df, col_df, output_md)

    logger.success(f"Profile written: {output_md}")
    return output_md


__all__ = ["app", "command_markdown", "main"]

if __name__ == "__main__":  # pragma: no cover
    app()
