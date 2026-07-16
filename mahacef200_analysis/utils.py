"""
utils.py
========
Reusable helper functions for the MAHACEF-200 Phase 1 analysis pipeline.

All functions are stateless, well-typed, and raise descriptive exceptions so
that calling scripts can fail fast with actionable messages.
"""

from __future__ import annotations

import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LOGGING HELPERS
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger pre-configured by config.configure_logging().

    Imports config lazily to avoid circular imports at module load time.

    Parameters
    ----------
    name : str
        Logger name (pass __name__ from the calling module).

    Returns
    -------
    logging.Logger
    """
    from mahacef200_analysis.config import configure_logging
    return configure_logging(name)


# ---------------------------------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------------------------------

def ensure_directories(*paths: Path) -> None:
    """
    Create all given directories (and intermediate parents) if they don't exist.

    Parameters
    ----------
    *paths : Path
        One or more directory paths to create.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DATAFRAME VALIDATION
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Assert that all required columns are present in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_cols : list[str]
        Expected column names.
    dataset_name : str
        Human-readable name used in error messages.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def validate_not_empty(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """
    Assert that *df* contains at least one row.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError(f"[{dataset_name}] DataFrame is empty after filtering.")


def validate_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Assert that all listed columns contain numeric (float/int) data.

    Non-numeric columns are coerced; if all values become NaN after coercion
    a warning is logged but no exception is raised so the pipeline continues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    numeric_cols : list[str]
        Columns that should be numeric.
    dataset_name : str
        Human-readable name used in log messages.
    """
    logger = logging.getLogger(__name__)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                "[%s] Column '%s' is not numeric (dtype=%s). "
                "Attempting coercion.",
                dataset_name,
                col,
                df[col].dtype,
            )


# ---------------------------------------------------------------------------
# DUPLICATE DETECTION
# ---------------------------------------------------------------------------

def detect_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    dataset_name: str = "dataset",
) -> int:
    """
    Return the count of duplicate rows and log a warning if any are found.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    subset : list[str] | None
        Columns to consider when identifying duplicates.
        If None, all columns are used.
    dataset_name : str
        Human-readable name used in log messages.

    Returns
    -------
    int
        Number of duplicate rows.
    """
    logger = logging.getLogger(__name__)
    n_dup = int(df.duplicated(subset=subset).sum())
    if n_dup > 0:
        logger.warning(
            "[%s] Found %d duplicate row(s)%s.",
            dataset_name,
            n_dup,
            f" on columns {subset}" if subset else "",
        )
    else:
        logger.info("[%s] No duplicates detected.", dataset_name)
    return n_dup


# ---------------------------------------------------------------------------
# MONTH PARSING
# ---------------------------------------------------------------------------

def parse_billing_month(series: pd.Series) -> pd.Series:
    """
    Convert an integer billing_month (YYYYMM) series to a pandas Period[M].

    Parameters
    ----------
    series : pd.Series
        Integer series of the form 202304 (April 2023).

    Returns
    -------
    pd.Series
        Series of pandas Period objects at monthly frequency.

    Raises
    ------
    ValueError
        If the series contains values that cannot be parsed.
    """
    try:
        str_series = series.astype(str)
        return str_series.apply(lambda x: pd.Period(x, freq="M"))
    except Exception as exc:
        raise ValueError(f"Cannot parse billing_month series: {exc}") from exc


def billing_month_to_date(series: pd.Series) -> pd.Series:
    """
    Convert an integer billing_month (YYYYMM) series to the first day of that
    month as a pandas Timestamp.

    Parameters
    ----------
    series : pd.Series
        Integer series of the form 202304.

    Returns
    -------
    pd.Series
        Series of pandas Timestamps (day=1 of each month).
    """
    return pd.to_datetime(series.astype(str), format="%Y%m")


def billing_month_label(series: pd.Series) -> pd.Series:
    """
    Convert integer billing_month to a human-readable label such as 'Apr-2023'.

    Parameters
    ----------
    series : pd.Series
        Integer series of the form 202304.

    Returns
    -------
    pd.Series
        String series like 'Apr-2023'.
    """
    return billing_month_to_date(series).dt.strftime("%b-%Y")


# ---------------------------------------------------------------------------
# STATE NAME NORMALISATION
# ---------------------------------------------------------------------------

def normalize_state_name(series: pd.Series) -> pd.Series:
    """
    Normalise state names to UPPER CASE with stripped whitespace.

    Parameters
    ----------
    series : pd.Series
        Raw state name strings.

    Returns
    -------
    pd.Series
        Cleaned state name strings.
    """
    return series.astype(str).str.strip().str.upper()


# ---------------------------------------------------------------------------
# EXPORT HELPERS
# ---------------------------------------------------------------------------

def export_csv(
    df: pd.DataFrame,
    path: Path,
    index: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """
    Export a DataFrame to CSV with UTF-8 BOM encoding (Excel-compatible).

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    path : Path
        Destination file path.
    index : bool
        Whether to write the row index.
    logger : logging.Logger | None
        Optional logger for confirmation message.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=index, encoding="utf-8-sig")
    if logger:
        logger.info("CSV exported → %s  (%d rows × %d cols)", path, *df.shape)


def export_excel(
    df: pd.DataFrame,
    path: Path,
    sheet_name: str = "Sheet1",
    index: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """
    Export a DataFrame to Excel (.xlsx) using openpyxl.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.
    path : Path
        Destination file path.
    sheet_name : str
        Name of the worksheet.
    index : bool
        Whether to write the row index.
    logger : logging.Logger | None
        Optional logger for confirmation message.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
    if logger:
        logger.info("Excel exported → %s  (%d rows × %d cols)", path, *df.shape)


# ---------------------------------------------------------------------------
# MARKDOWN REPORT WRITING
# ---------------------------------------------------------------------------

def write_markdown_report(path: Path, content: str, logger: logging.Logger | None = None) -> None:
    """
    Write a markdown string to file.

    Parameters
    ----------
    path : Path
        Destination .md file path.
    content : str
        Markdown text (may contain newlines, tables, headers, etc.).
    logger : logging.Logger | None
        Optional logger for confirmation message.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if logger:
        logger.info("Markdown report written → %s", path)


def md_table_from_dict(data: dict[str, Any], col1: str = "Metric", col2: str = "Value") -> str:
    """
    Convert a flat dictionary to a two-column Markdown table string.

    Parameters
    ----------
    data : dict
        {metric: value} pairs.
    col1, col2 : str
        Column header labels.

    Returns
    -------
    str
        Markdown table as a string.
    """
    lines = [f"| {col1} | {col2} |", "| --- | --- |"]
    for k, v in data.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def md_dataframe_table(df: pd.DataFrame, float_fmt: str = ",.2f") -> str:
    """
    Convert a DataFrame to a Markdown table string.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render.
    float_fmt : str
        Python format string for floating-point columns.

    Returns
    -------
    str
        Markdown table.
    """
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:{float_fmt}}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------

def summary_statistics(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """
    Compute descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
    for the listed numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    numeric_cols : list[str]
        Columns to summarise.

    Returns
    -------
    pd.DataFrame
        Transposed describe() output.
    """
    valid_cols = [c for c in numeric_cols if c in df.columns]
    return df[valid_cols].describe().T.round(2)


# ---------------------------------------------------------------------------
# MERGE VALIDATION
# ---------------------------------------------------------------------------

def validate_merge_result(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    merge_keys: list[str],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Validate a merge result and return a diagnostic dictionary.

    Checks performed:
    - Row count of merged vs left/right
    - Null counts in key columns after merge
    - Duplicate key combinations

    Parameters
    ----------
    left_df, right_df : pd.DataFrame
        Input frames before merge.
    merged_df : pd.DataFrame
        Result after merge.
    merge_keys : list[str]
        Columns used as merge keys.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    dict
        Diagnostic summary.
    """
    n_left = len(left_df)
    n_right = len(right_df)
    n_merged = len(merged_df)
    n_dups = int(merged_df.duplicated(subset=merge_keys).sum())

    # Missing values per column
    missing_per_col: dict[str, int] = {}
    for col in merged_df.columns:
        n_null = int(merged_df[col].isnull().sum())
        if n_null > 0:
            missing_per_col[col] = n_null

    diagnostics = {
        "left_rows": n_left,
        "right_rows": n_right,
        "merged_rows": n_merged,
        "duplicate_key_rows": n_dups,
        "missing_values": missing_per_col,
    }

    if logger:
        logger.info("Merge diagnostics: %s", diagnostics)

    return diagnostics


# ---------------------------------------------------------------------------
# MISCELLANEOUS
# ---------------------------------------------------------------------------

def format_number(value: float | int, decimals: int = 2) -> str:
    """Return a comma-formatted number string, e.g. 1,234,567.89."""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def current_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# DATASET METADATA WRITER  (Change 6 — reproducibility)
# ---------------------------------------------------------------------------

def write_dataset_metadata(
    output_path: Path,
    phase: str,
    script_name: str,
    source_dataset: str,
    extra: dict | None = None,
) -> None:
    """
    Write a JSON metadata sidecar alongside an exported CSV or Excel file.

    The sidecar is named  <stem>.metadata.json  in the same directory.
    It records provenance so every exported file is fully traceable.

    Parameters
    ----------
    output_path : Path
        The primary exported file (CSV or Excel).
    phase : str
        Phase label, e.g. "Phase 2 - Sales Trend Analysis".
    script_name : str
        Filename of the generating script.
    source_dataset : str
        Name of the input dataset consumed.
    extra : dict | None
        Optional additional key-value pairs.
    """
    import json
    import subprocess

    from mahacef200_analysis.config import PROJECT_VERSION

    git_commit: str = "N/A"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=3,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    metadata: dict[str, Any] = {
        "created_on": datetime.utcnow().isoformat() + "Z",
        "source_dataset": source_dataset,
        "phase": phase,
        "script_name": script_name,
        "git_commit": git_commit,
        "version": PROJECT_VERSION,
    }
    if extra:
        metadata.update(extra)

    sidecar = output_path.with_suffix("").with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# STANDARDIZED PHASE REPORT BUILDER  (Change 5 — consistent structure)
# ---------------------------------------------------------------------------

def build_phase_report(
    phase_number: str,
    phase_title: str,
    objective: str,
    dataset_used: str,
    methodology: str,
    key_findings: str,
    business_insights: str,
    limitations: str,
    next_phase: str,
    generated_by: str,
) -> str:
    """
    Assemble a full Markdown report using the standardized 7-section template
    applied uniformly across every analysis phase (2 through 9).

    Parameters
    ----------
    phase_number : str   Phase identifier, e.g. "2" or "1.5"
    phase_title : str    Section title, e.g. "Sales Trend Analysis"
    objective … next_phase : str   Markdown content for each section.
    generated_by : str   Script filename.

    Returns
    -------
    str
        Complete Markdown report string.
    """
    return (
        f"# Phase {phase_number} \u2014 {phase_title}\n"
        f"## MAHACEF-200 | Mankind Cures\n\n"
        f"**Generated:** {current_timestamp()} UTC  \n"
        f"**Script:** `{generated_by}`\n\n"
        f"---\n\n"
        f"## 1. Objective\n\n{objective}\n\n"
        f"---\n\n"
        f"## 2. Dataset Used\n\n{dataset_used}\n\n"
        f"---\n\n"
        f"## 3. Methodology\n\n{methodology}\n\n"
        f"---\n\n"
        f"## 4. Key Findings\n\n{key_findings}\n\n"
        f"---\n\n"
        f"## 5. Business Insights\n\n{business_insights}\n\n"
        f"---\n\n"
        f"## 6. Limitations\n\n{limitations}\n\n"
        f"---\n\n"
        f"## 7. Next Phase\n\n{next_phase}\n\n"
        f"---\n"
        f"*Auto-generated by `{generated_by}` | MAHACEF-200 Phase {phase_number} Pipeline*\n"
    )
