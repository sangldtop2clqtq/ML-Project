import re
from pathlib import Path

import pandas as pd

from ..config import ID_COLUMNS, TARGET_COLUMN


ALLELE_COLUMN_RE = re.compile(r"^(A[12])(?:\.(\d+))?$")


def load_genotype_table(path: str | Path) -> pd.DataFrame:
    """Load the cleaned STR genotype table."""
    return pd.read_csv(path)


def find_allele_pairs(columns: list[str] | pd.Index) -> list[tuple[str, str]]:
    """Return STR allele column pairs in locus order.

    The cleaned ST1 file uses columns like A1,A2,A1.1,A2.1,... where each
    A1/A2 pair represents one STR locus.
    """
    by_locus: dict[int, dict[str, str]] = {}
    for column in columns:
        match = ALLELE_COLUMN_RE.match(str(column))
        if not match:
            continue

        allele_name = match.group(1)
        locus_index = int(match.group(2) or 0)
        by_locus.setdefault(locus_index, {})[allele_name] = str(column)

    pairs: list[tuple[str, str]] = []
    for locus_index in sorted(by_locus):
        locus = by_locus[locus_index]
        if "A1" in locus and "A2" in locus:
            pairs.append((locus["A1"], locus["A2"]))

    return pairs


def validate_genotype_table(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[tuple[str, str]]:
    """Validate required columns and return detected allele pairs."""
    required_columns = {"SAMPLE", target_column}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    allele_pairs = find_allele_pairs(df.columns)
    if not allele_pairs:
        raise ValueError("No A1/A2 STR allele pairs were found.")

    missing_target = int(df[target_column].isna().sum())
    if missing_target:
        raise ValueError(f"Target column {target_column!r} has {missing_target} missing values.")

    return allele_pairs


def metadata_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ID_COLUMNS if column in df.columns]
