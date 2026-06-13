from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import ID_COLUMNS, TARGET_COLUMN


ALLELE_COLUMN_RE = re.compile(r"^(A[12])(?:\.(\d+))?$")
SNP_COLUMN_RE = re.compile(r"^rs\d+$", re.IGNORECASE)
SNP_COLUMN_ALIASES = {
    "sample_id": "SAMPLE",
    "pop": "SUBPOP",
    "super_pop": "POP",
}


def load_genotype_table(path: str | Path, genotype_type: str = "str") -> pd.DataFrame:
    """Load a cleaned genotype table and normalize common column names."""
    df = pd.read_csv(path)
    if genotype_type == "snp":
        df = df.rename(columns={source: target for source, target in SNP_COLUMN_ALIASES.items() if source in df.columns})
    return df


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
    return validate_str_genotype_table(df, target_column=target_column)


def validate_str_genotype_table(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[tuple[str, str]]:
    """Validate required columns and return detected allele pairs.

    # --- EXTENSION POINT: DATA VALIDATION FOR SNP & SUBPOP ---
    # 1. Target Column (POP vs SUBPOP):
    #    - Currently accepts either target_column="POP" or target_column="SUBPOP" based on input.
    #    - Make sure the data CSV actually contains the requested target column.
    # 2. Genotype Type Support (STR vs SNP):
    #    - Currently, this method searches for A1/A2 STR allele pairs.
    #    - For SNP genotype data:
    #      A. If SNP is formatted as A1/A2 alleles (like STR), this method can be reused.
    #      B. If SNP is formatted differently (e.g., direct SNP names like rs12345 with values 0, 1, 2 or AA, AT, TT):
    #         - Skip `find_allele_pairs` validation.
    #         - Validate that the features match SNP names and don't contain invalid genotypes.
    #         - You can add an `if genotype_type == "snp"` branch here or pass a parameter.
    """
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


def snp_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if SNP_COLUMN_RE.match(str(column))]


def validate_snp_genotype_table(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    required_columns = {"SAMPLE", target_column}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_target = int(df[target_column].isna().sum())
    if missing_target:
        raise ValueError(f"Target column {target_column!r} has {missing_target} missing values.")

    feature_columns = snp_feature_columns(df)
    if not feature_columns:
        raise ValueError("No SNP marker columns matching rs<digits> were found.")

    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"SNP columns must be numeric dosage/features. Non-numeric columns: {non_numeric[:10]}")

    return feature_columns


def validate_table_for_genotype(
    df: pd.DataFrame,
    genotype_type: str,
    target_column: str = TARGET_COLUMN,
) -> list[tuple[str, str]] | list[str]:
    if genotype_type == "str":
        return validate_str_genotype_table(df, target_column=target_column)
    if genotype_type == "snp":
        return validate_snp_genotype_table(df, target_column=target_column)
    raise ValueError(f"Unsupported genotype_type: {genotype_type!r}")


def metadata_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ID_COLUMNS if column in df.columns]
