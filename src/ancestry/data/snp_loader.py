"""SNP-specific data loading module.

Future work:
- load SNP raw/interim files
- normalize SNP genotype format
- validate SAMPLE, POP, SUBPOP, and SNP marker columns
"""
"""SNP-specific data loading module."""

import pandas as pd
from typing import Tuple, List
import os


def load_snp_data(data_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads SNP interim data, normalizes column names, validates structure,

    and splits features from the target label.

    Args:
        data_path (str): Path to the interim SNP CSV file.
        target_column (str): The column to extract as target label (e.g.,
        'SUBPOP').

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features) and y (target label).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Normalize column names based on README specification
    # Map original columns to system standard names
    rename_mapping = {
        "sample_id": "SAMPLE",
        "super_pop": "POP",
        "pop": "SUBPOP",  # Per README: pop -> SUBPOP (which is our target task)
    }
    df = df.rename(columns=rename_mapping)

    # 3. Validation Check
    required_metadata = ["SAMPLE", "POP", "SUBPOP"]
    for col in required_metadata:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' is missing after normalization. "
                f"Current columns: {list(df.columns[:5])}...{list(df.columns[-5:])}"
            )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the dataset."
        )

    # Set SAMPLE as index
    df = df.set_index("SAMPLE")

    # 4. Separate Features (SNP markers starting with 'rs') and Target
    # Identify SNP features dynamically
    feature_cols = [col for col in df.columns if col.startswith("rs")]

    if not feature_cols:
        raise ValueError(
            "No SNP feature columns (starting with 'rs') detected in the dataset."
        )

    X = df[feature_cols]
    y = df[target_column]

    print(
        f"Successfully loaded SNP task data from {data_path}:"
    )
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - SNP Features: {X.shape[1]}")
    print(f"  - Target Component: {target_column}")

    return X, y