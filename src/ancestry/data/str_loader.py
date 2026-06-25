"""STR-specific data loading module."""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple
from . import find_allele_pairs


def load_str_data(data_path: str | Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads STR interim data, validates structure, and splits features from the target label.

    Args:
        data_path (str | Path): Path to the interim STR CSV file.
        target_column (str): The column to extract as target label (e.g., 'SUBPOP').

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features) and y (target label).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Validation Check
    required_metadata = ["SAMPLE", target_column]
    for col in required_metadata:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' is missing. "
                f"Current columns: {list(df.columns[:5])}...{list(df.columns[-5:])}"
            )

    # Set SAMPLE as index
    df = df.set_index("SAMPLE")

    # 3. Identify STR allele feature columns
    allele_pairs = find_allele_pairs(df.columns)
    if not allele_pairs:
        raise ValueError("No STR allele pairs (A1/A2 columns) detected in the dataset.")

    # Collect all allele column names
    feature_cols = []
    for a1, a2 in allele_pairs:
        feature_cols.extend([a1, a2])

    X = df[feature_cols]
    y = df[target_column].astype(str)

    print(f"Successfully loaded STR task data from {data_path}:")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - STR Loci Features: {len(allele_pairs)} pairs ({X.shape[1]} columns)")
    print(f"  - Target Component: {target_column}")

    return X, y
