import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..data import find_allele_pairs


class STRFeatureTransformer(BaseEstimator, TransformerMixin):
    """Convert STR A1/A2 genotype columns into numeric model features.

    # --- EXTENSION POINT: FEATURE ENGINEERING FOR SNP ---
    # STRFeatureTransformer is specifically designed for STR (Short Tandem Repeat) alleles (low, high, sum, diff, heterozygosity).
    # To extend the pipeline for SNP genotype data, you can:
    # 1. Create a `SNPFeatureTransformer` class in this file or a new `snp_features.py` file.
    # 2. SNP data usually requires different preprocessing depending on its format:
    #    - If SNPs are represented as dosage (0, 1, 2 representing reference allele count):
    #      Use a simple identity transformer or Standard Scaler directly.
    #    - If SNPs are represented as string alleles (e.g., A/T, C/G, or AA, AT, TT):
    #      Implement a One-Hot Encoder or Label Encoder mapping 'AA'->0, 'AT'->1, 'TT'->2.
    # 3. Reference the newly created SNP feature transformer inside `src/ancestry/models.py` when building the pipeline.
    """

    def __init__(
        self,
        allele_pairs: list[tuple[str, str]] | None = None,
        include_sum: bool = True,
        include_diff: bool = True,
        include_heterozygosity: bool = True,
    ) -> None:

        self.allele_pairs = allele_pairs
        self.include_sum = include_sum
        self.include_diff = include_diff
        self.include_heterozygosity = include_heterozygosity

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("STRFeatureTransformer expects a pandas DataFrame.")

        self.allele_pairs_ = self.allele_pairs or find_allele_pairs(X.columns)
        if not self.allele_pairs_:
            raise ValueError("No allele pairs found in input data.")

        self.feature_names_out_ = self._build_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("STRFeatureTransformer expects a pandas DataFrame.")

        feature_blocks: list[pd.Series] = []
        for locus_number, (a1_column, a2_column) in enumerate(self.allele_pairs_, start=1):
            a1 = pd.to_numeric(X[a1_column], errors="coerce")
            a2 = pd.to_numeric(X[a2_column], errors="coerce")

            low = np.minimum(a1, a2)
            high = np.maximum(a1, a2)
            diff = np.abs(high - low)

            feature_blocks.append(pd.Series(low, index=X.index, name=f"locus_{locus_number:02d}_allele_low"))
            feature_blocks.append(pd.Series(high, index=X.index, name=f"locus_{locus_number:02d}_allele_high"))

            if self.include_sum:
                feature_blocks.append(pd.Series(a1 + a2, index=X.index, name=f"locus_{locus_number:02d}_allele_sum"))

            if self.include_diff:
                feature_blocks.append(pd.Series(diff, index=X.index, name=f"locus_{locus_number:02d}_allele_diff"))

            if self.include_heterozygosity:
                heterozygous = pd.Series(np.where(diff.isna(), np.nan, diff > 0), index=X.index)
                feature_blocks.append(heterozygous.astype(float).rename(f"locus_{locus_number:02d}_heterozygous"))

        return pd.concat(feature_blocks, axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        for locus_number, _pair in enumerate(self.allele_pairs_, start=1):
            names.extend(
                [
                    f"locus_{locus_number:02d}_allele_low",
                    f"locus_{locus_number:02d}_allele_high",
                ]
            )
            if self.include_sum:
                names.append(f"locus_{locus_number:02d}_allele_sum")
            if self.include_diff:
                names.append(f"locus_{locus_number:02d}_allele_diff")
            if self.include_heterozygosity:
                names.append(f"locus_{locus_number:02d}_heterozygous")
        return names
