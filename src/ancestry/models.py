from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import SNPFeatureTransformer, STRFeatureTransformer


def candidate_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=500,
            n_jobs=1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=500,
            n_jobs=1,
            random_state=random_state,
        ),
    }


def build_pipeline(
    estimator: object,
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str = "str",
) -> Pipeline:
    if genotype_type == "str":
        features_step = ("str_features", STRFeatureTransformer(allele_pairs=feature_columns))
    elif genotype_type == "snp":
        features_step = ("snp_features", SNPFeatureTransformer(feature_columns=feature_columns))
    else:
        raise ValueError(f"Unsupported genotype_type: {genotype_type!r}")

    return Pipeline(
        steps=[
            features_step,
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
