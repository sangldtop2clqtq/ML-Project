from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA  # Thay đổi ở đây
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
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=500,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def build_pipeline(
    estimator: object,
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str = "str",
    use_feature_selection: bool = False,
    n_components: int = 30,  # Đổi tên tham số thành n_components, thử nghiệm với 30 hoặc 40
) -> Pipeline:
    """Builds a scikit-learn Pipeline incorporating feature extraction,

    imputation, scaling, optional PCA, and the model estimator.
    """
    if genotype_type == "str":
        features_step = (
            "str_features",
            STRFeatureTransformer(allele_pairs=feature_columns),
        )
    elif genotype_type == "snp":
        features_step = (
            "snp_features",
            SNPFeatureTransformer(feature_columns=feature_columns),
        )
    else:
        raise ValueError(f"Unsupported genotype_type: {genotype_type!r}")

    pipeline_steps = [
        features_step,
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    # Thay thế hoàn toàn SelectKBest bằng PCA
    if genotype_type == "snp" and use_feature_selection:
        pipeline_steps.append(
            (
                "feature_selection",
                PCA(n_components=n_components, random_state=42),
            )
        )

    pipeline_steps.append(("model", estimator))

    return Pipeline(steps=pipeline_steps)