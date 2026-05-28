from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import STRFeatureTransformer


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


def build_pipeline(estimator: object, allele_pairs: list[tuple[str, str]]) -> Pipeline:
    # --- EXTENSION POINT: PIPELINE CONSTRUCTION FOR STR VS SNP ---
    # Currently, this helper function is built strictly for STR allele pairs and wraps it with `STRFeatureTransformer`.
    # To extend it for SNP or custom genotype pipelines:
    # 1. You can add a new parameter, e.g., `genotype_type: str = "str"`.
    # 2. Use a conditional branch to choose the feature engineering step:
    #    if genotype_type == "str":
    #        features_step = ("str_features", STRFeatureTransformer(allele_pairs=allele_pairs))
    #    elif genotype_type == "snp":
    #        features_step = ("snp_features", SNPFeatureTransformer())  # Or standard StandardScaler if already numeric
    # 3. Modify the list of pipeline steps accordingly:
    #    steps = [features_step, ("imputer", SimpleImputer(strategy="median")), ...]
    return Pipeline(
        steps=[
            ("str_features", STRFeatureTransformer(allele_pairs=allele_pairs)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )

