from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .features import SNPFeatureTransformer, STRFeatureTransformer


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrap XGBClassifier with a label encoder for string targets."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._xgb = XGBClassifier(**kwargs)
        self._le = LabelEncoder()

    def fit(self, X, y):
        y_encoded = self._le.fit_transform(y)
        self._xgb.fit(X, y_encoded)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        y_pred = self._xgb.predict(X)
        return self._le.inverse_transform(y_pred.astype(int))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **params):
        self.kwargs.update(params)
        self._xgb = XGBClassifier(**self.kwargs)
        return self


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
            max_depth=12,
            min_samples_leaf=4,
            min_samples_split=10,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=4,
            min_samples_split=10,
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifierWrapper(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=1.0,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def random_search_spaces(
    genotype_type: str,
    random_state: int = RANDOM_STATE,
    max_pca_components: int | None = None,
) -> dict[str, dict[str, list[object]]]:
    spaces: dict[str, dict[str, list[object]]] = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0],
            "model__solver": ["lbfgs", "saga"],
        },
        "random_forest": {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__random_state": [random_state],
        },
        "extra_trees": {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__random_state": [random_state],
        },
        "xgboost": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [1.0, 2.0, 5.0],
            "model__random_state": [random_state],
        },
    }

    if genotype_type == "str":
        str_feature_space = {
            "str_features__include_sum": [False, True],
            "str_features__include_diff": [False, True],
            "str_features__include_heterozygosity": [False, True],
        }
        for space in spaces.values():
            space.update(str_feature_space)
    elif genotype_type == "snp":
        if max_pca_components is None or max_pca_components < 1:
            max_pca_components = 1
        candidate_components = [
            value
            for value in [10, 20, 30, 40, 50]
            if value <= max_pca_components
        ]
        if not candidate_components:
            candidate_components = [max_pca_components]
        snp_space = {
            "feature_selection__n_components": candidate_components,
        }
        for space in spaces.values():
            space.update(snp_space)
    else:
        raise ValueError(f"Unsupported genotype_type: {genotype_type!r}")

    return spaces


def build_pipeline(
    estimator: object,
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str = "str",
    use_feature_selection: bool = False,
    n_components: int = 30,
) -> Pipeline:
    """Build a pipeline with feature extraction, preprocessing, and a model."""
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

    if genotype_type == "snp" and use_feature_selection:
        pipeline_steps.append(
            (
                "feature_selection",
                PCA(n_components=n_components, random_state=RANDOM_STATE),
            )
        )

    pipeline_steps.append(("model", estimator))

    return Pipeline(steps=pipeline_steps)
