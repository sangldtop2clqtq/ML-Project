from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from .config import (
    CV_SPLITS,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data.snp_loader import load_snp_data
from .data.str_loader import load_str_data
from .models import build_pipeline, candidate_models, random_search_spaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ancestry prediction models from genotype data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config for a task, e.g. configs/snp_pop.json.",
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--target-column", help="Target label, usually POP or SUBPOP."
    )
    parser.add_argument("--genotype-type", choices=("str", "snp"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--cv-splits", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument(
        "--model-name",
        choices=sorted(candidate_models().keys()),
        help="Train only one model.",
    )
    parser.add_argument("--random-search-iters", type=int)
    parser.add_argument("--tuning-sample-fraction", type=float)
    parser.add_argument("--tuning-max-samples", type=int)
    parser.add_argument("--tuning-cv-splits", type=int)
    parser.add_argument(
        "--disable-random-search",
        action="store_true",
        help="Skip random search and use the default model hyperparameters.",
    )
    return parser.parse_args()


def main(force_random_search: bool | None = False) -> None:
    args = resolve_args(parse_args(), force_random_search=force_random_search)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.genotype_type == "snp":
        X, y = load_snp_data(str(args.data_path), args.target_column)
        feature_columns = list(X.columns)
    else:
        X, y = load_str_data(str(args.data_path), args.target_column)
        from .data import find_allele_pairs

        feature_columns = find_allele_pairs(X.columns)

    use_fs = args.genotype_type == "snp"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = candidate_models(random_state=args.random_state)
    if args.model_name:
        models = {args.model_name: models[args.model_name]}

    tuning_rows: list[dict[str, object]] = []
    best_pipeline = None
    best_model_name = None
    best_model_params = None
    tuning_sample_size = len(X_train)

    if args.use_random_search:
        X_tune, y_tune = build_tuning_subset(
            X_train,
            y_train,
            sample_fraction=args.tuning_sample_fraction,
            max_samples=args.tuning_max_samples,
            random_state=args.random_state,
            cv_splits=args.tuning_cv_splits,
        )
        tuning_sample_size = len(X_tune)
        tuning_results, best_model_name, best_pipeline, best_model_params, tuned_pipelines = tune_models(
            X_tune=X_tune,
            y_tune=y_tune,
            models=models,
            feature_columns=feature_columns,
            genotype_type=args.genotype_type,
            use_feature_selection=use_fs,
            random_state=args.random_state,
            n_iter=args.random_search_iters,
            cv_splits=args.tuning_cv_splits,
        )
        tuning_rows.extend(tuning_results)
        models_to_evaluate = tuned_pipelines
    else:
        cv_results = evaluate_candidates(
            X_train=X_train,
            y_train=y_train,
            models=models,
            feature_columns=feature_columns,
            genotype_type=args.genotype_type,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            use_feature_selection=use_fs,
        )
        tuning_rows.extend(cv_results.to_dict(orient="records"))
        best_model_name = str(cv_results.iloc[0]["model"])
        best_model_params = {}
        best_pipeline = build_pipeline(
            models[best_model_name],
            feature_columns,
            genotype_type=args.genotype_type,
            use_feature_selection=use_fs,
        )
        models_to_evaluate = models

    if best_pipeline is None or best_model_name is None:
        raise RuntimeError("No model pipeline was selected for final training.")

    holdout_scores = evaluate_models_on_holdout(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models_to_evaluate,
        feature_columns=feature_columns,
        genotype_type=args.genotype_type,
        use_feature_selection=use_fs,
    )

    final_pipeline = clone(best_pipeline)
    final_pipeline.fit(X_train, y_train)

    labels = sorted(y.unique())
    metrics, predictions = evaluate_holdout(final_pipeline, X_test, y_test, labels, [])

    model_path = args.output_dir / f"ancestry_{args.genotype_type}_model.joblib"
    joblib.dump(final_pipeline, model_path)

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        "tuning_balanced_accuracy_mean",
        ascending=False,
        na_position="last",
    )
    holdout_df = pd.DataFrame(holdout_scores)
    tuning_df = tuning_df.merge(holdout_df, on="model", how="left")
    tuning_df.to_csv(args.report_dir / "cv_results.csv", index=False)
    tuning_df.to_csv(args.report_dir / "tuning_results.csv", index=False)

    predictions.to_csv(args.report_dir / "holdout_predictions.csv", index=True)
    save_json(metrics, args.report_dir / "metrics.json")
    save_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_path": str(args.data_path),
            "genotype_type": args.genotype_type,
            "target_column": args.target_column,
            "model_path": str(model_path),
            "best_model": best_model_name,
            "best_params": best_model_params,
            "random_search_enabled": args.use_random_search,
            "random_search_iterations": args.random_search_iters,
            "tuning_sample_fraction": args.tuning_sample_fraction,
            "tuning_max_samples": args.tuning_max_samples,
            "tuning_cv_splits": args.tuning_cv_splits,
            "tuning_sample_size": tuning_sample_size,
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "labels": labels,
            "features": feature_columns,
        },
        args.report_dir / "run_metadata.json",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels,
        args.report_dir / "figures" / "confusion_matrix.png",
    )

    if args.use_random_search:
        print(f"Best tuned model: {best_model_name}")
    else:
        print(f"Best CV model: {best_model_name}")
    print(f"Best params: {json.dumps(best_model_params, ensure_ascii=True)}")
    print(f"Holdout accuracy: {metrics['accuracy']:.4f}")
    print(f"Holdout balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Artifacts saved to: {args.output_dir} and {args.report_dir}")


def resolve_args(
    args: argparse.Namespace,
    force_random_search: bool | None = False,
) -> argparse.Namespace:
    config: dict[str, object] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    args.genotype_type = args.genotype_type or str(
        config.get("genotype_type", "str")
    )
    args.data_path = args.data_path or Path(
        str(config.get("data_path", DEFAULT_DATA_PATH))
    )
    args.target_column = args.target_column or str(
        config.get("target_column", TARGET_COLUMN)
    )
    args.output_dir = args.output_dir or Path(
        str(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    )
    args.report_dir = args.report_dir or Path(
        str(config.get("report_dir", DEFAULT_REPORT_DIR))
    )
    args.test_size = (
        args.test_size
        if args.test_size is not None
        else float(config.get("test_size", TEST_SIZE))
    )
    args.cv_splits = (
        args.cv_splits
        if args.cv_splits is not None
        else int(config.get("cv_splits", CV_SPLITS))
    )
    args.random_state = (
        args.random_state
        if args.random_state is not None
        else int(config.get("random_state", RANDOM_STATE))
    )
    args.random_search_iters = (
        args.random_search_iters
        if args.random_search_iters is not None
        else int(config.get("random_search_iters", 12))
    )
    args.tuning_sample_fraction = (
        args.tuning_sample_fraction
        if args.tuning_sample_fraction is not None
        else float(config.get("tuning_sample_fraction", 0.6))
    )
    args.tuning_max_samples = (
        args.tuning_max_samples
        if args.tuning_max_samples is not None
        else int(config.get("tuning_max_samples", 0)) or None
    )
    args.tuning_cv_splits = (
        args.tuning_cv_splits
        if args.tuning_cv_splits is not None
        else int(config.get("tuning_cv_splits", 3))
    )
    args.use_random_search = not args.disable_random_search
    if "use_random_search" in config and not args.disable_random_search:
        args.use_random_search = bool(config.get("use_random_search", True))
    if force_random_search is not None:
        args.use_random_search = force_random_search
    return args


def build_tuning_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_fraction: float,
    max_samples: int | None,
    random_state: int,
    cv_splits: int,
) -> tuple[pd.DataFrame, pd.Series]:
    requested_size = int(len(X_train) * sample_fraction)
    if max_samples is not None:
        requested_size = min(requested_size, max_samples)

    if requested_size <= 0 or requested_size >= len(X_train):
        return X_train, y_train

    min_class_count = int(y_train.value_counts().min())
    if min_class_count < cv_splits:
        return X_train, y_train

    min_size_for_classes = cv_splits * int(y_train.nunique())
    requested_size = max(requested_size, min_size_for_classes)
    if requested_size >= len(X_train):
        return X_train, y_train

    try:
        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=requested_size,
            random_state=random_state,
            stratify=y_train,
        )
    except ValueError:
        return X_train, y_train

    if int(y_tune.value_counts().min()) < cv_splits:
        return X_train, y_train

    return X_tune, y_tune


def tune_models(
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    models: dict[str, object],
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str,
    use_feature_selection: bool,
    random_state: int,
    n_iter: int,
    cv_splits: int,
) -> tuple[list[dict[str, object]], str, object, dict[str, object]]:
    max_pca_components = None
    if genotype_type == "snp":
        fold_train_size = len(X_tune) - int(np.ceil(len(X_tune) / cv_splits))
        max_pca_components = max(
            1,
            min(X_tune.shape[1], fold_train_size),
        )

    search_spaces = random_search_spaces(
        genotype_type=genotype_type,
        random_state=random_state,
        max_pca_components=max_pca_components,
    )
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows: list[dict[str, object]] = []
    best_model_name = ""
    best_pipeline = None
    best_params: dict[str, object] = {}
    best_score = -np.inf
    tuned_pipelines: dict[str, object] = {}

    for model_name, estimator in models.items():
        pipeline = build_pipeline(
            estimator,
            feature_columns,
            genotype_type=genotype_type,
            use_feature_selection=use_feature_selection,
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_spaces[model_name],
            n_iter=n_iter,
            scoring="balanced_accuracy",
            n_jobs=-1,
            cv=cv,
            refit=True,
            random_state=random_state,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_tune, y_tune)

        for i, params in enumerate(search.cv_results_["params"]):
            config_name = f"{model_name}_config_{i}"
            row = {
                "model": config_name,
                "tuning_balanced_accuracy_mean": float(search.cv_results_["mean_test_score"][i]),
                "tuning_balanced_accuracy_std": float(search.cv_results_["std_test_score"][i]),
                "tuning_train_balanced_accuracy_mean": float(search.cv_results_["mean_train_score"][i]),
                "best_params": json.dumps(params, ensure_ascii=True),
                "random_search_iterations": int(n_iter),
                "tuning_samples": int(len(X_tune)),
            }
            rows.append(row)
            
            cloned_pipeline = clone(pipeline).set_params(**params)
            tuned_pipelines[config_name] = cloned_pipeline

        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_model_name = f"{model_name}_config_{search.best_index_}"
            best_pipeline = search.best_estimator_
            best_params = search.best_params_

    if best_pipeline is None:
        raise RuntimeError("Random search did not produce a best estimator.")

    return rows, best_model_name, best_pipeline, best_params, tuned_pipelines


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, object],
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str,
    cv_splits: int,
    random_state: int,
    use_feature_selection: bool = False,
) -> pd.DataFrame:
    scorer = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows: list[dict[str, float | str]] = []
    for model_name, estimator in models.items():
        pipeline = build_pipeline(
            estimator,
            feature_columns,
            genotype_type=genotype_type,
            use_feature_selection=use_feature_selection,
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            return_train_score=True,
        )
        row: dict[str, float | str] = {
            "model": model_name,
            "best_params": "{}",
            "random_search_iterations": 0,
            "tuning_samples": int(len(X_train)),
        }
        for metric in scorer:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
            row[f"train_{metric}_mean"] = float(np.mean(scores[f"train_{metric}"]))
            row[f"train_{metric}_std"] = float(np.std(scores[f"train_{metric}"]))
        row["tuning_balanced_accuracy_mean"] = row["balanced_accuracy_mean"]
        row["tuning_balanced_accuracy_std"] = row["balanced_accuracy_std"]
        row["tuning_train_balanced_accuracy_mean"] = row["train_balanced_accuracy_mean"]
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("balanced_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_models_on_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, object],
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str,
    use_feature_selection: bool = False,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for model_name, estimator in models.items():
        if hasattr(estimator, "steps"):
            pipeline = clone(estimator)
        else:
            pipeline = build_pipeline(
                estimator,
                feature_columns,
                genotype_type=genotype_type,
                use_feature_selection=use_feature_selection,
            )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
                "holdout_balanced_accuracy": float(
                    balanced_accuracy_score(y_test, y_pred)
                ),
                "holdout_f1_macro": float(
                    f1_score(y_test, y_pred, average="macro")
                ),
            }
        )
    return rows


def evaluate_holdout(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    id_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    predictions = pd.DataFrame(index=X_test.index)
    predictions["true_label"] = y_test.values
    predictions["predicted_label"] = y_pred

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        for class_label, probability in zip(pipeline.classes_, probabilities.T):
            predictions[f"prob_{class_label}"] = probability

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, predictions


def plot_confusion_matrix(
    matrix: list[list[int]], labels: list[str], output_path: Path
) -> None:
    cell_size = 96
    left_margin = 96
    top_margin = 104
    right_margin = 24
    bottom_margin = 72
    width = left_margin + len(labels) * cell_size + right_margin
    height = top_margin + len(labels) * cell_size + bottom_margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max((max(row) for row in matrix), default=1) or 1

    draw.text(
        (left_margin, 24), "Holdout confusion matrix", fill="black", font=font
    )
    draw.text(
        (left_margin, height - 32), "Predicted label", fill="black", font=font
    )
    draw.text((16, top_margin - 32), "True", fill="black", font=font)

    for index, label in enumerate(labels):
        x = left_margin + index * cell_size
        y = top_margin + index * cell_size
        draw.text((x + 28, top_margin - 28), label, fill="black", font=font)
        draw.text((24, y + 40), label, fill="black", font=font)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = int(235 - (value / max_value) * 170)
            fill = (intensity, intensity + 10, 255)
            x0 = left_margin + column_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle(
                (x0, y0, x1, y1), fill=fill, outline=(120, 135, 160)
            )
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            color = "white" if value > max_value / 2 else "black"
            draw.text(
                (
                    x0 + (cell_size - text_width) / 2,
                    y0 + (cell_size - text_height) / 2,
                ),
                text,
                fill=color,
                font=font,
            )

    image.save(output_path)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
