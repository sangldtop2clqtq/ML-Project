import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import CV_SPLITS, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from .data import load_genotype_table, metadata_columns, validate_genotype_table
from .models import build_pipeline, candidate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ancestry prediction models from STR genotype data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-column", default=TARGET_COLUMN, help="Target label, usually POP or SUBPOP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--model-name", choices=sorted(candidate_models().keys()), help="Train only one model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_genotype_table(args.data_path)
    allele_pairs = validate_genotype_table(df, target_column=args.target_column)

    y = df[args.target_column].astype(str)
    X = df.drop(columns=[args.target_column])

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

    cv_results = evaluate_candidates(X_train, y_train, models, allele_pairs, args.cv_splits, args.random_state)
    cv_results.to_csv(args.report_dir / "cv_results.csv", index=False)

    best_model_name = cv_results.iloc[0]["model"]
    best_pipeline = build_pipeline(models[best_model_name], allele_pairs)
    best_pipeline.fit(X_train, y_train)

    labels = sorted(y.unique())
    metrics, predictions = evaluate_holdout(best_pipeline, X_test, y_test, labels, metadata_columns(X_test))

    model_path = args.output_dir / "ancestry_str_model.joblib"
    joblib.dump(best_pipeline, model_path)

    predictions.to_csv(args.report_dir / "holdout_predictions.csv", index=False)
    save_json(metrics, args.report_dir / "metrics.json")
    save_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_path": str(args.data_path),
            "target_column": args.target_column,
            "model_path": str(model_path),
            "best_model": best_model_name,
            "n_samples": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "labels": labels,
            "allele_pairs": [list(pair) for pair in allele_pairs],
        },
        args.report_dir / "run_metadata.json",
    )
    plot_confusion_matrix(metrics["confusion_matrix"], labels, args.report_dir / "figures" / "confusion_matrix.png")

    print(f"Best model: {best_model_name}")
    print(f"Holdout accuracy: {metrics['accuracy']:.4f}")
    print(f"Holdout balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Artifacts saved to: {args.output_dir} and {args.report_dir}")


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, object],
    allele_pairs: list[tuple[str, str]],
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    scorer = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | str]] = []
    for model_name, estimator in models.items():
        pipeline = build_pipeline(estimator, allele_pairs)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, float | str] = {"model": model_name}
        for metric in scorer:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)


def evaluate_holdout(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    id_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    predictions = X_test[id_columns].copy() if id_columns else pd.DataFrame(index=X_test.index)
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


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Holdout confusion matrix")

    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
