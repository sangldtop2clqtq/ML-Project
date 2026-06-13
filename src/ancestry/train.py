from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import CV_SPLITS, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from .data import load_genotype_table, metadata_columns, validate_table_for_genotype
from .models import build_pipeline, candidate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ancestry prediction models from genotype data.")
    parser.add_argument("--config", type=Path, help="JSON config for a task, e.g. configs/snp_pop.json.")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--target-column", help="Target label, usually POP or SUBPOP.")
    parser.add_argument("--genotype-type", choices=("str", "snp"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--cv-splits", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--model-name", choices=sorted(candidate_models().keys()), help="Train only one model.")
    return parser.parse_args()


def main() -> None:
    args = resolve_args(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_genotype_table(args.data_path, genotype_type=args.genotype_type)
    feature_columns = validate_table_for_genotype(df, genotype_type=args.genotype_type, target_column=args.target_column)

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

    cv_results = evaluate_candidates(
        X_train,
        y_train,
        models,
        feature_columns,
        args.genotype_type,
        args.cv_splits,
        args.random_state,
    )
    cv_results.to_csv(args.report_dir / "cv_results.csv", index=False)

    best_model_name = cv_results.iloc[0]["model"]
    best_pipeline = build_pipeline(models[best_model_name], feature_columns, genotype_type=args.genotype_type)
    best_pipeline.fit(X_train, y_train)

    labels = sorted(y.unique())
    metrics, predictions = evaluate_holdout(best_pipeline, X_test, y_test, labels, metadata_columns(X_test))

    model_path = args.output_dir / f"ancestry_{args.genotype_type}_model.joblib"
    joblib.dump(best_pipeline, model_path)

    predictions.to_csv(args.report_dir / "holdout_predictions.csv", index=False)
    save_json(metrics, args.report_dir / "metrics.json")
    save_json(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_path": str(args.data_path),
            "genotype_type": args.genotype_type,
            "target_column": args.target_column,
            "model_path": str(model_path),
            "best_model": best_model_name,
            "n_samples": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "labels": labels,
            "features": [list(item) if isinstance(item, tuple) else item for item in feature_columns],
        },
        args.report_dir / "run_metadata.json",
    )
    plot_confusion_matrix(metrics["confusion_matrix"], labels, args.report_dir / "figures" / "confusion_matrix.png")

    print(f"Best model: {best_model_name}")
    print(f"Holdout accuracy: {metrics['accuracy']:.4f}")
    print(f"Holdout balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Artifacts saved to: {args.output_dir} and {args.report_dir}")


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config: dict[str, object] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    args.genotype_type = args.genotype_type or str(config.get("genotype_type", "str"))
    args.data_path = args.data_path or Path(str(config.get("data_path", DEFAULT_DATA_PATH)))
    args.target_column = args.target_column or str(config.get("target_column", TARGET_COLUMN))
    args.output_dir = args.output_dir or Path(str(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
    args.report_dir = args.report_dir or Path(str(config.get("report_dir", DEFAULT_REPORT_DIR)))
    args.test_size = args.test_size if args.test_size is not None else float(config.get("test_size", TEST_SIZE))
    args.cv_splits = args.cv_splits if args.cv_splits is not None else int(config.get("cv_splits", CV_SPLITS))
    args.random_state = args.random_state if args.random_state is not None else int(config.get("random_state", RANDOM_STATE))
    return args


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, object],
    feature_columns: list[tuple[str, str]] | list[str],
    genotype_type: str,
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
        pipeline = build_pipeline(estimator, feature_columns, genotype_type=genotype_type)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=1,
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

    draw.text((left_margin, 24), "Holdout confusion matrix", fill="black", font=font)
    draw.text((left_margin, height - 32), "Predicted label", fill="black", font=font)
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
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(120, 135, 160))
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            color = "white" if value > max_value / 2 else "black"
            draw.text(
                (x0 + (cell_size - text_width) / 2, y0 + (cell_size - text_height) / 2),
                text,
                fill=color,
                font=font,
            )

    image.save(output_path)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
