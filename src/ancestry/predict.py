import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR
from .data import load_genotype_table, metadata_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ancestry labels for STR genotype data.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_OUTPUT_DIR / "ancestry_str_model.joblib")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=Path("reports") / "predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_path)
    df = load_genotype_table(args.data_path)
    X = df.drop(columns=[column for column in ("POP", "SUBPOP") if column in df.columns])

    predictions = df[metadata_columns(df)].copy()
    predictions["predicted_label"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for class_label, probability in zip(model.classes_, probabilities.T):
            predictions[f"prob_{class_label}"] = probability

    predictions.to_csv(args.output_path, index=False)
    print(f"Predictions saved to: {args.output_path}")


if __name__ == "__main__":
    main()

