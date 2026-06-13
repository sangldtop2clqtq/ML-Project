import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR
from .data import load_genotype_table, metadata_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ancestry labels for genotype data.")
    parser.add_argument("--config", type=Path, help="JSON config for a task, e.g. configs/snp_pop.json.")
    parser.add_argument("--genotype-type", choices=("str", "snp"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = resolve_args(parse_args())
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_path)
    df = load_genotype_table(args.data_path, genotype_type=args.genotype_type)
    X = df.drop(columns=[column for column in ("POP", "SUBPOP") if column in df.columns])

    predictions = df[metadata_columns(df)].copy()
    predictions["predicted_label"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for class_label, probability in zip(model.classes_, probabilities.T):
            predictions[f"prob_{class_label}"] = probability

    predictions.to_csv(args.output_path, index=False)
    print(f"Predictions saved to: {args.output_path}")


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config: dict[str, object] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    args.genotype_type = args.genotype_type or str(config.get("genotype_type", "str"))
    args.data_path = args.data_path or Path(str(config.get("data_path", DEFAULT_DATA_PATH)))
    default_model_dir = Path(str(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
    default_report_dir = Path(str(config.get("report_dir", DEFAULT_REPORT_DIR)))
    args.model_path = args.model_path or default_model_dir / f"ancestry_{args.genotype_type}_model.joblib"
    args.output_path = args.output_path or default_report_dir / "predictions.csv"
    return args


if __name__ == "__main__":
    main()
