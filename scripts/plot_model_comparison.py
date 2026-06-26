"""
plot_model_comparison.py
========================
Ve learning curves (duong + bong mean +/- std) cho tung mo hinh, tung task.

Output luu tai:
  reports/<task>/figures/learning_curves.png    (moi task)
  reports/figures/learning_curves_all_tasks.png (tong hop 4 task)

Chay:
  python scripts/plot_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ancestry.data.str_loader import load_str_data
from ancestry.data.snp_loader import load_snp_data
from ancestry.data import find_allele_pairs
from ancestry.models import build_pipeline
from ancestry.config import RANDOM_STATE

# ---------------------------------------------------------------------------
# Cau hinh
# ---------------------------------------------------------------------------
TASKS = [
    {
        "name":     "str_pop",
        "label":    "STR -> POP",
        "genotype": "str",
        "data":     PROJECT_ROOT / "data/interim/str/str_genotypes_cleaned.csv",
        "target":   "POP",
    },
    {
        "name":     "str_subpop",
        "label":    "STR -> SUBPOP",
        "genotype": "str",
        "data":     PROJECT_ROOT / "data/interim/str/str_genotypes_cleaned.csv",
        "target":   "SUBPOP",
    },
    {
        "name":     "snp_pop",
        "label":    "SNP -> POP",
        "genotype": "snp",
        "data":     PROJECT_ROOT / "data/interim/snp/model_train_data.csv",
        "target":   "POP",
    },
    {
        "name":     "snp_subpop",
        "label":    "SNP -> SUBPOP",
        "genotype": "snp",
        "data":     PROJECT_ROOT / "data/interim/snp/model_train_data.csv",
        "target":   "SUBPOP",
    },
]

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "extra_trees":         "Extra Trees",
}

MODEL_COLORS = {
    "logistic_regression": "#4C72B0",
    "random_forest":       "#DD8452",
    "extra_trees":         "#55A868",
}

# Mo hinh nhe hon de learning_curve chay nhanh
def fast_models(random_state: int = RANDOM_STATE) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, n_jobs=-1, random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            class_weight="balanced", n_estimators=100, n_jobs=-1, random_state=random_state,
        ),
    }

TRAIN_SIZES = np.linspace(0.15, 1.0, 7)  # 7 diem, bat dau tu 15%
CV_SPLITS   = 3                            # 3-fold de giam thoi gian

# ---------------------------------------------------------------------------
# Load du lieu
# ---------------------------------------------------------------------------

def load_task_data(task: dict):
    data_path = str(task["data"])
    if not Path(data_path).exists():
        return None, None, None

    if task["genotype"] == "str":
        X, y = load_str_data(data_path, task["target"])
        feature_columns = find_allele_pairs(X.columns)
    else:
        X, y = load_snp_data(data_path, task["target"])
        feature_columns = list(X.columns)

    return X, y, feature_columns


# ---------------------------------------------------------------------------
# Ve learning curve 1 task tren 1 Axes
# ---------------------------------------------------------------------------

def draw_learning_curves(ax: plt.Axes, task: dict) -> bool:
    X, y, feature_columns = load_task_data(task)
    if X is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return False

    use_fs = (task["genotype"] == "snp")
    models = fast_models(random_state=RANDOM_STATE)
    cv     = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    ax.set_facecolor("#F7F7F7")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    for model_key, estimator in models.items():
        color = MODEL_COLORS[model_key]
        label = MODEL_DISPLAY[model_key]

        pipeline = build_pipeline(
            estimator, feature_columns,
            genotype_type=task["genotype"],
            use_feature_selection=use_fs,
        )

        sizes_abs, _, val_scores = learning_curve(
            pipeline, X, y,
            train_sizes=TRAIN_SIZES,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        val_mean = val_scores.mean(axis=1)
        val_std  = val_scores.std(axis=1)

        ax.plot(
            sizes_abs, val_mean,
            color=color, linewidth=2.2,
            marker="o", markersize=5,
            label=label, zorder=4,
        )
        ax.fill_between(
            sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            color=color, alpha=0.18, zorder=3,
        )

    ax.set_title(f"Learning Curves — {task['label']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training examples", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


# ---------------------------------------------------------------------------
# Sinh bieu do tung task
# ---------------------------------------------------------------------------

def plot_single_task(task: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")

    draw_learning_curves(ax, task)

    plt.tight_layout()
    fig_dir = REPORTS_ROOT / task["name"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Da luu: {out_path}")


# ---------------------------------------------------------------------------
# Sinh bieu do tong hop 2x2
# ---------------------------------------------------------------------------

def plot_all_tasks_grid() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Learning Curves — Model Comparison (All Tasks)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, task in zip(axes.flatten(), TASKS):
        draw_learning_curves(ax, task)

    plt.tight_layout()
    out_dir = REPORTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "learning_curves_all_tasks.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Da luu tong hop: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Ve Learning Curves -- ML-Project")
    print("=" * 60)

    for task in TASKS:
        print(f"\n[Task: {task['name']}]")
        plot_single_task(task)

    print("\n[Tong hop 4 task (2x2)]")
    plot_all_tasks_grid()

    print("\n" + "=" * 60)
    print("  Hoan tat!")
    print("=" * 60)


if __name__ == "__main__":
    main()
