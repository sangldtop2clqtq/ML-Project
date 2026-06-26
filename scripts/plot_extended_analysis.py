"""
plot_extended_analysis.py
=========================
Sinh cac bieu do phan tich mo hinh chi tiet cho ca 4 task:

  1. Confusion Matrix dep hon (% theo class)
  2. Per-class Precision / Recall / F1 (bar chart)
  3. Phan phoi xac suat du doan (dung vs sai)
  4. ROC Curve / AUC theo tung class (one-vs-rest)
  5. Class balance (phan phoi nhan trong du lieu)

Output luu tai:
  reports/<task>/figures/confusion_matrix_pct.png
  reports/<task>/figures/per_class_metrics.png
  reports/<task>/figures/prob_distribution.png
  reports/<task>/figures/roc_curves.png
  reports/<task>/figures/class_balance.png

Chay:
  python scripts/plot_extended_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"

TASKS = ["str_pop", "str_subpop", "snp_pop", "snp_subpop"]
TASK_LABELS = {
    "str_pop":    "STR -> POP",
    "str_subpop": "STR -> SUBPOP",
    "snp_pop":    "SNP -> POP",
    "snp_subpop": "SNP -> SUBPOP",
}

# Palette dep cho nhieu class
CLASS_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2",
    "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78",
    "#98DF8A", "#FF9896", "#C5B0D5",
]


def get_color(i: int) -> str:
    return CLASS_PALETTE[i % len(CLASS_PALETTE)]


def load_metrics(task: str) -> dict | None:
    p = REPORTS_ROOT / task / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_predictions(task: str) -> pd.DataFrame | None:
    p = REPORTS_ROOT / task / "holdout_predictions.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


# ===========================================================================
# 1. Confusion Matrix dep (% theo hang = true class)
# ===========================================================================

def plot_confusion_matrix_pct(task: str) -> None:
    data = load_metrics(task)
    if data is None:
        return

    labels = data["labels"]
    matrix = np.array(data["confusion_matrix"], dtype=float)

    # Chuyen sang phan tram theo hang (true class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 1.5), max(5, n * 0.8 + 1.5)))
    fig.patch.set_facecolor("white")

    im = ax.imshow(matrix_pct, cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of True Class", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix (%) — {TASK_LABELS[task]}",
        fontsize=12, fontweight="bold", pad=10,
    )

    for r in range(n):
        for c in range(n):
            pct = matrix_pct[r, c]
            cnt = int(matrix[r, c])
            text_color = "white" if pct > 55 else "black"
            ax.text(
                c, r,
                f"{pct:.1f}%\n({cnt})",
                ha="center", va="center",
                fontsize=max(6, 10 - n // 4),
                color=text_color,
                fontweight="bold" if r == c else "normal",
            )

    plt.tight_layout()
    out = REPORTS_ROOT / task / "figures" / "confusion_matrix_pct.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] confusion_matrix_pct: {out}")


# ===========================================================================
# 2. Per-class Precision / Recall / F1
# ===========================================================================

def plot_per_class_metrics(task: str) -> None:
    data = load_metrics(task)
    if data is None:
        return

    labels    = data["labels"]
    report    = data["classification_report"]

    classes = [lbl for lbl in labels if lbl in report]
    precision = [report[c]["precision"] for c in classes]
    recall    = [report[c]["recall"]    for c in classes]
    f1        = [report[c]["f1-score"]  for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.65 + 2), 5.5))
    fig.patch.set_facecolor("white")

    ax.bar(x - w, precision, w, label="Precision", color="#4C72B0", alpha=0.88, edgecolor="white")
    ax.bar(x,     recall,    w, label="Recall",    color="#DD8452", alpha=0.88, edgecolor="white")
    ax.bar(x + w, f1,        w, label="F1-score",  color="#55A868", alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Per-class Precision / Recall / F1 — {TASK_LABELS[task]}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.9)

    # Duong ngang trung binh macro
    macro = report.get("macro avg", {})
    if macro:
        ax.axhline(macro.get("f1-score", 0), color="#C44E52", linestyle="--",
                   linewidth=1.4, label=f"Macro F1 = {macro['f1-score']:.3f}", alpha=0.8)
        ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = REPORTS_ROOT / task / "figures" / "per_class_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] per_class_metrics: {out}")


# ===========================================================================
# 3. Phan phoi xac suat du doan (dung vs sai)
# ===========================================================================

def plot_prob_distribution(task: str) -> None:
    df = load_predictions(task)
    if df is None:
        return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        print(f"  [SKIP] prob_distribution: khong co cot xac suat cho task {task}")
        return

    # Lay xac suat cua class duoc du doan (max prob)
    df["max_prob"] = df[prob_cols].max(axis=1)
    df["correct"]  = df["true_label"] == df["predicted_label"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    correct_probs   = df.loc[df["correct"],  "max_prob"]
    incorrect_probs = df.loc[~df["correct"], "max_prob"]

    bins = np.linspace(0, 1, 21)

    ax.hist(correct_probs,   bins=bins, color="#55A868", alpha=0.75,
            label=f"Correct ({len(correct_probs)})", edgecolor="white", linewidth=0.5)
    ax.hist(incorrect_probs, bins=bins, color="#C44E52", alpha=0.75,
            label=f"Incorrect ({len(incorrect_probs)})", edgecolor="white", linewidth=0.5)

    ax.axvline(correct_probs.mean(),   color="#55A868", linestyle="--",
               linewidth=1.8, label=f"Mean correct = {correct_probs.mean():.3f}")
    ax.axvline(incorrect_probs.mean(), color="#C44E52", linestyle="--",
               linewidth=1.8, label=f"Mean incorrect = {incorrect_probs.mean():.3f}")

    ax.set_xlabel("Max Predicted Probability", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Probability Distribution (Correct vs Incorrect) — {TASK_LABELS[task]}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = REPORTS_ROOT / task / "figures" / "prob_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] prob_distribution: {out}")


# ===========================================================================
# 4. ROC Curves (one-vs-rest, tung class)
# ===========================================================================

def plot_roc_curves(task: str) -> None:
    df   = load_predictions(task)
    data = load_metrics(task)
    if df is None or data is None:
        return

    labels    = data["labels"]
    prob_cols = [f"prob_{lbl}" for lbl in labels]

    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        print(f"  [SKIP] roc_curves: thieu cot {missing} cho task {task}")
        return

    y_true = label_binarize(df["true_label"], classes=labels)
    y_prob = df[prob_cols].values

    n_classes = len(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")

    for i, lbl in enumerate(labels):
        if y_true.shape[1] <= 1:
            break
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=get_color(i), linewidth=2.0,
                label=f"{lbl} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"ROC Curves (One-vs-Rest) — {TASK_LABELS[task]}",
        fontsize=12, fontweight="bold", pad=10,
    )
    # Dat legend o ngoai neu qua nhieu class
    if n_classes > 8:
        ax.legend(fontsize=7.5, framealpha=0.9, ncol=2, loc="lower right")
    else:
        ax.legend(fontsize=9, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    out = REPORTS_ROOT / task / "figures" / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] roc_curves: {out}")


# ===========================================================================
# 5. Class balance
# ===========================================================================

def plot_class_balance(task: str) -> None:
    df = load_predictions(task)
    if df is None:
        return

    counts = df["true_label"].value_counts().sort_index()
    n      = len(counts)
    colors = [get_color(i) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Class Balance (Holdout Set) — {TASK_LABELS[task]}",
        fontsize=12, fontweight="bold",
    )

    # --- Bar chart ---
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.7)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + counts.values.max() * 0.01,
            str(val),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n > 10:
        ax.set_xticklabels(counts.index, rotation=60, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=9)
    ax.set_title("Sample Count per Class", fontsize=11, fontweight="bold")

    # --- Pie chart ---
    ax2 = axes[1]
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    # Neu qua nhieu class, bo label tren pie
    if n > 12:
        wedges, _ = ax2.pie(
            counts.values, colors=colors,
            wedgeprops=wedge_props, startangle=90,
        )
        ax2.legend(
            wedges, [f"{lbl} ({v})" for lbl, v in zip(counts.index, counts.values)],
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7.5,
        )
    else:
        ax2.pie(
            counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", pctdistance=0.8,
            wedgeprops=wedge_props, startangle=90,
            textprops={"fontsize": 9},
        )
    ax2.set_title("Proportion per Class", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = REPORTS_ROOT / task / "figures" / "class_balance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] class_balance: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("  Extended Analysis Plots -- ML-Project")
    print("=" * 60)

    plot_funcs = [
        ("Confusion Matrix (%)",          plot_confusion_matrix_pct),
        ("Per-class Precision/Recall/F1", plot_per_class_metrics),
        ("Probability Distribution",       plot_prob_distribution),
        ("ROC Curves",                     plot_roc_curves),
        ("Class Balance",                  plot_class_balance),
    ]

    for task in TASKS:
        print(f"\n[Task: {task}]")
        for name, fn in plot_funcs:
            try:
                fn(task)
            except Exception as e:
                print(f"  [ERR] {name}: {e}")

    print("\n" + "=" * 60)
    print("  Hoan tat!")
    print("=" * 60)


if __name__ == "__main__":
    main()
