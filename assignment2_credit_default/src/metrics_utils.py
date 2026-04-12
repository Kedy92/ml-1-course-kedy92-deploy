from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "classification_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
    return metrics


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    metric_order = ["model", "accuracy", "precision", "recall", "f1", "auc_roc"]
    return table[metric_order].sort_values("auc_roc", ascending=False).reset_index(drop=True)


def save_metrics_json(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def plot_confusion(y_true: np.ndarray, probabilities: np.ndarray, title: str, path: Path, threshold: float = 0.5) -> None:
    preds = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        preds,
        display_labels=["No Default", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_history(history: list[dict], path: Path) -> None:
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    axes[0].set_title("MLP Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_auc"], label="Val AUC")
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="Val F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, title: str, path: Path, top_n: int = 15) -> None:
    top = importances.sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, orient="h", ax=ax, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
