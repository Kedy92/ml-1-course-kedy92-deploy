from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import torch

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    ARTIFACTS_DIR,
    CATEGORICAL_COLUMNS,
    MODELS_DIR,
    NOTEBOOK_ASSETS_DIR,
    NUMERICAL_COLUMNS,
    RANDOM_STATE,
)
from src.data_utils import (  # noqa: E402
    build_mlp_metadata,
    encode_mlp_frame,
    load_credit_default_data,
    prepare_tree_frame,
    save_joblib,
    split_credit_default_data,
)
from src.metrics_utils import (  # noqa: E402
    metrics_table,
    plot_confusion,
    plot_feature_importance,
    plot_training_history,
    save_metrics_json,
)
from src.mlp_pipeline import CreditDefaultDataset, train_mlp_configs  # noqa: E402
from src.tree_pipeline import (  # noqa: E402
    shap_values_for_sample,
    train_catboost_model,
    train_random_forest_baseline,
    tree_feature_importance,
)


def ensure_dirs() -> None:
    for directory in [ARTIFACTS_DIR, MODELS_DIR, NOTEBOOK_ASSETS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    torch.manual_seed(RANDOM_STATE)

    df = load_credit_default_data()
    splits = split_credit_default_data(df)

    split_summary = {
        "train_rows": len(splits.train),
        "val_rows": len(splits.val),
        "test_rows": len(splits.test),
        "target_rate_train": float(splits.train["default_next_month"].mean()),
        "target_rate_val": float(splits.val["default_next_month"].mean()),
        "target_rate_test": float(splits.test["default_next_month"].mean()),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
    }
    save_metrics_json(split_summary, ARTIFACTS_DIR / "split_summary.json")

    metadata = build_mlp_metadata(splits.train)
    train_cat, train_num, train_y = encode_mlp_frame(splits.train, metadata)
    val_cat, val_num, val_y = encode_mlp_frame(splits.val, metadata)
    test_cat, test_num, test_y = encode_mlp_frame(splits.test, metadata)

    mlp_train_ds = CreditDefaultDataset(train_cat, train_num, train_y)
    mlp_val_ds = CreditDefaultDataset(val_cat, val_num, val_y)
    mlp_test_ds = CreditDefaultDataset(test_cat, test_num, test_y)

    pos_weight = float((train_y == 0).sum() / max((train_y == 1).sum(), 1))
    cardinalities = [len(metadata["inverse_category_maps"][col]) for col in CATEGORICAL_COLUMNS]
    mlp_configs = [
        {
            "name": "baseline",
            "hidden_dims": [64, 32],
            "dropout": 0.10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 20,
            "patience": 4,
        },
        {
            "name": "wider",
            "hidden_dims": [128, 64],
            "dropout": 0.20,
            "learning_rate": 8e-4,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 24,
            "patience": 5,
        },
        {
            "name": "deeper",
            "hidden_dims": [128, 64, 32],
            "dropout": 0.25,
            "learning_rate": 6e-4,
            "weight_decay": 5e-4,
            "batch_size": 512,
            "epochs": 24,
            "patience": 5,
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_tuning_rows, best_mlp = train_mlp_configs(
        train_dataset=mlp_train_ds,
        val_dataset=mlp_val_ds,
        test_dataset=mlp_test_ds,
        cardinalities=cardinalities,
        num_numerical_features=len(NUMERICAL_COLUMNS),
        pos_weight=pos_weight,
        device=device,
        configs=mlp_configs,
    )
    pd.DataFrame(mlp_tuning_rows).to_csv(ARTIFACTS_DIR / "mlp_tuning_results.csv", index=False)
    plot_training_history(best_mlp.history, NOTEBOOK_ASSETS_DIR / "mlp_training_history.png")
    plot_confusion(
        test_y,
        best_mlp.test_probabilities,
        "MLP Confusion Matrix",
        NOTEBOOK_ASSETS_DIR / "mlp_confusion_matrix.png",
    )

    metadata["best_config"] = best_mlp.config
    metadata["cardinalities"] = cardinalities
    metadata["pos_weight"] = pos_weight
    metadata["threshold"] = best_mlp.test_metrics["threshold"]
    save_joblib(metadata, MODELS_DIR / "mlp_metadata.joblib")
    torch.save(best_mlp.model.state_dict(), MODELS_DIR / "mlp_model.pt")
    save_metrics_json(best_mlp.val_metrics, ARTIFACTS_DIR / "mlp_val_metrics.json")
    save_metrics_json(best_mlp.test_metrics, ARTIFACTS_DIR / "mlp_test_metrics.json")

    train_tree_X, train_tree_y = prepare_tree_frame(splits.train)
    val_tree_X, val_tree_y = prepare_tree_frame(splits.val)
    test_tree_X, test_tree_y = prepare_tree_frame(splits.test)

    rf_model, rf_metrics = train_random_forest_baseline(train_tree_X, train_tree_y, test_tree_X, test_tree_y)
    save_joblib(rf_model, MODELS_DIR / "random_forest_baseline.joblib")
    save_metrics_json(rf_metrics, ARTIFACTS_DIR / "random_forest_metrics.json")

    catboost_model, catboost_metrics, catboost_probs = train_catboost_model(
        train_tree_X,
        train_tree_y,
        val_tree_X,
        val_tree_y,
        test_tree_X,
        test_tree_y,
    )
    joblib.dump(catboost_model, MODELS_DIR / "catboost_model.joblib")
    save_metrics_json(catboost_metrics, ARTIFACTS_DIR / "catboost_metrics.json")
    plot_confusion(
        test_tree_y,
        catboost_probs,
        "CatBoost Confusion Matrix",
        NOTEBOOK_ASSETS_DIR / "catboost_confusion_matrix.png",
    )

    importances = tree_feature_importance(catboost_model, ALL_FEATURE_COLUMNS)
    importances.to_csv(ARTIFACTS_DIR / "catboost_feature_importance.csv", header=["importance"])
    plot_feature_importance(
        importances,
        "Top CatBoost Feature Importances",
        NOTEBOOK_ASSETS_DIR / "catboost_feature_importance.png",
        top_n=15,
    )

    shap_sample = val_tree_X.sample(n=min(800, len(val_tree_X)), random_state=RANDOM_STATE).copy()
    shap_explanation = shap_values_for_sample(catboost_model, shap_sample)

    plt.figure()
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(NOTEBOOK_ASSETS_DIR / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(NOTEBOOK_ASSETS_DIR / "shap_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    comparison = metrics_table(
        [
            {"model": "MLP (best config)", **best_mlp.test_metrics},
            {"model": "Random Forest", **rf_metrics},
            {"model": "CatBoost", **catboost_metrics},
        ]
    )
    comparison.to_csv(ARTIFACTS_DIR / "comparison_table.csv", index=False)

    summary = {
        "dataset": "UCI Default of Credit Card Clients",
        "feature_count": len(ALL_FEATURE_COLUMNS),
        "categorical_features": CATEGORICAL_COLUMNS,
        "numerical_features": NUMERICAL_COLUMNS,
        "best_mlp_config": best_mlp.config,
        "mlp_test_metrics": best_mlp.test_metrics,
        "random_forest_metrics": rf_metrics,
        "catboost_metrics": catboost_metrics,
    }
    (ARTIFACTS_DIR / "project_summary.json").write_text(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
