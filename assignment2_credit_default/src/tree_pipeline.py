from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from .metrics_utils import classification_metrics


def train_random_forest_baseline(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    test_features: pd.DataFrame,
    test_targets: np.ndarray,
) -> tuple[Pipeline, dict]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numerical", "passthrough", NUMERICAL_COLUMNS),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_features, train_targets)
    probabilities = model.predict_proba(test_features)[:, 1]
    metrics = classification_metrics(test_targets, probabilities)
    return model, metrics


def train_catboost_model(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    val_features: pd.DataFrame,
    val_targets: np.ndarray,
    test_features: pd.DataFrame,
    test_targets: np.ndarray,
) -> tuple[CatBoostClassifier, dict, np.ndarray]:
    scale_pos_weight = float((train_targets == 0).sum() / max((train_targets == 1).sum(), 1))
    categorical_indices = [train_features.columns.get_loc(col) for col in CATEGORICAL_COLUMNS]
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bagging_temperature=0.5,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=False,
    )
    model.fit(
        train_features,
        train_targets,
        cat_features=categorical_indices,
        eval_set=(val_features, val_targets),
        use_best_model=True,
        early_stopping_rounds=50,
    )
    probabilities = model.predict_proba(test_features)[:, 1]
    metrics = classification_metrics(test_targets, probabilities)
    return model, metrics, probabilities


def tree_feature_importance(model: CatBoostClassifier, feature_names: list[str]) -> pd.Series:
    return pd.Series(model.get_feature_importance(), index=feature_names).sort_values(ascending=False)


def shap_values_for_sample(model: CatBoostClassifier, sample_features: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample_features)
    return explanation
