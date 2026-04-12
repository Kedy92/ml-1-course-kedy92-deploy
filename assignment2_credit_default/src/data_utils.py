from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    EDUCATION_MAP,
    MARRIAGE_MAP,
    NUMERICAL_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SEX_MAP,
    TARGET_COLUMN,
)


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_credit_default_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH).rename(
        columns={
            "LIMIT_BAL": "limit_bal",
            "SEX": "sex",
            "EDUCATION": "education",
            "MARRIAGE": "marriage",
            "AGE": "age",
            "PAY_0": "pay_0",
            "PAY_2": "pay_2",
            "PAY_3": "pay_3",
            "PAY_4": "pay_4",
            "PAY_5": "pay_5",
            "PAY_6": "pay_6",
            "BILL_AMT1": "bill_amt1",
            "BILL_AMT2": "bill_amt2",
            "BILL_AMT3": "bill_amt3",
            "BILL_AMT4": "bill_amt4",
            "BILL_AMT5": "bill_amt5",
            "BILL_AMT6": "bill_amt6",
            "PAY_AMT1": "pay_amt1",
            "PAY_AMT2": "pay_amt2",
            "PAY_AMT3": "pay_amt3",
            "PAY_AMT4": "pay_amt4",
            "PAY_AMT5": "pay_amt5",
            "PAY_AMT6": "pay_amt6",
            "default payment next month": TARGET_COLUMN,
            "ID": "id",
        }
    )

    df["sex"] = df["sex"].map(SEX_MAP).fillna("unknown")
    df["education"] = df["education"].map(EDUCATION_MAP).fillna("unknown")
    df["marriage"] = df["marriage"].map(MARRIAGE_MAP).fillna("unknown")
    df = df.drop(columns=["id"])
    return df


def split_credit_default_data(df: pd.DataFrame) -> DataSplits:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return DataSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def build_mlp_metadata(train_df: pd.DataFrame) -> dict:
    category_maps: dict[str, dict[str, int]] = {}
    inverse_category_maps: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLUMNS:
        ordered = sorted(train_df[col].astype(str).unique().tolist())
        mapping = {"__UNK__": 0}
        mapping.update({value: idx + 1 for idx, value in enumerate(ordered)})
        category_maps[col] = mapping
        inverse_category_maps[col] = ["__UNK__"] + ordered

    means = train_df[NUMERICAL_COLUMNS].mean()
    stds = train_df[NUMERICAL_COLUMNS].std().replace(0, 1.0)

    return {
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "category_maps": category_maps,
        "inverse_category_maps": inverse_category_maps,
        "numerical_means": means.to_dict(),
        "numerical_stds": stds.to_dict(),
        "target_column": TARGET_COLUMN,
        "target_names": {0: "no_default", 1: "default"},
    }


def encode_mlp_frame(df: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical_arrays = []
    for col in metadata["categorical_columns"]:
        mapping = metadata["category_maps"][col]
        categorical_arrays.append(
            df[col].astype(str).map(lambda value: mapping.get(value, 0)).to_numpy(dtype=np.int64)
        )
    categorical_array = np.column_stack(categorical_arrays)

    means = pd.Series(metadata["numerical_means"])
    stds = pd.Series(metadata["numerical_stds"])
    numerical_frame = (df[metadata["numerical_columns"]] - means) / stds
    numerical_array = numerical_frame.to_numpy(dtype=np.float32)

    labels = df[metadata["target_column"]].to_numpy(dtype=np.float32)
    return categorical_array, numerical_array, labels


def prepare_tree_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = df[ALL_FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        features[col] = features[col].astype("category")
    labels = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return features, labels


def save_joblib(obj: object, path) -> None:
    joblib.dump(obj, path)
