from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = REPO_ROOT / "data" / "credit_card_default"
RAW_DATA_PATH = DATA_DIR / "default_of_credit_card_clients.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOK_ASSETS_DIR = PROJECT_ROOT / "notebook_assets"

RANDOM_STATE = 42
TARGET_COLUMN = "default_next_month"

CATEGORICAL_COLUMNS = ["sex", "education", "marriage"]
NUMERICAL_COLUMNS = [
    "limit_bal",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]

ALL_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

SEX_MAP = {
    1: "male",
    2: "female",
}

EDUCATION_MAP = {
    0: "unknown",
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "other",
    5: "unknown",
    6: "unknown",
}

MARRIAGE_MAP = {
    0: "unknown",
    1: "married",
    2: "single",
    3: "other",
}
