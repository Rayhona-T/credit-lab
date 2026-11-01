"""Shared schema, encodings, and evaluation settings for CREDIT LAB.

Feature names match the UCI Default of Credit Card Clients dataset and the
existing `credit_model.joblib` artifact (`feature_names_in_`).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "credit_model.joblib"

UCI_DATASET_ID = 350
UCI_DATASET_NAME = "Default of Credit Card Clients"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_XLS_NAME = "default of credit card clients.xls"
LOCAL_CSV_NAME = "default_of_credit_card_clients.csv"

ID_COLUMN = "ID"
TARGET_COLUMN = "default payment next month"
POSITIVE_CLASS = 1
POSITIVE_LABEL = "Default"
NEGATIVE_LABEL = "No default"

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Month labels follow the original Yeh & Lien documentation (Apr–Sep 2005).
PAY_STATUS_MONTHS = {
    "PAY_0": "September 2005",
    "PAY_2": "August 2005",
    "PAY_3": "July 2005",
    "PAY_4": "June 2005",
    "PAY_5": "May 2005",
    "PAY_6": "April 2005",
}
BILL_MONTHS = {
    "BILL_AMT1": "September 2005",
    "BILL_AMT2": "August 2005",
    "BILL_AMT3": "July 2005",
    "BILL_AMT4": "June 2005",
    "BILL_AMT5": "May 2005",
    "BILL_AMT6": "April 2005",
}
PAY_AMT_MONTHS = {
    "PAY_AMT1": "September 2005",
    "PAY_AMT2": "August 2005",
    "PAY_AMT3": "July 2005",
    "PAY_AMT4": "June 2005",
    "PAY_AMT5": "May 2005",
    "PAY_AMT6": "April 2005",
}

# Dataset documentation is incomplete for some codes. Keep undocumented values
# visible rather than silently remapping them.
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    0: "Undocumented (0)",
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    5: "Unknown (5)",
    6: "Unknown (6)",
}
MARRIAGE_LABELS = {
    0: "Undocumented (0)",
    1: "Married",
    2: "Single",
    3: "Others",
}
PAY_STATUS_LABELS = {
    -2: "No consumption",
    -1: "Paid in full",
    0: "Revolving credit",
    1: "Delay 1 month",
    2: "Delay 2 months",
    3: "Delay 3 months",
    4: "Delay 4 months",
    5: "Delay 5 months",
    6: "Delay 6 months",
    7: "Delay 7 months",
    8: "Delay 8 months",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_KNN = "knn"
MODEL_TREE = "decision_tree"
MODEL_FOREST = "random_forest"
MODEL_KEYS = (MODEL_KNN, MODEL_TREE, MODEL_FOREST)

MODEL_DISPLAY_NAMES = {
    MODEL_KNN: "K-Nearest Neighbors",
    MODEL_TREE: "Decision Tree",
    MODEL_FOREST: "Random Forest",
}

MODEL_SHORT_NAMES = {
    MODEL_KNN: "KNN",
    MODEL_TREE: "Decision Tree",
    MODEL_FOREST: "Random Forest",
}

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}
PRIMARY_METRIC = "f1"

KNN_JOBLIB = MODELS_DIR / "knn_pipeline.joblib"
FOREST_JOBLIB = MODELS_DIR / "random_forest.joblib"
EVALUATION_JSON = MODELS_DIR / "evaluation.json"
