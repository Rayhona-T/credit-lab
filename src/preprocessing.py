"""Train/test split and model-specific preprocessing pipelines.

KNN uses a StandardScaler fitted only on training data. Tree models use the
raw integer-encoded UCI features, matching the supplied Decision Tree artifact.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.constants import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE


def make_train_test_split(x, y, *, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE):
    """Stratified split so the default-rate is preserved in both partitions."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_knn_pipeline(n_neighbors: int = 15) -> Pipeline:
    scaler = ColumnTransformer(
        [("scale", StandardScaler(), list(FEATURE_COLUMNS))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", scaler),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def build_random_forest(
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def frame_for_model(record: dict) -> "pd.DataFrame":
    import pandas as pd

    missing = [col for col in FEATURE_COLUMNS if col not in record]
    extra = [key for key in record if key not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if extra:
        raise ValueError(f"Unexpected features: {extra}")
    return pd.DataFrame([[record[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
