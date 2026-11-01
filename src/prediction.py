"""Validated inference for hypothetical client records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    FEATURE_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_CLASS,
    POSITIVE_LABEL,
)
from src.preprocessing import frame_for_model


ALLOWED_RANGES = {
    "LIMIT_BAL": (1000, 2_000_000),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    "AGE": (18, 100),
    **{f"PAY_{k}": (-2, 8) for k in (0, 2, 3, 4, 5, 6)},
    **{f"BILL_AMT{i}": (-500_000, 3_000_000) for i in range(1, 7)},
    **{f"PAY_AMT{i}": (0, 3_000_000) for i in range(1, 7)},
}


class InvalidClientInput(ValueError):
    """User-facing validation error for simulator inputs."""


def validate_client_record(record: dict) -> dict[str, float]:
    missing = [col for col in FEATURE_COLUMNS if col not in record]
    if missing:
        raise InvalidClientInput(f"Missing fields: {', '.join(missing)}")
    clean: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        value = record[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise InvalidClientInput(f"{col} is required.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidClientInput(f"{col} must be numeric.") from exc
        low, high = ALLOWED_RANGES[col]
        if numeric < low or numeric > high:
            raise InvalidClientInput(f"{col} must be between {low} and {high}.")
        if col in {"SEX", "EDUCATION", "MARRIAGE"} or col.startswith("PAY_"):
            if not float(numeric).is_integer():
                raise InvalidClientInput(f"{col} must be a whole-number code.")
            clean[col] = int(numeric)
        else:
            clean[col] = numeric
    return clean


def _positive_index(estimator) -> int:
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        final = estimator.named_steps[list(estimator.named_steps)[-1]]
        classes = getattr(final, "classes_", None)
    classes = list(classes) if classes is not None else [0, 1]
    return classes.index(POSITIVE_CLASS) if POSITIVE_CLASS in classes else 1


def predict_client(estimator, record: dict, *, model_key: str) -> dict[str, Any]:
    clean = validate_client_record(record)
    frame = frame_for_model(clean)
    predicted = int(estimator.predict(frame)[0])
    probability = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(frame)[0]
        probability = float(proba[_positive_index(estimator)])
    return {
        "model_key": model_key,
        "predicted_class": predicted,
        "predicted_label": POSITIVE_LABEL if predicted == POSITIVE_CLASS else NEGATIVE_LABEL,
        "default_probability": probability,
        "supports_probability": probability is not None,
        "features": clean,
    }


def compare_predictions(original: dict[str, Any], modified: dict[str, Any]) -> dict[str, Any]:
    changed = [
        col
        for col in FEATURE_COLUMNS
        if original["features"][col] != modified["features"][col]
    ]
    delta = None
    if original["default_probability"] is not None and modified["default_probability"] is not None:
        delta = modified["default_probability"] - original["default_probability"]
    return {
        "changed_features": changed,
        "class_changed": original["predicted_class"] != modified["predicted_class"],
        "probability_delta": delta,
    }


def records_equal(left: dict, right: dict) -> bool:
    return all(left.get(col) == right.get(col) for col in FEATURE_COLUMNS)


def to_feature_frame(records: list[dict]) -> pd.DataFrame:
    cleaned = [validate_client_record(row) for row in records]
    return pd.DataFrame(cleaned, columns=FEATURE_COLUMNS)
