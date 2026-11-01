"""Load the supplied Decision Tree and train KNN / Random Forest for comparison.

The original `credit_model.joblib` is never overwritten. Inspection showed it is
a DecisionTreeClassifier(max_depth=3, random_state=42) trained on the same
stratified 80/20 split used here, so its test metrics are comparable to the
newly trained models.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.constants import (
    EVALUATION_JSON,
    FEATURE_COLUMNS,
    FOREST_JOBLIB,
    KNN_JOBLIB,
    MODEL_FOREST,
    MODEL_KNN,
    MODEL_TREE,
    MODELS_DIR,
    ORIGINAL_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_credit_dataset, split_xy, validate_dataset
from src.evaluation import evaluate_estimator, select_best_model
from src.preprocessing import build_knn_pipeline, build_random_forest, make_train_test_split


class ModelLoadError(RuntimeError):
    """Raised when the supplied artifact cannot be used safely."""


def inspect_original_model(path=ORIGINAL_MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ModelLoadError(f"Original model not found at {path}")
    model = joblib.load(path)
    info = {
        "path": str(path),
        "type": type(model).__name__,
        "module": type(model).__module__,
        "is_pipeline": hasattr(model, "named_steps"),
        "has_predict": hasattr(model, "predict"),
        "has_predict_proba": hasattr(model, "predict_proba"),
        "n_features_in_": int(getattr(model, "n_features_in_", -1)),
        "feature_names_in_": [str(x) for x in getattr(model, "feature_names_in_", [])],
        "classes_": [int(c) for c in getattr(model, "classes_", [])],
        "params": model.get_params(deep=False) if hasattr(model, "get_params") else {},
    }
    if info["feature_names_in_"] and list(info["feature_names_in_"]) != list(FEATURE_COLUMNS):
        raise ModelLoadError(
            "Original model feature names do not match the UCI schema expected by CREDIT LAB."
        )
    if not info["has_predict"]:
        raise ModelLoadError("Original model does not implement predict().")
    return info


def load_original_model():
    inspect_original_model()
    return joblib.load(ORIGINAL_MODEL_PATH)


def original_matches_split(model, x_train, y_train) -> bool:
    """True if the artifact equals a max_depth=3 tree fitted on this training split."""
    if not isinstance(model, DecisionTreeClassifier):
        return False
    clone = DecisionTreeClassifier(max_depth=3, random_state=42)
    clone.fit(x_train, y_train)
    return bool(
        np.array_equal(clone.tree_.feature, model.tree_.feature)
        and np.allclose(clone.tree_.threshold, model.tree_.threshold)
    )


def train_comparison_models(df=None, persist: bool = True) -> dict[str, Any]:
    """Train KNN and Random Forest; reuse the original tree as the Decision Tree."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = df if df is not None else load_credit_dataset()
    quality = validate_dataset(df)
    x, y = split_xy(df)
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    original = load_original_model()
    comparable = original_matches_split(original, x_train, y_train)

    knn = build_knn_pipeline()
    knn.fit(x_train, y_train)
    forest = build_random_forest()
    forest.fit(x_train, y_train)

    estimators = {
        MODEL_KNN: knn,
        MODEL_TREE: original,
        MODEL_FOREST: forest,
    }
    results = {}
    for key, estimator in estimators.items():
        payload = evaluate_estimator(estimator, x_test, y_test)
        payload["comparable_test_set"] = True if key != MODEL_TREE else comparable
        results[key] = payload

    best_key, best_value = select_best_model(results, "f1")
    bundle = {
        "n_rows": quality["n_rows"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "positive_class": 1,
        "primary_metric": "f1",
        "best_model": best_key,
        "best_metric_value": best_value,
        "original_model": {
            "path": ORIGINAL_MODEL_PATH.name,
            "type": type(original).__name__,
            "params": {
                "max_depth": original.get_params().get("max_depth"),
                "random_state": original.get_params().get("random_state"),
            },
            "matches_current_training_split": comparable,
            "used_as": MODEL_TREE,
        },
        "knn_params": {"n_neighbors": 15, "scaling": "StandardScaler fitted on train only"},
        "forest_params": {"n_estimators": 100, "random_state": RANDOM_STATE},
        "results": results,
        "class_counts": quality["class_counts"],
    }

    if persist:
        joblib.dump(knn, KNN_JOBLIB)
        joblib.dump(forest, FOREST_JOBLIB)
        EVALUATION_JSON.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return {"bundle": bundle, "estimators": estimators, "split": (x_train, x_test, y_train, y_test)}


def _artifacts_ready() -> bool:
    return KNN_JOBLIB.exists() and FOREST_JOBLIB.exists() and EVALUATION_JSON.exists()


def load_lab(df=None) -> dict[str, Any]:
    """Return estimators plus evaluation bundle, training only if artifacts are missing."""
    if not _artifacts_ready():
        trained = train_comparison_models(df=df, persist=True)
        return trained

    knn = joblib.load(KNN_JOBLIB)
    forest = joblib.load(FOREST_JOBLIB)
    original = load_original_model()
    bundle = json.loads(EVALUATION_JSON.read_text(encoding="utf-8"))
    return {
        "bundle": bundle,
        "estimators": {
            MODEL_KNN: knn,
            MODEL_TREE: original,
            MODEL_FOREST: forest,
        },
        "split": None,
    }
