"""Limited, model-appropriate explanations.

Global tree importances describe the fitted forest or tree as a whole. They are
not explanations of a single simulated client. KNN has no native feature
importance; we report that limitation instead of inventing one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constants import FEATURE_COLUMNS, MODEL_FOREST, MODEL_KNN, MODEL_TREE


def global_feature_importance(estimator, model_key: str, top_n: int = 8) -> dict[str, Any]:
    if model_key == MODEL_KNN:
        return {
            "method": None,
            "available": False,
            "note": (
                "KNN does not produce feature importances. A neighbor vote depends "
                "on distance in the scaled feature space, not on a ranked feature list."
            ),
            "items": [],
        }

    values = None
    names = list(FEATURE_COLUMNS)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if hasattr(estimator, "feature_names_in_"):
            names = [str(x) for x in estimator.feature_names_in_]
    elif hasattr(estimator, "named_steps"):
        final = estimator.named_steps[list(estimator.named_steps)[-1]]
        if hasattr(final, "feature_importances_"):
            values = np.asarray(final.feature_importances_, dtype=float)

    if values is None:
        return {
            "method": None,
            "available": False,
            "note": "This estimator does not expose feature_importances_.",
            "items": [],
        }

    order = np.argsort(values)[::-1][:top_n]
    items = [{"feature": names[i], "importance": float(values[i])} for i in order]
    method = "gini_decrease"
    if model_key == MODEL_TREE:
        note = (
            "These values are global Gini importances for the fitted decision tree. "
            "They describe splits across the training set, not one simulated client."
        )
    elif model_key == MODEL_FOREST:
        note = (
            "These values are mean Gini importances across the trees in the forest. "
            "They are a global ranking, not a local explanation of one prediction."
        )
    else:
        note = "Global impurity-based feature importance."
    return {"method": method, "available": True, "note": note, "items": items}
