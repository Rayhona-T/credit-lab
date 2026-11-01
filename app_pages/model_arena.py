import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.lab_data import lab_or_stop
from components.styles import lab_footer
from src.charts import GREEN, MODEL_COLORS, MUTED, PLOT, ROSE, lab_layout
from src.constants import (
    METRIC_DISPLAY,
    METRIC_KEYS,
    MODEL_DISPLAY_NAMES,
    MODEL_FOREST,
    MODEL_KNN,
    MODEL_SHORT_NAMES,
    MODEL_TREE,
    PRIMARY_METRIC,
)
from src.explanations import global_feature_importance

lab = lab_or_stop()
bundle = lab["bundle"]
results = bundle["results"]
estimators = lab["estimators"]

st.html('<div class="lab-kicker">Section 02 · Evaluation</div>')
st.header("Model Arena")
st.markdown(
    "All three models are scored on the **same stratified test set** "
    f"({bundle['n_test']:,} rows, seed {bundle['random_state']}). "
    "Precision, recall, and F1 treat **default (1)** as the positive class. "
    "Accuracy is shown because it is familiar, not because it is sufficient."
)

METRIC_NOTES = {
    "accuracy": "Share of test rows whose predicted class matches the observed label. Easy to inflate when most customers did not default.",
    "precision": "Among customers predicted to default, the share who actually defaulted. Low precision means extra false alarms for a lender.",
    "recall": "Among customers who actually defaulted, the share the model caught. Low recall means missed defaults.",
    "f1": "Harmonic mean of precision and recall for the default class. A compromise when both error types matter.",
    "roc_auc": "Ranking quality using predicted default probability. 0.5 is chance-level ranking; 1.0 is perfect ranking on this test set.",
}

PERSONALITIES = {
    MODEL_KNN: {
        "title": "The Neighborhood Thinker",
        "mechanism": "Looks up the closest training customers in scaled feature space and lets them vote.",
        "strength": "Simple geometry; no global parametric shape.",
        "limit": "Needs scaling, slows down on large data, and is hard to interpret feature-by-feature.",
    },
    MODEL_TREE: {
        "title": "The Question Asker",
        "mechanism": "Learns a short sequence of yes/no splits. This lab uses the supplied depth-3 tree.",
        "strength": "Readable rules and cheap predictions.",
        "limit": "A shallow tree can miss interactions; a deep tree can overfit.",
    },
    MODEL_FOREST: {
        "title": "The Committee",
        "mechanism": "Averages many trees trained on bootstrap samples and random feature subsets.",
        "strength": "Often ranks examples well and absorbs noisy features.",
        "limit": "The committee is harder to read than one tree.",
    },
}

metric = st.segmented_control(
    "Comparison metric",
    options=list(METRIC_KEYS),
    format_func=lambda k: METRIC_DISPLAY[k],
    key="selected_metric",
    width="stretch",
)
st.caption(METRIC_NOTES[metric])

fig = go.Figure(
    go.Bar(
        x=[MODEL_SHORT_NAMES[k] for k in results],
        y=[results[k]["metrics"][metric] for k in results],
        marker_color=[MODEL_COLORS[k] for k in results],
        text=[f"{results[k]['metrics'][metric]:.3f}" for k in results],
        textposition="outside",
        hovertemplate="%{x}<br>" + METRIC_DISPLAY[metric] + ": %{y:.3f}<extra></extra>",
    )
)
fig.update_yaxes(range=[0, 1], title=METRIC_DISPLAY[metric])
lab_layout(fig, height=360)
st.plotly_chart(
    fig,
    width="stretch",
    alt=f"Interactive metric comparison for {METRIC_DISPLAY[metric]}",
)

st.subheader("Algorithm personalities")
cols = st.columns(3)
for col, key in zip(cols, PERSONALITIES):
    persona = PERSONALITIES[key]
    with col:
        with st.container(border=True, height="stretch"):
            st.markdown(f"**{MODEL_SHORT_NAMES[key]}**")
            st.caption(persona["title"])
            st.markdown(persona["mechanism"])
            st.markdown(f"Strength: {persona['strength']}")
            st.markdown(f"Limit: {persona['limit']}")
            if st.button("View details", key=f"goto_{key}", width="stretch"):
                st.session_state.selected_model = key
                st.rerun()

st.subheader("Model dossier")
model_key = st.segmented_control(
    "Selected model",
    options=list(MODEL_SHORT_NAMES.keys()),
    format_func=lambda k: MODEL_SHORT_NAMES[k],
    key="selected_model",
    width="stretch",
)
payload = results[model_key]
metrics = payload["metrics"]
detail_cols = st.columns(5)
for col, key in zip(detail_cols, METRIC_KEYS):
    value = metrics[key]
    col.metric(METRIC_DISPLAY[key], f"{value:.3f}" if value is not None else "—", border=True)

cfg = {
    MODEL_KNN: bundle["knn_params"],
    MODEL_TREE: bundle["original_model"]["params"],
    MODEL_FOREST: bundle["forest_params"],
}[model_key]
st.markdown(
    f"**{MODEL_DISPLAY_NAMES[model_key]}** · {PERSONALITIES[model_key]['title']}. "
    f"Configuration: `{cfg}`."
)
if model_key == MODEL_TREE:
    st.caption(
        "This is the original `credit_model.joblib` file (DecisionTreeClassifier, sklearn 1.6.1 artifact). "
        "It is loaded as-is and not retrained."
    )

importance = global_feature_importance(estimators[model_key], model_key)
with st.expander("Global feature importance (not a single-client explanation)"):
    if not importance["available"]:
        st.markdown(importance["note"])
    else:
        st.markdown(importance["note"])
        names = [row["feature"] for row in importance["items"]][::-1]
        vals = [row["importance"] for row in importance["items"]][::-1]
        imp_fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=GREEN))
        lab_layout(imp_fig, height=320)
        st.plotly_chart(imp_fig, width="stretch", alt="Global feature importance bars")

st.subheader("Confusion matrix laboratory")
st.markdown(
    "Rows are **actual** labels. Columns are **predicted** labels. "
    "Click a cell, or choose it below, to read what that error means for a lender."
)
normalize = st.toggle("Show row percentages", value=False)
cm = np.array(payload["confusion"]["matrix"], dtype=float)
z = cm.copy()
text = [[f"{int(v)}" for v in row] for row in cm]
if normalize:
    row_sums = cm.sum(axis=1, keepdims=True)
    z = np.divide(cm, row_sums, where=row_sums != 0)
    text = [[f"{int(cm[i, j])}<br>{z[i, j]*100:.1f}%" for j in range(2)] for i in range(2)]

x_labels = ["Predicted no default", "Predicted default"]
y_labels = ["Actual no default", "Actual default"]
cell_ids = np.array([["true_negative", "false_positive"], ["false_negative", "true_positive"]])
heat = go.Figure(
    go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        customdata=cell_ids,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, PLOT], [0.5, "#163028"], [1, GREEN]],
        showscale=False,
        hovertemplate="%{y}<br>%{x}<br>count shown in cell<extra></extra>",
    )
)
heat.update_yaxes(autorange="reversed")
lab_layout(heat, height=420)
event = st.plotly_chart(
    heat,
    width="stretch",
    on_select="rerun",
    selection_mode="points",
    key=f"cm_{model_key}",
    alt="Confusion matrix heatmap with actual rows and predicted columns",
)
if event.selection and event.selection.get("points"):
    point = event.selection["points"][0]
    custom = point.get("customdata")
    if isinstance(custom, list) and custom:
        st.session_state.cm_cell = custom[0]
    elif isinstance(custom, str):
        st.session_state.cm_cell = custom

cell = st.segmented_control(
    "Clicked mistake",
    options=["true_negative", "false_positive", "false_negative", "true_positive"],
    format_func=lambda k: {
        "true_negative": "True negative",
        "false_positive": "False positive",
        "false_negative": "False negative",
        "true_positive": "True positive",
    }[k],
    key="cm_cell",
    width="stretch",
)
conf = payload["confusion"]
CELL_COPY = {
    "true_negative": (
        "ok",
        f"{conf['true_negative']:,} customers",
        "The model predicted no default and the observed outcome was no default. "
        "A lender using this threshold would have treated these accounts as continuing.",
    ),
    "false_positive": (
        "warn",
        f"{conf['false_positive']:,} customers",
        "The model predicted default, but the observed outcome was no default. "
        "For a lender, this is a false alarm: extra friction for people who did not default in the data. "
        "It is not evidence that those people were 'risky' in a moral sense.",
    ),
    "false_negative": (
        "risk",
        f"{conf['false_negative']:,} customers",
        "The model predicted no default, but the observed outcome was default. "
        "For a lender, this is a missed default in the historical sample. "
        "Those losses are why recall on the default class matters more than raw accuracy here.",
    ),
    "true_positive": (
        "ok",
        f"{conf['true_positive']:,} customers",
        "The model predicted default and the observed outcome was default. "
        "This is a correct catch on the test set, not a guarantee the model would catch the next default.",
    ),
}
tone, count_label, explanation = CELL_COPY[cell]
cls = {"ok": "lab-ok-bar", "warn": "lab-warn-bar", "risk": "lab-risk-bar"}[tone]
st.html(f'<div class="lab-cell-note {cls}"><strong>{count_label}</strong><br/>{explanation}</div>')

st.caption(
    f"Primary ranking metric in Command Center defaults to {METRIC_DISPLAY[PRIMARY_METRIC]} "
    "because the classes are imbalanced. Random Forest may still lead on ROC-AUC."
)

lab_footer()
