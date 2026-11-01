import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.lab_data import dataset_or_stop
from components.styles import lab_footer
from src.charts import CYAN, GREEN, ROSE, lab_layout
from src.constants import (
    BILL_COLUMNS,
    BILL_MONTHS,
    NEGATIVE_LABEL,
    PAY_AMT_COLUMNS,
    PAY_AMT_MONTHS,
    PAY_STATUS_COLUMNS,
    PAY_STATUS_LABELS,
    PAY_STATUS_MONTHS,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)

df, quality = dataset_or_stop()
labeled = df.copy()
labeled["outcome"] = np.where(labeled[TARGET_COLUMN] == 1, POSITIVE_LABEL, NEGATIVE_LABEL)

st.html('<div class="lab-kicker">Section 03 · Data</div>')
st.header("Data Observatory")
st.markdown(
    "These charts describe **associations in the historical sample**. "
    "A different distribution between groups is not proof that a feature causes default, "
    "and it is not a window into how a model will reason about a new person."
)

st.subheader("A · The default landscape")
counts = quality["class_counts"]
n0, n1 = counts[0], counts[1]
n = n0 + n1
fig = go.Figure(
    go.Bar(
        x=[NEGATIVE_LABEL, POSITIVE_LABEL],
        y=[n0, n1],
        marker_color=[CYAN, ROSE],
        text=[f"{n0:,} ({n0/n:.1%})", f"{n1:,} ({n1/n:.1%})"],
        textposition="outside",
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    )
)
fig.update_yaxes(title="Customers")
lab_layout(fig, height=360)
st.plotly_chart(fig, width="stretch", alt="Bar chart of default versus no-default counts")
st.caption(
    f"No default: {n0:,} ({n0/n:.1%}). Default: {n1:,} ({n1/n:.1%}). "
    "A classifier that always predicts no default would already be about 77.9% accurate. "
    "That is why precision, recall, F1, and ROC-AUC sit next to accuracy in the Arena."
)

st.subheader("B · The repayment timeline")
st.markdown(
    "UCI repayment-status codes used here: "
    + ", ".join(f"`{k}` = {v}" for k, v in PAY_STATUS_LABELS.items())
    + ". Codes outside this list are shown as numeric values rather than guessed categories."
)
mode = st.segmented_control(
    "Timeline view",
    options=["mean_code", "share_delay"],
    format_func=lambda k: "Mean status code" if k == "mean_code" else "Share with delay ≥ 1 month",
    key="timeline_view",
    width="stretch",
)
rows = []
for col in PAY_STATUS_COLUMNS:
    month = PAY_STATUS_MONTHS[col]
    for outcome, label in [(0, NEGATIVE_LABEL), (1, POSITIVE_LABEL)]:
        series = labeled.loc[labeled[TARGET_COLUMN] == outcome, col]
        value = float(series.mean()) if mode == "mean_code" else float((series >= 1).mean())
        rows.append({"Month": month, "Outcome": label, "Value": value, "Feature": col})
timeline = pd.DataFrame(rows)
fig_t = px.bar(
    timeline,
    x="Month",
    y="Value",
    color="Outcome",
    barmode="group",
    color_discrete_map={NEGATIVE_LABEL: CYAN, POSITIVE_LABEL: ROSE},
)
fig_t.update_yaxes(title="Mean code" if mode == "mean_code" else "Share delayed")
lab_layout(fig_t, height=380)
st.plotly_chart(fig_t, width="stretch", alt="Grouped bars of repayment status by month and outcome")
st.caption(
    "Later months in this file are listed first because PAY_0 is September 2005. "
    "A higher mean code is associated with more delay in this sample; it does not identify a cause."
)

st.subheader("C · The credit limit explorer")
fig_lim = px.violin(
    labeled,
    x="outcome",
    y="LIMIT_BAL",
    color="outcome",
    box=True,
    points=False,
    color_discrete_map={NEGATIVE_LABEL: CYAN, POSITIVE_LABEL: ROSE},
)
fig_lim.update_layout(showlegend=False)
fig_lim.update_xaxes(title="")
fig_lim.update_yaxes(title="Credit limit (NT dollar)")
lab_layout(fig_lim, height=400)
st.plotly_chart(fig_lim, width="stretch", alt="Violin plot of credit limits by observed outcome")
stats = labeled.groupby("outcome")["LIMIT_BAL"].describe()[["count", "mean", "50%", "std"]]
st.dataframe(stats, width="stretch", alt="Credit limit summary statistics by outcome")
st.caption(
    "Limits are right-skewed. Group differences here are descriptive. "
    "Credit limit is assigned by a lender, so it already embeds prior decisions."
)

st.subheader("D · The payment reality check")
pair = st.selectbox(
    "Month to compare",
    options=list(range(6)),
    format_func=lambda i: BILL_MONTHS[BILL_COLUMNS[i]],
    key="bill_month",
)
bill_col = BILL_COLUMNS[pair]
pay_col = PAY_AMT_COLUMNS[pair]
c1, c2 = st.columns(2)
with c1:
    fig_b = px.box(
        labeled,
        x="outcome",
        y=bill_col,
        color="outcome",
        color_discrete_map={NEGATIVE_LABEL: CYAN, POSITIVE_LABEL: ROSE},
        points=False,
    )
    fig_b.update_layout(showlegend=False, title=f"Bill amount · {BILL_MONTHS[bill_col]}")
    fig_b.update_yaxes(title="NT dollar")
    lab_layout(fig_b, height=380)
    st.plotly_chart(fig_b, width="stretch", alt=f"Box plot of {bill_col} by outcome")
with c2:
    fig_p = px.box(
        labeled,
        x="outcome",
        y=pay_col,
        color="outcome",
        color_discrete_map={NEGATIVE_LABEL: CYAN, POSITIVE_LABEL: ROSE},
        points=False,
    )
    fig_p.update_layout(showlegend=False, title=f"Payment amount · {PAY_AMT_MONTHS[pay_col]}")
    fig_p.update_yaxes(title="NT dollar")
    lab_layout(fig_p, height=380)
    st.plotly_chart(fig_p, width="stretch", alt=f"Box plot of {pay_col} by outcome")
st.caption(
    "Bills and payments have heavy tails and occasional negatives (credits). "
    "Boxes hide extreme outliers so the typical range stays readable."
)

lab_footer()
