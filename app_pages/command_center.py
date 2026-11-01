import plotly.graph_objects as go
import streamlit as st

from components.lab_data import dataset_or_stop, lab_or_stop
from components.styles import lab_footer
from src.charts import MODEL_COLORS, lab_layout
from src.constants import (
    METRIC_DISPLAY,
    METRIC_KEYS,
    MODEL_DISPLAY_NAMES,
    MODEL_SHORT_NAMES,
    PRIMARY_METRIC,
    UCI_DATASET_URL,
)

df, quality = dataset_or_stop()
lab = lab_or_stop()
bundle = lab["bundle"]
results = bundle["results"]

st.html('<div class="lab-kicker">Experiment 350 · UCI credit default</div>')
st.html(
    """
    <div class="lab-hero">
      <h1 class="lab-hero-title">CREDIT <span class="accent">LAB</span></h1>
    </div>
    """
)
st.markdown("### Inside the mind of a credit risk model.")
st.markdown(
    "This laboratory explores how three classical classifiers estimate the chance of "
    "**default payment next month** from historical Taiwan credit-card records. "
    "You can inspect the data, compare real test-set metrics, and run hypothetical clients "
    "through a trained model. Nothing here is a lending decision."
)

cta = st.container(horizontal=True, gap="small")
with cta:
    if st.button("Open Model Arena", icon=":material/swords:", type="primary"):
        st.switch_page("app_pages/model_arena.py")
    if st.button("Open Credit Simulator", icon=":material/tune:"):
        st.switch_page("app_pages/credit_simulator.py")

st.space("medium")

metric_key = st.session_state.get("selected_metric", PRIMARY_METRIC)
best_key, best_value = max(
    ((k, results[k]["metrics"][metric_key]) for k in results if results[k]["metrics"][metric_key] is not None),
    key=lambda item: item[1],
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Records analyzed", f"{quality['n_rows']:,}", border=True)
m2.metric("Models evaluated", "3", border=True)
m3.metric("Lead model", MODEL_SHORT_NAMES[best_key], border=True)
m4.metric(f"{METRIC_DISPLAY[metric_key]}", f"{best_value:.3f}", border=True)
st.caption(
    f"Lead model is the highest **{METRIC_DISPLAY[metric_key]}** on the held-out test set "
    f"(n = {bundle['n_test']:,}, stratified {int((1 - bundle['test_size']) * 100)}/"
    f"{int(bundle['test_size'] * 100)} split, seed {bundle['random_state']}). "
    "A different metric can change the ranking."
)

st.subheader("Model pulse")
pulse_metric = st.segmented_control(
    "Pulse metric",
    options=list(METRIC_KEYS),
    format_func=lambda k: METRIC_DISPLAY[k],
    key="selected_metric",
    width="stretch",
)
values = [results[k]["metrics"][pulse_metric] for k in results]
labels = [MODEL_SHORT_NAMES[k] for k in results]
colors = [MODEL_COLORS[k] for k in results]
fig = go.Figure(
    go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" if v is not None else "—" for v in values],
        textposition="outside",
        hovertemplate="%{x}<br>" + METRIC_DISPLAY[pulse_metric] + ": %{y:.3f}<extra></extra>",
    )
)
fig.update_yaxes(range=[0, 1], title=METRIC_DISPLAY[pulse_metric])
lab_layout(fig, height=340, title=None)
st.plotly_chart(
    fig,
    width="stretch",
    alt=f"Bar chart comparing {METRIC_DISPLAY[pulse_metric]} for KNN, Decision Tree, and Random Forest",
)

st.caption("Open the Arena to inspect confusion matrices and algorithm details.")

st.space("medium")
st.subheader("The experiment")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Research question**")
    st.markdown(
        "Given a customer's limit, demographics, six-month repayment status, bills, and payments, "
        "can a classifier usefully rank who defaulted in the following month in this historical sample?"
    )
    st.markdown("**Task**")
    st.markdown(
        "Binary classification. The positive class is **default = 1**. "
        "About 22.1% of the 30,000 records are defaults, so accuracy alone is a weak scoreboard."
    )
    st.markdown("**Dataset**")
    st.markdown(
        f"[UCI Default of Credit Card Clients]({UCI_DATASET_URL}) (Yeh & Lien). "
        "Identifier `ID` is never used as a predictor."
    )
with c2:
    st.markdown("**Algorithms**")
    st.markdown(
        "- **KNN** — majority vote of nearby scaled examples.\n"
        "- **Decision Tree** — the supplied `credit_model.joblib` artifact (`max_depth=3`).\n"
        "- **Random Forest** — 100 trees trained on the same split."
    )
    st.markdown("**Limits**")
    st.markdown(
        "This is not a production credit-scoring system. Features are observational, "
        "the labels are historical, protected attributes are present, and a probability "
        "is a model estimate—not a person's reliability."
    )

if not bundle["original_model"]["matches_current_training_split"]:
    st.warning(
        "The supplied Decision Tree does not match a tree fitted on the current training split, "
        "so its test metrics may not be comparable.",
        icon=":material/warning:",
    )
else:
    st.info(
        "The supplied Decision Tree matches a `max_depth=3, random_state=42` fit on this training split, "
        "so its test metrics are comparable to KNN and Random Forest.",
        icon=":material/verified:",
    )

lab_footer()
