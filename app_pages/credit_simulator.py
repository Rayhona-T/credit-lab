from collections.abc import Callable
from typing import Any

import streamlit as st

from components.lab_data import lab_or_stop
from components.styles import lab_footer
from src.constants import (
    BILL_COLUMNS,
    BILL_MONTHS,
    EDUCATION_LABELS,
    FEATURE_COLUMNS,
    MARRIAGE_LABELS,
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    MODEL_SHORT_NAMES,
    PAY_AMT_COLUMNS,
    PAY_AMT_MONTHS,
    PAY_STATUS_COLUMNS,
    PAY_STATUS_LABELS,
    PAY_STATUS_MONTHS,
    SEX_LABELS,
)
from src.prediction import (
    ALLOWED_RANGES,
    InvalidClientInput,
    compare_predictions,
    predict_client,
)
from src.profiles import PRESETS

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Sex code",
    "EDUCATION": "Education code",
    "MARRIAGE": "Marriage code",
    "AGE": "Age",
    **{column: f"Repayment status · {month}" for column, month in PAY_STATUS_MONTHS.items()},
    **{column: f"Bill amount · {month}" for column, month in BILL_MONTHS.items()},
    **{column: f"Payment amount · {month}" for column, month in PAY_AMT_MONTHS.items()},
}


def _widget_key(feature: str) -> str:
    return f"simulator_{feature}"


def _reset_comparison() -> None:
    st.session_state.baseline_prediction = None
    st.session_state.baseline_client = None
    st.session_state.modified_prediction = None
    st.session_state.modified_client = None


def _load_preset() -> None:
    record = PRESETS[st.session_state.client_preset]
    st.session_state.client = dict(record)
    for feature, value in record.items():
        st.session_state[_widget_key(feature)] = value
    _reset_comparison()


def _code_label(labels: dict[int, str]) -> Callable[[int], str]:
    return lambda code: f"{code} · {labels.get(code, 'Undocumented code')}"


def _format_prediction(prediction: dict[str, Any]) -> None:
    st.markdown(f"**Predicted class:** {prediction['predicted_label']}")
    probability = prediction["default_probability"]
    if probability is not None:
        st.metric("Estimated default probability", f"{probability:.1%}")
    else:
        st.caption("This model does not provide a probability estimate.")


def _get_form_record() -> dict[str, Any]:
    return {
        feature: st.session_state[_widget_key(feature)]
        for feature in FEATURE_COLUMNS
    }


lab = lab_or_stop()
estimators = lab["estimators"]

for feature, value in st.session_state.client.items():
    st.session_state.setdefault(_widget_key(feature), value)
st.session_state.setdefault("modified_client", None)

st.html('<div class="lab-kicker">Section 04 · Inference</div>')
st.header("Credit Simulator")
st.markdown(
    "Enter a hypothetical profile and send it through a trained classifier. "
    "The result estimates a historical label; it is not an official credit score, "
    "a lending decision, or a guarantee about anyone's financial future."
)

st.selectbox(
    "Start from a hypothetical client preset",
    options=list(PRESETS),
    key="client_preset",
    on_change=_load_preset,
)
st.caption("Presets contain example inputs only. Every displayed output comes from model inference.")

model_key = st.segmented_control(
    "Prediction model",
    options=list(MODEL_KEYS),
    format_func=lambda key: MODEL_SHORT_NAMES[key],
    key="simulator_model",
    on_change=_reset_comparison,
    width="stretch",
)
st.caption(f"Selected model: {MODEL_DISPLAY_NAMES[model_key]}.")

with st.form("client_simulator_form", clear_on_submit=False):
    st.subheader("1 · Financial identity")
    identity = st.columns(3)
    with identity[0]:
        st.number_input(
            "Credit limit (NT dollar)",
            min_value=ALLOWED_RANGES["LIMIT_BAL"][0],
            max_value=ALLOWED_RANGES["LIMIT_BAL"][1],
            step=10_000,
            key=_widget_key("LIMIT_BAL"),
        )
        st.selectbox(
            "Sex code",
            options=list(SEX_LABELS),
            format_func=_code_label(SEX_LABELS),
            key=_widget_key("SEX"),
        )
    with identity[1]:
        st.selectbox(
            "Education code",
            options=list(EDUCATION_LABELS),
            format_func=_code_label(EDUCATION_LABELS),
            key=_widget_key("EDUCATION"),
        )
        st.selectbox(
            "Marriage code",
            options=list(MARRIAGE_LABELS),
            format_func=_code_label(MARRIAGE_LABELS),
            key=_widget_key("MARRIAGE"),
        )
    with identity[2]:
        st.number_input(
            "Age",
            min_value=ALLOWED_RANGES["AGE"][0],
            max_value=ALLOWED_RANGES["AGE"][1],
            step=1,
            key=_widget_key("AGE"),
        )

    st.subheader("2 · Repayment behavior")
    st.caption("Dataset repayment-status codes are labels, not ordered dollar amounts.")
    repayment_columns = st.columns(3)
    for index, feature in enumerate(PAY_STATUS_COLUMNS):
        with repayment_columns[index % 3]:
            st.selectbox(
                PAY_STATUS_MONTHS[feature],
                options=list(PAY_STATUS_LABELS),
                format_func=_code_label(PAY_STATUS_LABELS),
                key=_widget_key(feature),
            )

    st.subheader("3 · Financial activity")
    bill_columns = st.columns(3)
    for index, feature in enumerate(BILL_COLUMNS):
        with bill_columns[index % 3]:
            low, high = ALLOWED_RANGES[feature]
            st.number_input(
                f"Bill · {BILL_MONTHS[feature]} (NT dollar)",
                min_value=low,
                max_value=high,
                step=1_000,
                key=_widget_key(feature),
            )

    payment_columns = st.columns(3)
    for index, feature in enumerate(PAY_AMT_COLUMNS):
        with payment_columns[index % 3]:
            low, high = ALLOWED_RANGES[feature]
            st.number_input(
                f"Payment · {PAY_AMT_MONTHS[feature]} (NT dollar)",
                min_value=low,
                max_value=high,
                step=1_000,
                key=_widget_key(feature),
            )

    st.subheader("4 · Analyze this client")
    st.caption(
        "Analyze saves the current profile as the comparison baseline. "
        "Edit any inputs, then run a what-if comparison against that baseline."
    )
    analyze_clicked = st.form_submit_button(
        "Analyze client and set baseline",
        type="primary",
    )
    compare_clicked = st.form_submit_button("Compare modified client")

if analyze_clicked or compare_clicked:
    if compare_clicked and st.session_state.baseline_prediction is None:
        st.warning("Analyze a client first to create a comparison baseline.")
    else:
        try:
            current_client = _get_form_record()
            current_prediction = predict_client(
                estimators[model_key],
                current_client,
                model_key=model_key,
            )
        except InvalidClientInput as exc:
            st.error(f"Check the client inputs: {exc}", icon=":material/error:")
        else:
            st.session_state.client = current_prediction["features"]
            if analyze_clicked:
                st.session_state.baseline_client = current_prediction["features"]
                st.session_state.baseline_prediction = current_prediction
                st.session_state.modified_client = None
                st.session_state.modified_prediction = None
            else:
                st.session_state.modified_client = current_prediction["features"]
                st.session_state.modified_prediction = current_prediction

baseline = st.session_state.baseline_prediction
modified = st.session_state.modified_prediction
if baseline is not None:
    st.space("medium")
    st.subheader("Baseline prediction")
    with st.container(border=True):
        st.caption(f"Model: {MODEL_DISPLAY_NAMES[baseline['model_key']]}")
        _format_prediction(baseline)
    st.caption(
        "A model probability, when available, is an estimate on this model's output scale; "
        "it is not a calibrated personal credit score."
    )

if modified is not None and baseline is not None:
    comparison = compare_predictions(baseline, modified)
    st.subheader("What-if comparison")
    original_column, modified_column = st.columns(2)
    with original_column:
        with st.container(border=True):
            st.markdown("**Baseline**")
            _format_prediction(baseline)
    with modified_column:
        with st.container(border=True):
            st.markdown("**Modified client**")
            _format_prediction(modified)

    if comparison["probability_delta"] is not None:
        delta = comparison["probability_delta"]
        st.metric(
            "Absolute probability change",
            f"{abs(delta) * 100:.1f} percentage points",
            delta="Modified estimate higher" if delta > 0 else (
                "Modified estimate lower" if delta < 0 else "No change"
            ),
            delta_color="off",
        )
    st.markdown("**Inputs changed**")
    if comparison["changed_features"]:
        st.write(
            ", ".join(
                f"{feature} ({FEATURE_LABELS[feature]})"
                for feature in comparison["changed_features"]
            )
        )
    else:
        st.write("No inputs changed from the baseline.")
    st.caption(
        "This comparison shows how this fitted model responds to changed inputs. "
        "It does not establish that changing a feature causes a real-world outcome."
    )

st.space("medium")
st.info(
    "This classroom demonstration uses historical data with known limitations, including "
    "class imbalance and demographic inputs. Do not use it to make real credit decisions.",
    title="Interpret the estimate with care",
)
lab_footer()
