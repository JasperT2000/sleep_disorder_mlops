import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Sleep Disorder Analysis", layout="wide")
st.title("Sleep Disorder Analysis Demo")

st.markdown(
    """
Upload either:

- a **prediction CSV** with columns: `SubNo, SegNo, y_pred`
- or a **4-class labeled dataset CSV** with columns: `SubNo, SegNo, Class`

The app will compute sleep metrics and estimate overall disorder risk.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=57)
with col3:
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=29.4)

default_api_url = os.getenv("API_URL", "http://localhost:8000/analyze")
api_url = st.text_input("API URL", value=default_api_url)

STAGE_NAME_MAP = {0: "Awake", 1: "Light", 2: "Deep", 3: "REM"}
STAGE_Y_MAP = {0: 3, 1: 2, 2: 1, 3: 0}


def metric_card(label, value, suffix=""):
    if value is None or pd.isna(value):
        display = "N/A"
    elif isinstance(value, float):
        display = f"{value:.2f}{suffix}"
    else:
        display = f"{value}{suffix}"
    st.metric(label=label, value=display)


def render_risk_banner(prob, pred, risk_level, threshold):
    if risk_level == "Very High":
        st.error(f"Very high disorder risk — probability: {prob:.3f}")
    elif risk_level == "High":
        st.error(f"High disorder risk — probability: {prob:.3f}")
    elif risk_level == "Moderate":
        st.warning(f"Moderate disorder risk — probability: {prob:.3f}")
    else:
        st.success(f"Low disorder risk — probability: {prob:.3f}")

    st.caption(
        f"Predicted class: {'Possible disorder' if pred == 1 else 'Low / no disorder'} | "
        f"Threshold used: {threshold:.2f}"
    )


def plot_sleep_stage_timeline(df_sub):
    stage_col = "y_pred" if "y_pred" in df_sub.columns else "Class"
    df_plot = df_sub.sort_values("SegNo").copy()
    df_plot["time_min"] = (df_plot["SegNo"] - df_plot["SegNo"].min()) * 0.5
    df_plot["stage_y"] = df_plot[stage_col].map(STAGE_Y_MAP)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.step(df_plot["time_min"], df_plot["stage_y"], where="post")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["Awake", "Light", "Deep", "REM"])
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Sleep Stage Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_stage_distribution(df_sub):
    stage_col = "y_pred" if "y_pred" in df_sub.columns else "Class"
    counts = df_sub[stage_col].value_counts().sort_index()
    labels = [STAGE_NAME_MAP.get(i, str(i)) for i in counts.index]
    values = counts.values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Stage Distribution")
    ax.set_ylabel("Epoch Count")
    plt.tight_layout()
    return fig


def validate_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    required_base = {"SubNo", "SegNo"}
    if not required_base.issubset(df.columns):
        missing = required_base - set(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "y_pred" not in df.columns and "Class" not in df.columns:
        raise ValueError("CSV must contain either 'y_pred' or 'Class' column.")

    if "y_pred" not in df.columns and "Class" in df.columns:
        df = df.copy()
        df["y_pred"] = df["Class"]

    return df


if uploaded_file and st.button("Run analysis"):
    try:
        file_bytes = uploaded_file.getvalue()
        uploaded_df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))
        uploaded_df = validate_uploaded_df(uploaded_df)
    except Exception as e:
        st.error(f"Invalid CSV: {e}")
        st.stop()

    files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
    data = {"gender": gender, "age": age, "bmi": bmi}

    try:
        with st.spinner("Running analysis..."):
            response = requests.post(api_url, files=files, data=data, timeout=120)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API at {api_url}")
        st.exception(e)
        st.stop()

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        st.stop()

    try:
        payload = response.json()
        results = payload["results"]
    except Exception as e:
        st.error("API returned an invalid response.")
        st.exception(e)
        st.stop()

    subject_ids = sorted(uploaded_df["SubNo"].unique().tolist())
    selected_subject = st.selectbox("Select Subject", subject_ids) if len(subject_ids) > 1 else subject_ids[0]

    result_map = {int(r["SubNo"]): r for r in results}
    subject_result = result_map.get(int(selected_subject))

    if subject_result is None:
        st.error("No result returned for selected subject.")
        st.stop()

    subject_df = uploaded_df[uploaded_df["SubNo"] == selected_subject].copy().sort_values("SegNo")

    st.subheader(f"Subject {selected_subject} Summary")

    render_risk_banner(
        prob=subject_result["disorder_risk_probability"],
        pred=subject_result["disorder_risk_prediction"],
        risk_level=subject_result["risk_level"],
        threshold=subject_result["threshold_used"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Risk Probability", subject_result["disorder_risk_probability"])
    with c2:
        metric_card("Risk Level", subject_result["risk_level"])
    with c3:
        metric_card("Sleep Efficiency", subject_result["sleep_efficiency_pct"], "%")
    with c4:
        metric_card("Total Sleep Time", subject_result["total_sleep_time_min"], " min")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Sleep Latency", subject_result["sleep_latency_min"], " min")
    with c6:
        metric_card("REM Latency", subject_result["rem_latency_min"], " min")
    with c7:
        metric_card("Arousal Index", subject_result["arousal_index"], "/hr")
    with c8:
        metric_card("Awakenings", int(subject_result["number_of_awakenings"]))

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        metric_card("WASO", subject_result["waso_min"], " min")
    with c10:
        metric_card("Transitions/hr", subject_result["stage_transitions_per_hour"])
    with c11:
        metric_card("Light Sleep", subject_result["light_sleep_pct"], "%")
    with c12:
        metric_card("Deep Sleep", subject_result["deep_sleep_pct"], "%")

    c13, c14, c15 = st.columns(3)
    with c13:
        metric_card("REM Sleep", subject_result["rem_sleep_pct"], "%")
    with c14:
        metric_card("Longest Sleep Run", subject_result["longest_sleep_run_min"], " min")
    with c15:
        metric_card("REM Bouts", int(subject_result["rem_bouts"]))

    st.markdown("---")

    left, right = st.columns([2, 1])
    with left:
        st.pyplot(plot_sleep_stage_timeline(subject_df))
    with right:
        st.pyplot(plot_stage_distribution(subject_df))

    st.markdown("---")
    st.subheader("Detailed Metrics")
    st.dataframe(pd.DataFrame([subject_result]), use_container_width=True)