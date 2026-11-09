# app.py
# Student Performance Prediction - classroom prototype (Streamlit)

import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="centered")

# ---------------------------------------------------------------
# Look & feel: a quiet "field notebook" style - paper, ink, one signal colour
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');

    :root {
        --paper: #F3F0E8;
        --ink: #1D2B2A;
        --muted: #6B7776;
        --line: #D9D4C7;
        --risk: #D9482B;
        --safe: #1F7A6B;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: var(--ink); letter-spacing: -0.01em; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 880px; padding-top: 3rem; }

    .kicker { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
              letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
    .title { font-family: 'Fraunces', Georgia, serif; font-size: 2.8rem; line-height: 1.05; margin: 0.3rem 0 0.6rem; }
    .lede { color: var(--muted); max-width: 34rem; margin-bottom: 1.8rem; }

    .stat { border-top: 2px solid var(--ink); padding-top: 0.6rem; }
    .stat .n { font-family: 'Fraunces', Georgia, serif; font-size: 2.2rem; line-height: 1; }
    .stat .l { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
               letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-top: 0.35rem; }
    .stat.risk .n { color: var(--risk); }

    .section { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.14em;
               text-transform: uppercase; color: var(--muted); margin: 2.4rem 0 0.6rem;
               border-bottom: 1px solid var(--line); padding-bottom: 0.4rem; }

    .note { border-left: 3px solid var(--risk); padding: 0.1rem 0 0.1rem 0.9rem;
            color: var(--ink); font-size: 0.92rem; margin-top: 2rem; }

    .stButton > button, .stDownloadButton > button {
        background: var(--ink); color: var(--paper); border: 0; border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.05em;
        padding: 0.55rem 1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background: var(--risk); color: #fff; }
    [data-testid="stFileUploader"] section { background: transparent; border: 1.5px dashed var(--ink); border-radius: 2px; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--ink); }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * { background: var(--ink); color: var(--paper) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat(number, label, risk=False):
    cls = "stat risk" if risk else "stat"
    return f'<div class="{cls}"><div class="n">{number}</div><div class="l">{label}</div></div>'


# ---------------------------------------------------------------
# Load model and data
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


if not (os.path.exists("student_model.pkl") and os.path.exists("student_data.csv")):
    st.error("student_model.pkl or student_data.csv not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = pd.read_csv("student_data.csv")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown('<div class="kicker">Data Mining Laboratory · Prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Student Performance<br>Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lede">Upload student records and the model flags who may need extra '
    "academic support in Mathematics.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.markdown(stat(len(data), "Records"), unsafe_allow_html=True)
c2.markdown(stat(round(data["G3"].mean(), 2), "Average final grade"), unsafe_allow_html=True)
c3.markdown(stat(int((data["G3"] < 10).sum()), "Historical at-risk", risk=True), unsafe_allow_html=True)

# ---------------------------------------------------------------
# Model evaluation summary (saved by train_model.py)
# ---------------------------------------------------------------
if os.path.exists("model_metrics.json"):
    with open("model_metrics.json") as f:
        m = json.load(f)
    st.markdown('<div class="section">Model evaluation · 20% held-out test set</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(stat(f"{m['accuracy']:.2f}", "Accuracy"), unsafe_allow_html=True)
    e2.markdown(stat(f"{m['precision_at_risk']:.2f}", "Precision · at risk"), unsafe_allow_html=True)
    e3.markdown(stat(f"{m['recall_at_risk']:.2f}", "Recall · at risk"), unsafe_allow_html=True)
    e4.markdown(stat(f"{m['f1_at_risk']:.2f}", "F1 · at risk"), unsafe_allow_html=True)

# ---------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------
st.markdown('<div class="section">Dataset preview</div>', unsafe_allow_html=True)
st.dataframe(data.head(10), width="stretch")

# ---------------------------------------------------------------
# Predict uploaded records
# ---------------------------------------------------------------
st.markdown('<div class="section">Predict uploaded records</div>', unsafe_allow_html=True)
st.write("Upload a CSV with the same predictor columns as the training data (G3 is not needed).")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV. Please check the format.")
        st.stop()

    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]

        n_risk = int((predictions == 1).sum())
        st.markdown('<div class="section">Prediction results</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        r1.markdown(stat(n_risk, "Predicted at risk", risk=True), unsafe_allow_html=True)
        r2.markdown(stat(len(results) - n_risk, "Predicted not at risk"), unsafe_allow_html=True)

        def colour(val):
            if val == "AT RISK":
                return "color: #D9482B; font-weight: 600"
            if val == "NOT AT RISK":
                return "color: #1F7A6B; font-weight: 600"
            return ""

        st.write("")
        # Show the Prediction column first so it is visible without scrolling sideways
        results = results[["Prediction"] + [c for c in results.columns if c != "Prediction"]]
        st.dataframe(results.style.map(colour, subset=["Prediction"]), width="stretch")

        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )

st.markdown(
    '<div class="note"><b>Responsible use.</b> This is a classroom demonstration on public, '
    "historical data. A prediction is not a diagnosis and must never be the sole basis for "
    "a real decision about a student.</div>",
    unsafe_allow_html=True,
)
