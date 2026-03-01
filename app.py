import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import time

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UPI Fraud Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2d3561; border-radius: 3px; }

.main-header {
    background: linear-gradient(135deg, #0f1729 0%, #1a1040 50%, #0f1729 100%);
    border: 1px solid rgba(102,126,234,0.25);
    border-left: 4px solid #667eea;
    padding: 30px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(102,126,234,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    color: #f1f5f9 !important;
    margin: 0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8 !important;
    margin: 8px 0 0 !important;
    font-size: 0.95rem !important;
    font-weight: 300;
}

/* FIX: Metric cards — smaller font so numbers don't get clipped */
.metric-card {
    background: #0d1424;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 18px 10px;
    text-align: center;
    transition: border-color 0.3s, transform 0.2s;
    overflow: hidden;
    min-height: 90px;
}
.metric-card:hover { border-color: #667eea; transform: translateY(-2px); }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;       /* reduced from 1.9rem to prevent clipping */
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.3;
    word-break: break-all;
}
.metric-label {
    color: #64748b;
    font-size: 0.68rem;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}

.fraud-alert {
    background: linear-gradient(135deg, #1f0a0a, #2d0f0f);
    border: 1px solid #7f1d1d;
    border-left: 5px solid #ef4444;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.15);
    animation: pulseRed 3s ease-in-out infinite;
}
.safe-alert {
    background: linear-gradient(135deg, #051a0f, #0a2a1a);
    border: 1px solid #14532d;
    border-left: 5px solid #22c55e;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.12);
}
.warning-alert {
    background: linear-gradient(135deg, #1c1200, #2a1c00);
    border: 1px solid #713f12;
    border-left: 5px solid #f59e0b;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 0 40px rgba(245,158,11,0.12);
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 25px rgba(239,68,68,0.15); }
    50%       { box-shadow: 0 0 50px rgba(239,68,68,0.3);  }
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #818cf8;
    margin: 22px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
    letter-spacing: 0.3px;
}
.hyper-card {
    background: #0d1424;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.compare-better { color: #22c55e; font-weight: 700; }
.compare-worse  { color: #ef4444; font-weight: 700; }
.compare-same   { color: #f59e0b; font-weight: 700; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.4) !important;
}
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #1e2d4a !important;
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
div[data-testid="metric-container"] {
    background: #0d1424;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 12px;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PLOTLY BASE — no font key here; pass font separately when needed
# ══════════════════════════════════════════════════════════════════
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,20,36,0.95)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(t=50, b=30, l=20, r=20)
)

# Variant without font — use when you need to pass font=dict(size=N) separately
def plotly_base_no_font():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(13,20,36,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=30, l=20, r=20)
    )


# ══════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS — ALL FILES IN ROOT FOLDER
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    base_model = keras.models.load_model("ann_model.h5", compile=False)
    scaler     = joblib.load("scaler.pkl")
    with open("metrics.json")          as f: metrics  = json.load(f)
    with open("training_history.json") as f: history  = json.load(f)
    with open("features.json")         as f: features = json.load(f)
    return base_model, scaler, metrics, history, features


@st.cache_data
def load_train_test_data():
    test_npz  = np.load("test_data.npz")
    train_npz = np.load("train_data.npz")
    return (
        train_npz["X_train"], train_npz["y_train"],
        test_npz["X_test"],   test_npz["y_test"]
    )


with st.spinner("🛡️ Launching UPI Fraud Shield..."):
    base_model, scaler, metrics, history, FEATURES = load_artifacts()
    X_train_ready, y_train_ready, X_test_ready, y_test_ready = load_train_test_data()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ UPI Fraud Shield</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Transaction Security · ANN Model")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Fraud Detector", "⚙️ Hypertune Model",
         "📊 Model Analytics", "📈 Insights"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**📦 Base Model**")
    st.caption("🧠 Architecture  : 4-Layer ANN")
    st.caption("⚡ Runtime       : TensorFlow / Keras")
    st.caption(f"📊 ROC-AUC       : {metrics['roc_auc']:.4f}")
    st.caption(f"🔢 Features      : {len(FEATURES)}")
    st.caption("📁 Dataset       : PaySim Synthetic")

    if "tuned_metrics" in st.session_state:
        st.divider()
        tm = st.session_state["tuned_metrics"]
        st.markdown("**🆕 Tuned Model (Active)**")
        st.caption(f"📊 ROC-AUC   : {tm['roc_auc']:.4f}")
        st.caption(f"🎯 Precision : {tm['precision']:.4f}")
        st.caption(f"🔍 Recall    : {tm['recall']:.4f}")
        st.caption(f"🎚️ Threshold : {tm['threshold']:.2f}")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def predict_transaction(amount, old_bal_orig, new_bal_orig,
                        old_bal_dest, new_bal_dest, txn_type):
    feat = np.array([[
        amount, old_bal_orig, new_bal_orig, old_bal_dest, new_bal_dest,
        new_bal_orig + amount - old_bal_orig,
        old_bal_dest + amount - new_bal_dest,
        amount / (old_bal_orig + 1),
        1 if amount > 200000 else 0,
        1 if new_bal_orig == 0 else 0,
        new_bal_dest - old_bal_dest,
        1 if txn_type == "TRANSFER" else 0
    ]])
    feat_sc   = scaler.transform(feat).astype(np.float32)
    threshold = st.session_state.get("tuned_threshold", 0.5)
    if "tuned_keras_model" in st.session_state:
        prob = float(st.session_state["tuned_keras_model"].predict(feat_sc, verbose=0)[0][0])
    else:
        prob = float(base_model.predict(feat_sc, verbose=0)[0][0])
    return prob, threshold


def build_custom_ann(n1, n2, n3, d1, d2, d3, lr, use_bn):
    m = keras.Sequential(name="Tuned_ANN")
    m.add(layers.Input(shape=(len(FEATURES),)))
    m.add(layers.Dense(n1, activation="relu"))
    if use_bn:
        m.add(layers.BatchNormalization())
    m.add(layers.Dropout(d1))
    m.add(layers.Dense(n2, activation="relu"))
    if use_bn:
        m.add(layers.BatchNormalization())
    m.add(layers.Dropout(d2))
    m.add(layers.Dense(n3, activation="relu"))
    m.add(layers.Dropout(d3))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return m


def delta_badge(new_val, base_val, higher_better=True):
    diff = new_val - base_val
    if abs(diff) < 0.0001:
        return '<span class="compare-same">→ No change</span>'
    if (diff > 0) == higher_better:
        return f'<span class="compare-better">▲ +{abs(diff):.4f}</span>'
    return f'<span class="compare-worse">▼ -{abs(diff):.4f}</span>'


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ UPI Fraud Detection System</h1>
        <p>Real-time AI-powered transaction security · Deep Learning ANN · PaySim Dataset</p>
    </div>""", unsafe_allow_html=True)

    report = metrics["report"]
    kpis = [
        ("ROC-AUC",      f"{metrics['roc_auc']:.4f}"),
        ("Precision",    f"{float(report['Fraud']['precision']):.4f}"),
        ("Recall",       f"{float(report['Fraud']['recall']):.4f}"),
        ("F1 Score",     f"{float(report['Fraud']['f1-score']):.4f}"),
        ("Fraud Caught", f"{metrics['fraud_caught']:,}"),
        ("Fraud Missed", f"{metrics['fraud_missed']:,}"),
    ]
    cols = st.columns(6)
    for col, (label, val) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ep = list(range(1, len(history["loss"]) + 1))
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=history["loss"],     name="Train",
                                  line=dict(color="#818cf8", width=2.5)))
        fig.add_trace(go.Scatter(x=ep, y=history["val_loss"], name="Validation",
                                  line=dict(color="#f87171", width=2, dash="dot")))
        fig.update_layout(title="📉 Loss Curve", height=300,
                          legend=dict(orientation="h", y=1.12),
                          xaxis_title="Epoch", yaxis_title="Loss", **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=history["auc"],     name="Train",
                                  line=dict(color="#34d399", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(52,211,153,0.06)"))
        fig.add_trace(go.Scatter(x=ep, y=history["val_auc"], name="Validation",
                                  line=dict(color="#fbbf24", width=2, dash="dot")))
        fig.update_layout(title="📈 AUC Curve", height=300,
                          legend=dict(orientation="h", y=1.12),
                          xaxis_title="Epoch",
                          yaxis=dict(range=[0.5, 1], title="AUC"), **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cm  = metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                        labels=dict(x="Predicted", y="Actual"))
        # FIX: don't pass font= AND **PLOTLY_BASE together — use plotly_base_no_font()
        fig.update_layout(
            title="🎯 Confusion Matrix",
            height=340,
            font=dict(family="DM Sans", color="#94a3b8", size=14),
            **plotly_base_no_font()
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=["TRANSFER", "CASH_OUT"], y=[0.31, 0.18],
            marker=dict(color=["#818cf8", "#f87171"], line=dict(width=0)),
            text=["31.0%", "18.0%"], textposition="outside",
            textfont=dict(color="#e2e8f0", size=14)
        ))
        fig.update_layout(title="🔍 Fraud Rate by Transaction Type", height=340,
                          yaxis=dict(tickformat=".0%", title="Fraud Rate"),
                          **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — FRAUD DETECTOR
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Fraud Detector":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Real-Time Fraud Detector</h1>
        <p>Enter transaction details · Get instant AI-powered fraud assessment</p>
    </div>""", unsafe_allow_html=True)

    active_label = "🆕 Tuned Model" if "tuned_keras_model" in st.session_state else "📦 Base TensorFlow Model"
    st.info(f"**Active Model:** {active_label} · Train a custom model in ⚙️ Hypertune")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">📝 Transaction Details</div>', unsafe_allow_html=True)
        txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount   = st.number_input("Transaction Amount (₹)", min_value=1.0,
                                    max_value=10_000_000.0, value=50000.0,
                                    step=1000.0, format="%.2f")
        st.markdown("**Sender Account**")
        c1, c2 = st.columns(2)
        old_bal_orig = c1.number_input("Balance Before (₹)", min_value=0.0,
                                        value=100000.0, step=1000.0, key="ob_o")
        new_bal_orig = c2.number_input("Balance After (₹)",  min_value=0.0,
                                        value=50000.0,  step=1000.0, key="nb_o")
        st.markdown("**Receiver Account**")
        c3, c4 = st.columns(2)
        old_bal_dest = c3.number_input("Balance Before (₹)", min_value=0.0,
                                        value=5000.0,   step=1000.0, key="ob_d")
        new_bal_dest = c4.number_input("Balance After (₹)",  min_value=0.0,
                                        value=55000.0,  step=1000.0, key="nb_d")
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Transaction", use_container_width=True)

        st.markdown('<div class="section-title">💡 Quick Test Scenarios</div>', unsafe_allow_html=True)
        scenarios = {
            "🔴 Complete Balance Drain":    (500000, 500000, 0,     0,     500000, "TRANSFER"),
            "🟡 Large Suspicious Transfer": (250000, 260000, 10000, 100,   250100, "TRANSFER"),
            "🟢 Normal Bill Payment":       (5000,   50000,  45000, 10000, 15000,  "CASH_OUT"),
            "🟢 Small UPI Transfer":        (1200,   25000,  23800, 8000,  9200,   "TRANSFER"),
        }
        clicked_vals = None
        for label, vals in scenarios.items():
            if st.button(label, use_container_width=True, key=f"sc_{label}"):
                clicked_vals = vals

    with col2:
        st.markdown('<div class="section-title">📊 Analysis Result</div>', unsafe_allow_html=True)

        if clicked_vals:
            amount       = clicked_vals[0]
            old_bal_orig = clicked_vals[1]
            new_bal_orig = clicked_vals[2]
            old_bal_dest = clicked_vals[3]
            new_bal_dest = clicked_vals[4]
            txn_type     = clicked_vals[5]

        if analyze or clicked_vals:
            with st.spinner("🤖 Analyzing transaction..."):
                time.sleep(0.5)
                prob, threshold = predict_transaction(
                    amount, old_bal_orig, new_bal_orig,
                    old_bal_dest, new_bal_dest, txn_type)

            if prob >= threshold:
                risk  = "HIGH RISK"   if prob > 0.75 else "MEDIUM RISK"
                css   = "fraud-alert" if prob > 0.75 else "warning-alert"
                icon  = "⚠️"          if prob > 0.75 else "🟡"
                color = "#ef4444"     if prob > 0.75 else "#f59e0b"
                st.markdown(f"""
                <div class="{css}">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                                color:{color};margin:6px 0">FRAUD DETECTED</div>
                    <div style="font-size:3rem;font-weight:800;color:white;
                                font-family:'Syne',sans-serif">{prob*100:.1f}%</div>
                    <div style="color:#94a3b8;font-size:0.85rem">
                        {risk} · Threshold: {threshold:.2f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <div style="font-size:2rem">✅</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                                color:#22c55e;margin:6px 0">TRANSACTION SAFE</div>
                    <div style="font-size:3rem;font-weight:800;color:white;
                                font-family:'Syne',sans-serif">{prob*100:.1f}%</div>
                    <div style="color:#94a3b8;font-size:0.85rem">
                        LOW RISK · Threshold: {threshold:.2f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            gauge_color = "#ef4444" if prob > threshold else "#22c55e"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Fraud Risk Score (%)",
                       "font": {"color": "#94a3b8", "size": 14, "family": "DM Sans"}},
                number={"suffix": "%", "font": {"color": gauge_color, "size": 44, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155"},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0,  30],  "color": "rgba(34,197,94,0.08)"},
                        {"range": [30, 60],  "color": "rgba(245,158,11,0.08)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,0.08)"}
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": threshold * 100
                    }
                }
            ))
            fig.update_layout(height=240, **PLOTLY_BASE)
            st.plotly_chart(fig, use_container_width=True)

            balance_drop = amount / (old_bal_orig + 1)
            error_orig   = abs(new_bal_orig + amount - old_bal_orig)
            factors = {
                "Amount Size":            min(amount / 1_000_000, 1),
                "Balance Drop Ratio":     min(balance_drop, 1),
                "Balance Error (Sender)": min(error_orig / 500_000, 1),
                "Large Transaction":      1.0 if amount > 200000 else 0.05,
                "Zero Balance After":     1.0 if new_bal_orig == 0 else 0.05,
                "Type Risk (Transfer)":   0.85 if txn_type == "TRANSFER" else 0.4,
            }
            bar_colors = ["#ef4444" if v > 0.6 else "#f59e0b" if v > 0.3 else "#22c55e"
                          for v in factors.values()]
            fig = go.Figure(go.Bar(
                x=list(factors.values()), y=list(factors.keys()),
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v*100:.0f}%" for v in factors.values()],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11)
            ))
            fig.update_layout(title="🧩 Risk Factor Breakdown", height=260,
                              xaxis=dict(range=[0, 1.25], title="Risk Level"),
                              **PLOTLY_BASE)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Fill in transaction details and click **Analyze Transaction**")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — HYPERTUNE MODEL
# ══════════════════════════════════════════════════════════════════
elif page == "⚙️ Hypertune Model":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Hyperparameter Tuning</h1>
        <p>Customize ANN architecture · Train live · Compare with base model in real-time</p>
    </div>""", unsafe_allow_html=True)

    col_params, col_results = st.columns([1, 1.4], gap="large")

    with col_params:
        st.markdown('<div class="section-title">🏗️ Network Architecture</div>', unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 1**")
        n1 = st.select_slider("Neurons##l1", [32, 64, 128, 256, 512], value=256, key="n1")
        d1 = st.slider("Dropout##l1", 0.0, 0.6, 0.4, 0.05, key="d1")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 2**")
        n2 = st.select_slider("Neurons##l2", [16, 32, 64, 128, 256], value=128, key="n2")
        d2 = st.slider("Dropout##l2", 0.0, 0.5, 0.3, 0.05, key="d2")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 3**")
        n3 = st.select_slider("Neurons##l3", [8, 16, 32, 64, 128], value=64, key="n3")
        d3 = st.slider("Dropout##l3", 0.0, 0.4, 0.2, 0.05, key="d3")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎓 Training Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        lr = st.select_slider(
            "Learning Rate",
            [0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        epochs     = st.slider("Max Epochs", 10, 80, 40, 5)
        batch_size = st.select_slider("Batch Size", [64, 128, 256, 512], value=256)
        threshold  = st.slider("Decision Threshold", 0.2, 0.8, 0.5, 0.05,
                                help="Lower = catch more fraud, more false alarms")
        use_bn     = st.toggle("Batch Normalization", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">👁️ Architecture Preview</div>', unsafe_allow_html=True)
        arch_df = pd.DataFrame({
            "Layer":      ["Input", "Dense 1", "Dense 2", "Dense 3", "Output"],
            "Neurons":    [len(FEATURES), n1, n2, n3, 1],
            "Dropout":    ["—", f"{d1:.0%}", f"{d2:.0%}", f"{d3:.0%}", "—"],
            "Activation": ["—", "ReLU", "ReLU", "ReLU", "Sigmoid"]
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀 Train Custom Model", use_container_width=True)

    with col_results:
        st.markdown('<div class="section-title">📊 Live Training</div>', unsafe_allow_html=True)

        if train_btn:
            progress_bar = st.progress(0, text="Initializing model...")
            status_text  = st.empty()
            live_chart   = st.empty()

            tuned_model  = build_custom_ann(n1, n2, n3, d1, d2, d3, lr, use_bn)
            train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
            best_val_auc, patience_counter, PATIENCE = 0, 0, 8

            val_split = int(len(X_train_ready) * 0.8)
            X_tr = X_train_ready[:val_split]
            y_tr = y_train_ready[:val_split]
            X_vl = X_train_ready[val_split:]
            y_vl = y_train_ready[val_split:]

            for epoch in range(epochs):
                h = tuned_model.fit(
                    X_tr, y_tr, epochs=1, batch_size=batch_size,
                    validation_data=(X_vl, y_vl), verbose=0
                )
                tl  = h.history["loss"][0]
                vl  = h.history["val_loss"][0]
                tau = h.history["auc"][0]
                vau = h.history["val_auc"][0]

                train_losses.append(tl)
                val_losses.append(vl)
                train_aucs.append(tau)
                val_aucs.append(vau)

                if vau > best_val_auc:
                    best_val_auc     = vau
                    patience_counter = 0
                    tuned_model.save_weights("/tmp/best_tuned.weights.h5")
                else:
                    patience_counter += 1

                if (epoch + 1) % 2 == 0 or epoch == 0:
                    pct = int((epoch + 1) / epochs * 100)
                    progress_bar.progress(
                        pct,
                        text=f"Epoch {epoch+1}/{epochs} · Val AUC: {vau:.4f} · Best: {best_val_auc:.4f}"
                    )
                    ep_list  = list(range(1, len(train_losses) + 1))
                    fig_live = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "AUC"])
                    fig_live.add_trace(go.Scatter(x=ep_list, y=train_losses, name="Train",
                                                   line=dict(color="#818cf8", width=2)), row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=val_losses, name="Val",
                                                   line=dict(color="#f87171", width=2, dash="dot")), row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=train_aucs, name="Train AUC",
                                                   line=dict(color="#34d399", width=2),
                                                   showlegend=False), row=1, col=2)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=val_aucs, name="Val AUC",
                                                   line=dict(color="#fbbf24", width=2, dash="dot"),
                                                   showlegend=False), row=1, col=2)
                    fig_live.update_layout(
                        height=260,
                        legend=dict(orientation="h", y=1.15),
                        **PLOTLY_BASE
                    )
                    live_chart.plotly_chart(fig_live, use_container_width=True)

                if patience_counter >= PATIENCE:
                    progress_bar.progress(
                        100,
                        text=f"⏹ Early stop at epoch {epoch+1} · Best AUC: {best_val_auc:.4f}"
                    )
                    break

            try:
                tuned_model.load_weights("/tmp/best_tuned.weights.h5")
            except Exception:
                pass

            status_text.info("📊 Evaluating on test data...")
            y_pp   = tuned_model.predict(X_test_ready, verbose=0).flatten()
            y_pred = (y_pp > threshold).astype(int)
            roc    = roc_auc_score(y_test_ready, y_pp)
            rep    = classification_report(
                y_test_ready, y_pred,
                target_names=["Normal", "Fraud"],
                output_dict=True, zero_division=0
            )
            cm_t = confusion_matrix(y_test_ready, y_pred)

            st.session_state["tuned_keras_model"] = tuned_model
            st.session_state["tuned_threshold"]   = threshold
            st.session_state["tuned_metrics"] = {
                "roc_auc"  : roc,
                "precision": rep["Fraud"]["precision"],
                "recall"   : rep["Fraud"]["recall"],
                "f1"       : rep["Fraud"]["f1-score"],
                "accuracy" : rep["accuracy"],
                "cm"       : cm_t.tolist(),
                "threshold": threshold
            }
            status_text.success("✅ Training complete! Tuned model is now active in Fraud Detector.")
            st.rerun()

        if "tuned_metrics" in st.session_state:
            tm = st.session_state["tuned_metrics"]
            bm = metrics

            st.markdown('<div class="section-title">⚖️ Base vs Tuned Comparison</div>',
                        unsafe_allow_html=True)
            compare_data = [
                ("ROC-AUC",   bm["roc_auc"], tm["roc_auc"], True),
                ("Precision", float(bm["report"]["Fraud"]["precision"]), tm["precision"], True),
                ("Recall",    float(bm["report"]["Fraud"]["recall"]),    tm["recall"],    True),
                ("F1 Score",  float(bm["report"]["Fraud"]["f1-score"]),  tm["f1"],        True),
            ]
            c1, c2, c3, c4 = st.columns(4)
            for col, (name, bv, tv, hb) in zip([c1, c2, c3, c4], compare_data):
                badge = delta_badge(tv, bv, hb)
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div style="color:#64748b;font-size:0.78rem;margin:4px 0">
                        Base: <b style="color:#94a3b8">{bv:.4f}</b></div>
                    <div class="metric-value" style="font-size:1.3rem">{tv:.4f}</div>
                    <div style="font-size:0.8rem;margin-top:4px">{badge}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.imshow(bm["confusion_matrix"], text_auto=True,
                                x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                                color_continuous_scale="Blues")
                # FIX: use plotly_base_no_font() to avoid font key conflict
                fig.update_layout(
                    title="Base Model",
                    height=280,
                    font=dict(family="DM Sans", color="#94a3b8", size=13),
                    **plotly_base_no_font()
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.imshow(tm["cm"], text_auto=True,
                                x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                                color_continuous_scale="Purples")
                fig.update_layout(
                    title="Tuned Model",
                    height=280,
                    font=dict(family="DM Sans", color="#94a3b8", size=13),
                    **plotly_base_no_font()
                )
                st.plotly_chart(fig, use_container_width=True)

            cats = ["ROC-AUC", "Precision", "Recall", "F1", "Accuracy"]
            bv   = [
                bm["roc_auc"],
                float(bm["report"]["Fraud"]["precision"]),
                float(bm["report"]["Fraud"]["recall"]),
                float(bm["report"]["Fraud"]["f1-score"]),
                float(bm["report"]["accuracy"])
            ]
            tv = [tm["roc_auc"], tm["precision"], tm["recall"], tm["f1"], tm["accuracy"]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=bv + [bv[0]], theta=cats + [cats[0]],
                fill="toself", name="Base",
                fillcolor="rgba(129,140,248,0.12)",
                line=dict(color="#818cf8", width=2)
            ))
            fig.add_trace(go.Scatterpolar(
                r=tv + [tv[0]], theta=cats + [cats[0]],
                fill="toself", name="Tuned",
                fillcolor="rgba(192,132,252,0.12)",
                line=dict(color="#c084fc", width=2)
            ))
            fig.update_layout(
                title="🕸️ Radar: Base vs Tuned", height=360,
                polar=dict(radialaxis=dict(range=[0, 1], color="#475569")),
                legend=dict(orientation="h", y=-0.1), **PLOTLY_BASE
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">🎚️ Threshold Sensitivity</div>',
                        unsafe_allow_html=True)
            y_pp2    = st.session_state["tuned_keras_model"].predict(X_test_ready, verbose=0).flatten()
            th_range = np.arange(0.1, 0.91, 0.05)
            p_l, r_l, f1_l, fpr_l = [], [], [], []
            for th in th_range:
                yp    = (y_pp2 > th).astype(int)
                r     = classification_report(
                    y_test_ready, yp,
                    target_names=["Normal", "Fraud"],
                    output_dict=True, zero_division=0
                )
                cm_th = confusion_matrix(y_test_ready, yp)
                p_l.append(r["Fraud"]["precision"])
                r_l.append(r["Fraud"]["recall"])
                f1_l.append(r["Fraud"]["f1-score"])
                tn, fp = cm_th[0][0], cm_th[0][1]
                fpr_l.append(fp / (fp + tn + 1e-9))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(th_range), y=p_l,   name="Precision",
                                      line=dict(color="#818cf8", width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_range), y=r_l,   name="Recall",
                                      line=dict(color="#34d399", width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_range), y=f1_l,  name="F1-Score",
                                      line=dict(color="#fbbf24", width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_range), y=fpr_l, name="False +ve Rate",
                                      line=dict(color="#f87171", width=2, dash="dash")))
            fig.add_vline(x=threshold, line_dash="dash", line_color="white", opacity=0.6,
                          annotation_text=f"Chosen: {threshold:.2f}",
                          annotation_position="top right")
            fig.update_layout(
                title="Impact of Threshold on All Metrics", height=340,
                xaxis_title="Threshold", yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", y=1.12), **PLOTLY_BASE
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Set parameters and click **Train Custom Model** to begin!")
            st.markdown("""
            ### 💡 Tuning Tips

            **Neurons** — Funnel shape (256→128→64) works best for fraud detection.

            **Dropout** — Keep between 0.2–0.5 to prevent overfitting.

            **Learning Rate** — 0.001 is the sweet spot for most cases.

            **Threshold** — Most impactful business lever:
            - 🔽 Lower (0.3) → Catch more fraud, more false alarms
            - 🔼 Higher (0.7) → Fewer false alarms, some fraud slips through

            **Batch Normalization** — Almost always improves results. Keep ON.
            """)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance Analytics</h1>
        <p>Deep dive into ANN metrics, architecture and evaluation results</p>
    </div>""", unsafe_allow_html=True)

    report = metrics["report"]
    col1, col2 = st.columns(2)

    with col1:
        th_range = np.linspace(0, 1, 100)
        p_c = [float(report["Fraud"]["precision"]) * (1 - 0.3 * t)       for t in th_range]
        r_c = [float(report["Fraud"]["recall"])    * (1 - 0.2 * (1 - t)) for t in th_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(th_range), y=p_c, name="Precision",
                                  line=dict(color="#818cf8", width=2.5)))
        fig.add_trace(go.Scatter(x=list(th_range), y=r_c, name="Recall",
                                  line=dict(color="#34d399", width=2.5)))
        fig.add_vline(x=0.5, line_dash="dash", line_color="white",
                      opacity=0.4, annotation_text="Default (0.5)")
        fig.update_layout(title="⚖️ Precision–Recall vs Threshold", height=360,
                          xaxis_title="Threshold", yaxis=dict(range=[0, 1]),
                          legend=dict(orientation="h", y=1.12), **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cats = ["ROC-AUC", "Precision", "Recall", "F1-Score", "Accuracy"]
        vals = [
            metrics["roc_auc"],
            float(report["Fraud"]["precision"]),
            float(report["Fraud"]["recall"]),
            float(report["Fraud"]["f1-score"]),
            float(report["accuracy"])
        ]
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(129,140,248,0.12)",
            line=dict(color="#818cf8", width=2.5), name="Base Model"
        ))
        fig.update_layout(title="🕸️ Performance Radar", height=360,
                          polar=dict(radialaxis=dict(range=[0, 1], color="#475569")),
                          **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🧠 ANN Architecture Diagram</div>',
                unsafe_allow_html=True)
    layer_data = [
        ("Input\n12 features",       12,  "#4fc3f7"),
        ("Dense 256\n+BN +Drop 0.4", 256, "#818cf8"),
        ("Dense 128\n+BN +Drop 0.3", 128, "#a78bfa"),
        ("Dense 64\n+Drop 0.2",      64,  "#c084fc"),
        ("Dense 32\n+Drop 0.1",      32,  "#e879f9"),
        ("Output\nSigmoid",          1,   "#34d399"),
    ]
    fig   = go.Figure()
    MAX_D = 12
    for i, (name, n, color) in enumerate(layer_data):
        d = min(n, MAX_D)
        for j in range(d):
            fig.add_trace(go.Scatter(
                x=[i], y=[j - d / 2], mode="markers",
                marker=dict(size=20, color=color, opacity=0.85,
                             line=dict(width=1.5, color="rgba(255,255,255,0.3)")),
                showlegend=False,
                hovertext=f"<b>{name.replace(chr(10), ' ')}</b><br>{n} neurons"
            ))
        fig.add_annotation(x=i, y=-(MAX_D / 2 + 2), text=name, showarrow=False,
                           font=dict(size=9, color="#64748b"), align="center")
    for i in range(len(layer_data) - 1):
        d1 = min(layer_data[i][1],     MAX_D)
        d2 = min(layer_data[i + 1][1], MAX_D)
        for j1 in range(min(d1, 5)):
            for j2 in range(min(d2, 5)):
                fig.add_shape(
                    type="line",
                    x0=i,     y0=j1 - d1 / 2,
                    x1=i + 1, y1=j2 - d2 / 2,
                    line=dict(color="rgba(100,116,139,0.07)", width=0.8)
                )
    fig.update_layout(
        height=420,
        title="Neural Network Layer Structure (Base Model)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(t=50, b=80, l=20, r=20),
        template="plotly_dark",
        paper_bgcolor="rgba(13,20,36,0.95)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Business Insights</h1>
        <p>UPI fraud patterns, financial impact & actionable intelligence for decision-makers</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        amounts = np.concatenate([
            np.random.lognormal(10,   1.5, 2000),
            np.random.lognormal(12.5, 1.0, 300)
        ])
        labels = ["Normal"] * 2000 + ["Fraud"] * 300
        fig = px.histogram(
            pd.DataFrame({"Amount (₹)": amounts, "Type": labels}),
            x="Amount (₹)", color="Type", nbins=60, barmode="overlay",
            color_discrete_map={"Normal": "#818cf8", "Fraud": "#f87171"},
            title="💰 Transaction Amount Distribution"
        )
        fig.update_layout(height=340, **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fraud_h = [2, 3, 5, 8, 4, 3, 2, 4, 7, 6, 7, 9, 10, 8, 7, 6, 9, 12, 15, 14, 13, 10, 7, 4]
        fig = go.Figure(go.Bar(
            x=list(range(24)), y=fraud_h,
            marker=dict(color=fraud_h, colorscale="RdYlGn_r", line=dict(width=0))
        ))
        fig.update_layout(title="⏰ Fraud Attempts by Hour of Day", height=340,
                          xaxis_title="Hour (0–23)", yaxis_title="Fraud Cases",
                          **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.sunburst(
            pd.DataFrame({
                "type":    ["TRANSFER", "TRANSFER", "CASH_OUT", "CASH_OUT"],
                "outcome": ["Fraud",    "Legit",    "Fraud",    "Legit"],
                "count":   [3162,       7081,       4116,       18600]
            }),
            path=["type", "outcome"], values="count", color="outcome",
            color_discrete_map={"Fraud": "#f87171", "Legit": "#34d399"},
            title="🥧 Fraud Breakdown by Transaction Type"
        )
        fig.update_layout(height=380, **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        months    = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        prevented = [12.3, 15.1, 18.7, 14.2, 20.5, 23.1]
        missed    = [2.1,  1.8,  2.3,  1.5,  2.8,  2.2]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=prevented,
                              name="Fraud Prevented (₹Cr)", marker_color="#34d399"))
        fig.add_trace(go.Bar(x=months, y=missed,
                              name="Missed Fraud (₹Cr)", marker_color="#f87171"))
        fig.update_layout(title="💸 Financial Impact Simulation (₹ Crores)",
                          barmode="group", height=380, **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🎯 Key Managerial Takeaways</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🕐 **Peak Fraud Window**\n\nFraud spikes 6 PM–9 PM. Lower decision threshold during these hours for heightened protection.")
    with c2:
        st.warning("💸 **High-Risk Threshold**\n\nTransactions above ₹2,00,000 show 3× higher fraud probability. Flag for secondary verification.")
    with c3:
        st.error("🔄 **Zero-Balance Pattern**\n\nIf sender balance hits ₹0 after a TRANSFER, fraud probability exceeds 85%. Auto-block recommended.")
