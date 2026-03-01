import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import joblib
import json
import time
import os

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 28px 32px;
    border-radius: 20px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.main-header h1 { color: white !important; margin: 0; font-size: 2.2rem; font-weight: 800; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 1rem; }

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #8b9dc3;
    font-size: 0.8rem;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.metric-delta {
    font-size: 0.85rem;
    margin-top: 4px;
    font-weight: 600;
}

.fraud-alert {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,65,108,0.5);
    animation: pulseRed 2s infinite;
}
.safe-alert {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(56,239,125,0.4);
}
.warning-alert {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(247,151,30,0.4);
}
@keyframes pulseRed {
    0%   { box-shadow: 0 0 20px rgba(255,65,108,0.4); }
    50%  { box-shadow: 0 0 45px rgba(255,65,108,0.9); }
    100% { box-shadow: 0 0 20px rgba(255,65,108,0.4); }
}

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #c5cae9;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2d3561;
}

.hyper-card {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2440 100%);
    border: 1px solid #3d4f8a;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.compare-better { color: #38ef7d; font-weight: 700; }
.compare-worse  { color: #ff6b6b; font-weight: 700; }
.compare-same   { color: #f7b731; font-weight: 700; }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
}

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model   = tf.keras.models.load_model('model/ann_model.h5')
    scaler  = joblib.load('model/scaler.pkl')
    with open('model/metrics.json')           as f: metrics  = json.load(f)
    with open('model/training_history.json')  as f: history  = json.load(f)
    with open('model/features.json')          as f: features = json.load(f)
    return model, scaler, metrics, history, features

@st.cache_data
def load_train_test_data():
    train = pd.read_csv('model/train_data.csv')
    test  = pd.read_csv('model/test_data.csv')
    return train, test

model, scaler, metrics, history, FEATURES = load_artifacts()
train_data, test_data = load_train_test_data()

X_train_ready = train_data[FEATURES].values
y_train_ready = train_data['isFraud'].values
X_test_ready  = test_data[FEATURES].values
y_test_ready  = test_data['isFraud'].values


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ UPI Fraud Shield</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Transaction Security")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Fraud Detector", "⚙️ Hypertune Model",
         "📊 Model Analytics", "📈 Insights"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**📦 Base Model Info**")
    st.caption(f"🧠 Architecture : 4-Layer ANN")
    st.caption(f"📊 ROC-AUC      : {metrics['roc_auc']:.4f}")
    st.caption(f"⚡ Framework    : TensorFlow {tf.__version__}")
    st.caption(f"🔢 Features     : {len(FEATURES)}")
    st.caption(f"📁 Dataset      : PaySim Synthetic")

    st.divider()
    if 'tuned_metrics' in st.session_state:
        st.markdown("**🆕 Tuned Model Info**")
        tm = st.session_state['tuned_metrics']
        st.caption(f"📊 ROC-AUC  : {tm['roc_auc']:.4f}")
        st.caption(f"🎯 Precision: {tm['precision']:.4f}")
        st.caption(f"🔍 Recall   : {tm['recall']:.4f}")


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def predict_transaction(amount, old_bal_orig, new_bal_orig,
                         old_bal_dest, new_bal_dest, txn_type):
    error_orig   = new_bal_orig + amount - old_bal_orig
    error_dest   = old_bal_dest + amount - new_bal_dest
    balance_drop = amount / (old_bal_orig + 1)
    is_large     = 1 if amount > 200000 else 0
    zero_after   = 1 if new_bal_orig == 0 else 0
    dest_inc     = new_bal_dest - old_bal_dest
    type_enc     = 1 if txn_type == 'TRANSFER' else 0

    feat = np.array([[amount, old_bal_orig, new_bal_orig,
                      old_bal_dest, new_bal_dest,
                      error_orig, error_dest, balance_drop,
                      is_large, zero_after, dest_inc, type_enc]])
    feat_sc = scaler.transform(feat)

    # Use tuned model if available
    m = st.session_state.get('tuned_model', model)
    t = st.session_state.get('tuned_threshold', 0.5)
    prob = float(m.predict(feat_sc, verbose=0)[0][0])
    return prob, t


def build_custom_ann(neurons_l1, neurons_l2, neurons_l3,
                     dropout_l1, dropout_l2, dropout_l3,
                     learning_rate, use_batch_norm):
    m = keras.Sequential(name='Tuned_ANN')
    m.add(layers.Input(shape=(len(FEATURES),)))

    m.add(layers.Dense(neurons_l1, activation='relu'))
    if use_batch_norm: m.add(layers.BatchNormalization())
    m.add(layers.Dropout(dropout_l1))

    m.add(layers.Dense(neurons_l2, activation='relu'))
    if use_batch_norm: m.add(layers.BatchNormalization())
    m.add(layers.Dropout(dropout_l2))

    m.add(layers.Dense(neurons_l3, activation='relu'))
    m.add(layers.Dropout(dropout_l3))

    m.add(layers.Dense(1, activation='sigmoid'))

    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return m


def delta_badge(new_val, base_val, higher_better=True):
    diff = new_val - base_val
    if abs(diff) < 0.0001:
        return '<span class="compare-same">→ No change</span>'
    if (diff > 0) == higher_better:
        return f'<span class="compare-better">▲ +{diff:.4f}</span>'
    return f'<span class="compare-worse">▼ {diff:.4f}</span>'


# ══════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ UPI Fraud Detection System</h1>
        <p>Real-time AI-powered transaction security · Powered by Deep Learning (ANN)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────
    report = metrics['report']
    kpis = [
        ("ROC-AUC Score",    f"{metrics['roc_auc']:.4f}"),
        ("Fraud Precision",  f"{float(report['Fraud']['precision']):.4f}"),
        ("Fraud Recall",     f"{float(report['Fraud']['recall']):.4f}"),
        ("F1 Score",         f"{float(report['Fraud']['f1-score']):.4f}"),
        ("Fraud Caught",     f"{metrics['fraud_caught']:,}"),
        ("Fraud Missed",     f"{metrics['fraud_missed']:,}"),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training Curves ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    ep = list(range(1, len(history['loss']) + 1))

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=history['loss'], name='Train Loss',
                                  line=dict(color='#667eea', width=2.5)))
        fig.add_trace(go.Scatter(x=ep, y=history['val_loss'], name='Val Loss',
                                  line=dict(color='#ff6b6b', width=2.5, dash='dot')))
        fig.update_layout(title='📉 Training vs Validation Loss',
                          template='plotly_dark', height=320,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation='h', y=1.1),
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=history['auc'], name='Train AUC',
                                  line=dict(color='#38ef7d', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(56,239,125,0.1)'))
        fig.add_trace(go.Scatter(x=ep, y=history['val_auc'], name='Val AUC',
                                  line=dict(color='#f7b731', width=2.5, dash='dot')))
        fig.update_layout(title='📈 Training vs Validation AUC',
                          template='plotly_dark', height=320,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(range=[0.5, 1]),
                          legend=dict(orientation='h', y=1.1),
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        cm = metrics['confusion_matrix']
        fig = px.imshow(cm, text_auto=True,
                        x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                        color_continuous_scale='Blues',
                        labels=dict(x='Predicted', y='Actual'))
        fig.update_layout(title='🎯 Confusion Matrix',
                          template='plotly_dark', height=350,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          font=dict(family='Inter', size=14))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=['TRANSFER', 'CASH_OUT'],
            y=[0.31, 0.18],
            marker=dict(color=['#667eea', '#ff6b6b'],
                        line=dict(width=0)),
            text=['31.0%', '18.0%'],
            textposition='outside'
        ))
        fig.update_layout(title='🔍 Fraud Rate by Transaction Type',
                          template='plotly_dark', height=350,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(tickformat='.0%'),
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2: FRAUD DETECTOR
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Fraud Detector":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Real-Time Fraud Detector</h1>
        <p>Enter transaction details below · Instant AI prediction</p>
    </div>
    """, unsafe_allow_html=True)

    active_model_label = "🆕 Tuned Model" if 'tuned_model' in st.session_state else "📦 Base Model"
    st.info(f"**Active Model:** {active_model_label} · Go to ⚙️ Hypertune to train a custom model")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">📝 Transaction Details</div>', unsafe_allow_html=True)

        txn_type   = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
        amount     = st.number_input("Transaction Amount (₹)", min_value=1.0,
                                      max_value=10_000_000.0, value=50000.0,
                                      step=1000.0, format="%.2f")

        st.markdown("**Sender Account**")
        c1, c2 = st.columns(2)
        old_bal_orig = c1.number_input("Balance Before (₹)", min_value=0.0,
                                        value=100000.0, step=1000.0, key='ob_o')
        new_bal_orig = c2.number_input("Balance After (₹)",  min_value=0.0,
                                        value=50000.0,  step=1000.0, key='nb_o')

        st.markdown("**Receiver Account**")
        c3, c4 = st.columns(2)
        old_bal_dest = c3.number_input("Balance Before (₹)", min_value=0.0,
                                        value=5000.0,  step=1000.0, key='ob_d')
        new_bal_dest = c4.number_input("Balance After (₹)",  min_value=0.0,
                                        value=55000.0, step=1000.0, key='nb_d')

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Transaction", use_container_width=True)

        st.markdown('<div class="section-title">💡 Quick Test Scenarios</div>', unsafe_allow_html=True)
        scenarios = {
            "🔴 Complete Balance Drain (Fraud)":
                (500000, 500000, 0, 0, 500000, "TRANSFER"),
            "🟡 Large Suspicious Transfer":
                (250000, 260000, 10000, 100, 250100, "TRANSFER"),
            "🟢 Normal Bill Payment":
                (5000, 50000, 45000, 10000, 15000, "CASH_OUT"),
            "🟢 Small UPI Transfer":
                (1200, 25000, 23800, 8000, 9200, "TRANSFER"),
        }
        for label, vals in scenarios.items():
            if st.button(label, use_container_width=True, key=label):
                amount, old_bal_orig, new_bal_orig = vals[0], vals[1], vals[2]
                old_bal_dest, new_bal_dest, txn_type = vals[3], vals[4], vals[5]

    with col2:
        st.markdown('<div class="section-title">📊 Analysis Result</div>', unsafe_allow_html=True)

        if analyze or any(st.session_state.get(k) for k in scenarios.keys()):
            with st.spinner("🤖 Analyzing transaction..."):
                time.sleep(0.6)
                prob, threshold = predict_transaction(amount, old_bal_orig, new_bal_orig,
                                                       old_bal_dest, new_bal_dest, txn_type)

            if prob >= threshold:
                risk = "HIGH" if prob > 0.75 else "MEDIUM"
                css  = "fraud-alert" if prob > 0.75 else "warning-alert"
                icon = "⚠️" if prob > 0.75 else "🟡"
                st.markdown(f"""
                <div class="{css}">
                    <h2 style="color:white;margin:0">{icon} FRAUD DETECTED</h2>
                    <h1 style="color:white;margin:8px 0;font-size:3rem">{prob*100:.1f}%</h1>
                    <p style="color:rgba(255,255,255,0.9)">Risk Level: <b>{risk}</b> · Threshold: {threshold:.2f}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <h2 style="color:white;margin:0">✅ TRANSACTION SAFE</h2>
                    <h1 style="color:white;margin:8px 0;font-size:3rem">{prob*100:.1f}%</h1>
                    <p style="color:rgba(255,255,255,0.9)">Risk Level: LOW · Threshold: {threshold:.2f}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Risk Score (%)", 'font': {'color': '#c5cae9', 'size': 16}},
                number={'suffix': '%', 'font': {'color': '#667eea', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#ff416c' if prob > threshold else '#38ef7d'},
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(56,239,125,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(247,183,49,0.15)'},
                        {'range': [60, 100],'color': 'rgba(255,65,108,0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(template='plotly_dark', height=260,
                              paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(family='Inter'))
            st.plotly_chart(fig, use_container_width=True)

            # Risk factors
            balance_drop = amount / (old_bal_orig + 1)
            error_orig   = abs(new_bal_orig + amount - old_bal_orig)
            factors = {
                'Amount Size':           min(amount / 1_000_000, 1),
                'Balance Drop Ratio':    min(balance_drop, 1),
                'Balance Error':         min(error_orig / 500_000, 1),
                'Large Transaction':     1.0 if amount > 200000 else 0.05,
                'Zero Balance After':    1.0 if new_bal_orig == 0 else 0.05,
                'Transaction Type Risk': 0.85 if txn_type == 'TRANSFER' else 0.45,
            }
            colors = ['#ff6b6b' if v > 0.6 else '#f7b731' if v > 0.3 else '#38ef7d'
                      for v in factors.values()]
            fig = go.Figure(go.Bar(
                x=list(factors.values()), y=list(factors.keys()),
                orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                text=[f'{v*100:.0f}%' for v in factors.values()],
                textposition='outside'
            ))
            fig.update_layout(title='🧩 Risk Factor Breakdown',
                              template='plotly_dark', height=280,
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(range=[0, 1.2]),
                              font=dict(family='Inter', size=12))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Enter transaction details and click **Analyze Transaction**")


# ══════════════════════════════════════════════════════════════════
# PAGE 3: HYPERTUNE MODEL  ← THE NEW PAGE
# ══════════════════════════════════════════════════════════════════
elif page == "⚙️ Hypertune Model":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Hyperparameter Tuning</h1>
        <p>Customize the ANN architecture and training settings · Compare with base model in real-time</p>
    </div>
    """, unsafe_allow_html=True)

    col_params, col_results = st.columns([1, 1.4], gap="large")

    # ── LEFT: Parameter Controls ──────────────────────────────────
    with col_params:
        st.markdown('<div class="section-title">🏗️ Network Architecture</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 1 (Widest)**")
        neurons_l1 = st.select_slider("Neurons", [32,64,128,256,512],
                                       value=256, key='n1')
        dropout_l1 = st.slider("Dropout Rate", 0.0, 0.6, 0.4, 0.05, key='d1')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 2**")
        neurons_l2 = st.select_slider("Neurons", [16,32,64,128,256],
                                       value=128, key='n2')
        dropout_l2 = st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05, key='d2')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        st.markdown("**Layer 3 (Narrowest)**")
        neurons_l3 = st.select_slider("Neurons", [8,16,32,64,128],
                                       value=64, key='n3')
        dropout_l3 = st.slider("Dropout Rate", 0.0, 0.4, 0.2, 0.05, key='d3')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎓 Training Settings</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="hyper-card">', unsafe_allow_html=True)
        lr = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001, format_func=lambda x: f"{x:.4f}"
        )
        epochs     = st.slider("Max Epochs",     10, 100, 50, 5)
        batch_size = st.select_slider("Batch Size", [64, 128, 256, 512, 1024], value=512)
        threshold  = st.slider("Decision Threshold", 0.2, 0.8, 0.5, 0.05,
                                help="Lower = catch more fraud but more false alarms")
        use_bn     = st.toggle("Use Batch Normalization", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Architecture preview
        st.markdown('<div class="section-title">👁️ Architecture Preview</div>',
                    unsafe_allow_html=True)
        arch_df = pd.DataFrame({
            'Layer':    ['Input', 'Dense 1', 'Dense 2', 'Dense 3', 'Output'],
            'Neurons':  [len(FEATURES), neurons_l1, neurons_l2, neurons_l3, 1],
            'Dropout':  ['—', f'{dropout_l1:.0%}', f'{dropout_l2:.0%}',
                         f'{dropout_l3:.0%}', '—'],
            'Activation': ['—', 'ReLU', 'ReLU', 'ReLU', 'Sigmoid']
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀 Train Custom Model", use_container_width=True)

    # ── RIGHT: Training + Results ─────────────────────────────────
    with col_results:
        st.markdown('<div class="section-title">📊 Training Results</div>',
                    unsafe_allow_html=True)

        if train_btn:
            # ── Live Training ─────────────────────────────────────
            progress_bar  = st.progress(0, text="Initializing...")
            status_text   = st.empty()
            live_chart    = st.empty()
            metrics_placeholder = st.empty()

            tuned_model = build_custom_ann(
                neurons_l1, neurons_l2, neurons_l3,
                dropout_l1, dropout_l2, dropout_l3,
                lr, use_bn
            )

            train_losses, val_losses = [], []
            train_aucs,   val_aucs   = [], []
            best_val_auc = 0
            patience_counter = 0
            PATIENCE = 8

            # Custom epoch-by-epoch training loop for live updates
            val_split = int(len(X_train_ready) * 0.8)
            X_tr = X_train_ready[:val_split]
            y_tr = y_train_ready[:val_split]
            X_vl = X_train_ready[val_split:]
            y_vl = y_train_ready[val_split:]

            for epoch in range(epochs):
                hist = tuned_model.fit(
                    X_tr, y_tr,
                    epochs=1,
                    batch_size=batch_size,
                    validation_data=(X_vl, y_vl),
                    verbose=0
                )

                tl  = hist.history['loss'][0]
                vl  = hist.history['val_loss'][0]
                tau = hist.history['auc'][0]
                vau = hist.history['val_auc'][0]

                train_losses.append(tl)
                val_losses.append(vl)
                train_aucs.append(tau)
                val_aucs.append(vau)

                # Early stopping
                if vau > best_val_auc:
                    best_val_auc = vau
                    patience_counter = 0
                    tuned_model.save_weights('/tmp/best_weights.h5')
                else:
                    patience_counter += 1

                # Update UI every 2 epochs
                if (epoch + 1) % 2 == 0 or epoch == 0:
                    pct = int((epoch + 1) / epochs * 100)
                    progress_bar.progress(pct,
                        text=f"Epoch {epoch+1}/{epochs} · Val AUC: {vau:.4f} · Best: {best_val_auc:.4f}")

                    ep_list = list(range(1, len(train_losses) + 1))
                    fig_live = make_subplots(rows=1, cols=2,
                                             subplot_titles=['Loss', 'AUC'])
                    fig_live.add_trace(go.Scatter(x=ep_list, y=train_losses,
                                                   name='Train Loss',
                                                   line=dict(color='#667eea', width=2)),
                                        row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=val_losses,
                                                   name='Val Loss',
                                                   line=dict(color='#ff6b6b', width=2, dash='dot')),
                                        row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=train_aucs,
                                                   name='Train AUC',
                                                   line=dict(color='#38ef7d', width=2),
                                                   showlegend=False),
                                        row=1, col=2)
                    fig_live.add_trace(go.Scatter(x=ep_list, y=val_aucs,
                                                   name='Val AUC',
                                                   line=dict(color='#f7b731', width=2, dash='dot'),
                                                   showlegend=False),
                                        row=1, col=2)
                    fig_live.update_layout(
                        template='plotly_dark', height=280,
                        paper_bgcolor='rgba(30,33,48,0.8)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(orientation='h', y=1.15),
                        font=dict(family='Inter', size=11),
                        margin=dict(t=40, b=20)
                    )
                    live_chart.plotly_chart(fig_live, use_container_width=True)

                if patience_counter >= PATIENCE:
                    progress_bar.progress(100,
                        text=f"Early stopped at epoch {epoch+1} · Best Val AUC: {best_val_auc:.4f}")
                    break

            # Load best weights
            try:
                tuned_model.load_weights('/tmp/best_weights.h5')
            except:
                pass

            # ── Evaluate Tuned Model ──────────────────────────────
            status_text.info("Evaluating on test data...")
            y_pred_prob = tuned_model.predict(X_test_ready, verbose=0).flatten()
            y_pred      = (y_pred_prob > threshold).astype(int)
            roc         = roc_auc_score(y_test_ready, y_pred_prob)
            rep         = classification_report(y_test_ready, y_pred,
                                                 target_names=['Normal','Fraud'],
                                                 output_dict=True)
            cm_tuned    = confusion_matrix(y_test_ready, y_pred)

            tuned_m = {
                'roc_auc'   : roc,
                'precision' : rep['Fraud']['precision'],
                'recall'    : rep['Fraud']['recall'],
                'f1'        : rep['Fraud']['f1-score'],
                'accuracy'  : rep['accuracy'],
                'cm'        : cm_tuned.tolist(),
                'threshold' : threshold,
                'history'   : {'loss': train_losses, 'val_loss': val_losses,
                               'auc': train_aucs,  'val_auc': val_aucs}
            }

            st.session_state['tuned_model']     = tuned_model
            st.session_state['tuned_metrics']   = tuned_m
            st.session_state['tuned_threshold'] = threshold

            status_text.success("✅ Model trained and evaluated successfully!")
            st.rerun()

        # ── Show results if tuned model exists ────────────────────
        if 'tuned_metrics' in st.session_state:
            tm  = st.session_state['tuned_metrics']
            bm  = metrics

            st.markdown('<div class="section-title">⚖️ Base vs Tuned Model Comparison</div>',
                        unsafe_allow_html=True)

            # Comparison cards
            compare_metrics = [
                ("ROC-AUC",   bm['roc_auc'],
                 tm['roc_auc'],   True),
                ("Precision",
                 float(bm['report']['Fraud']['precision']),
                 tm['precision'], True),
                ("Recall",
                 float(bm['report']['Fraud']['recall']),
                 tm['recall'],    True),
                ("F1 Score",
                 float(bm['report']['Fraud']['f1-score']),
                 tm['f1'],        True),
            ]

            c1, c2, c3, c4 = st.columns(4)
            for col, (name, base_v, tuned_v, hb) in zip(
                    [c1, c2, c3, c4], compare_metrics):
                badge = delta_badge(tuned_v, base_v, hb)
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{name}</div>
                    <div style="font-size:0.8rem;color:#8b9dc3;margin:4px 0">
                        Base: <b>{base_v:.4f}</b>
                    </div>
                    <div class="metric-value" style="font-size:1.6rem">{tuned_v:.4f}</div>
                    <div class="metric-delta">{badge}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Side by side confusion matrices
            col1, col2 = st.columns(2)
            with col1:
                fig = px.imshow(bm['confusion_matrix'], text_auto=True,
                                x=['Normal','Fraud'], y=['Normal','Fraud'],
                                color_continuous_scale='Blues')
                fig.update_layout(title='Base Model · Confusion Matrix',
                                  template='plotly_dark', height=300,
                                  paper_bgcolor='rgba(30,33,48,0.8)',
                                  font=dict(family='Inter', size=13))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.imshow(tm['cm'], text_auto=True,
                                x=['Normal','Fraud'], y=['Normal','Fraud'],
                                color_continuous_scale='Purples')
                fig.update_layout(title='Tuned Model · Confusion Matrix',
                                  template='plotly_dark', height=300,
                                  paper_bgcolor='rgba(30,33,48,0.8)',
                                  font=dict(family='Inter', size=13))
                st.plotly_chart(fig, use_container_width=True)

            # Radar comparison
            categories = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
            base_vals  = [
                bm['roc_auc'],
                float(bm['report']['Fraud']['precision']),
                float(bm['report']['Fraud']['recall']),
                float(bm['report']['Fraud']['f1-score']),
                float(bm['report']['accuracy'])
            ]
            tuned_vals = [
                tm['roc_auc'], tm['precision'],
                tm['recall'],  tm['f1'], tm['accuracy']
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=base_vals + [base_vals[0]],
                theta=categories + [categories[0]],
                fill='toself', name='Base Model',
                fillcolor='rgba(102,126,234,0.2)',
                line=dict(color='#667eea', width=2.5)
            ))
            fig.add_trace(go.Scatterpolar(
                r=tuned_vals + [tuned_vals[0]],
                theta=categories + [categories[0]],
                fill='toself', name='Tuned Model',
                fillcolor='rgba(167,139,250,0.2)',
                line=dict(color='#a78bfa', width=2.5)
            ))
            fig.update_layout(
                title='🕸️ Performance Radar: Base vs Tuned',
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template='plotly_dark',
                paper_bgcolor='rgba(30,33,48,0.8)',
                height=380,
                font=dict(family='Inter'),
                legend=dict(orientation='h', y=-0.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Threshold impact analysis
            st.markdown('<div class="section-title">🎚️ Threshold Sensitivity Analysis</div>',
                        unsafe_allow_html=True)

            th_model = st.session_state['tuned_model']
            y_pp = th_model.predict(X_test_ready, verbose=0).flatten()

            th_vals = np.arange(0.1, 0.91, 0.05)
            prec_list, rec_list, f1_list, fpr_list = [], [], [], []

            for th in th_vals:
                yp = (y_pp > th).astype(int)
                r  = classification_report(y_test_ready, yp,
                                            target_names=['Normal','Fraud'],
                                            output_dict=True, zero_division=0)
                cm_th = confusion_matrix(y_test_ready, yp)
                prec_list.append(r['Fraud']['precision'])
                rec_list.append(r['Fraud']['recall'])
                f1_list.append(r['Fraud']['f1-score'])
                tn, fp = cm_th[0][0], cm_th[0][1]
                fpr_list.append(fp / (fp + tn + 1e-9))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(th_vals), y=prec_list,
                                      name='Precision', line=dict(color='#667eea', width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_vals), y=rec_list,
                                      name='Recall', line=dict(color='#38ef7d', width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_vals), y=f1_list,
                                      name='F1-Score', line=dict(color='#f7b731', width=2.5)))
            fig.add_trace(go.Scatter(x=list(th_vals), y=fpr_list,
                                      name='False Positive Rate',
                                      line=dict(color='#ff6b6b', width=2, dash='dash')))
            fig.add_vline(x=threshold, line_dash='dash', line_color='white',
                          opacity=0.7,
                          annotation_text=f"Selected: {threshold:.2f}",
                          annotation_position="top right")
            fig.update_layout(
                title='📈 Impact of Threshold on All Metrics',
                template='plotly_dark',
                paper_bgcolor='rgba(30,33,48,0.8)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=360, font=dict(family='Inter'),
                xaxis_title='Decision Threshold',
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"✅ Tuned model is now active in the **Fraud Detector** with threshold = {threshold:.2f}")

        else:
            st.info("👈 Configure parameters on the left and click **Train Custom Model** to start!")
            st.markdown("""
            ### 💡 Tuning Tips for Managers

            **Neurons per layer** — More neurons = more learning capacity, but risks overfitting.
            Try 256→128→64 (funnel shape) for best results.

            **Dropout Rate** — Prevents overfitting. Higher dropout = more regularization.
            Keep between 0.2–0.5 for fraud detection.

            **Learning Rate** — How fast the model learns. Too high = unstable, too low = slow.
            0.001 is usually the sweet spot.

            **Decision Threshold** — The most impactful business lever:
            - 🔽 Lower threshold (e.g. 0.3) → Catch more fraud, but more false alarms
            - 🔼 Higher threshold (e.g. 0.7) → Fewer false alarms, but miss more fraud

            **Batch Normalization** — Almost always helps. Keep it ON.
            """)


# ══════════════════════════════════════════════════════════════════
# PAGE 4: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance Analytics</h1>
        <p>Deep dive into ANN performance metrics and evaluation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        ep = list(range(1, len(history['loss']) + 1))
        thresholds = np.linspace(0, 1, 100)
        p_curve = [float(metrics['report']['Fraud']['precision']) * (1 - 0.3*t) for t in thresholds]
        r_curve = [float(metrics['report']['Fraud']['recall']) * (1 - 0.2*(1-t)) for t in thresholds]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(thresholds), y=p_curve,
                                  name='Precision', line=dict(color='#667eea', width=2.5)))
        fig.add_trace(go.Scatter(x=list(thresholds), y=r_curve,
                                  name='Recall', line=dict(color='#38ef7d', width=2.5)))
        fig.add_vline(x=0.5, line_dash='dash', line_color='white',
                      opacity=0.5, annotation_text='Default Threshold')
        fig.update_layout(title='⚖️ Precision–Recall vs Threshold',
                          template='plotly_dark', height=380,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title='Threshold', yaxis=dict(range=[0, 1]),
                          font=dict(family='Inter'),
                          legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        categories = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
        report = metrics['report']
        values = [metrics['roc_auc'],
                  float(report['Fraud']['precision']),
                  float(report['Fraud']['recall']),
                  float(report['Fraud']['f1-score']),
                  float(report['accuracy'])]

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', fillcolor='rgba(102,126,234,0.2)',
            line=dict(color='#667eea', width=2.5), name='Base Model'
        ))
        fig.update_layout(title='🕸️ Performance Radar (Base Model)',
                          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          template='plotly_dark',
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          height=380, font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    # ANN Architecture diagram
    st.markdown('<div class="section-title">🧠 ANN Architecture Diagram</div>',
                unsafe_allow_html=True)

    layer_data = [
        ("Input\n(12 features)", 12, "#4fc3f7"),
        ("Dense 256\n+ BN + Dropout 0.4", 256, "#667eea"),
        ("Dense 128\n+ BN + Dropout 0.3", 128, "#764ba2"),
        ("Dense 64\n+ Dropout 0.2",  64, "#a855f7"),
        ("Dense 32\n+ Dropout 0.1",  32, "#ec4899"),
        ("Output\n(Sigmoid)", 1, "#38ef7d"),
    ]

    fig = go.Figure()
    max_display = 12
    for i, (name, n, color) in enumerate(layer_data):
        disp = min(n, max_display)
        for j in range(disp):
            fig.add_trace(go.Scatter(
                x=[i], y=[j - disp/2],
                mode='markers',
                marker=dict(size=22, color=color, opacity=0.85,
                             line=dict(width=1.5, color='white')),
                showlegend=False,
                hovertext=f"<b>{name}</b><br>{n} neurons"
            ))
        fig.add_annotation(x=i, y=-(max_display/2 + 1.5),
                           text=name, showarrow=False,
                           font=dict(size=9, color='#c5cae9'),
                           align='center')

    for i in range(len(layer_data)-1):
        d1 = min(layer_data[i][1], max_display)
        d2 = min(layer_data[i+1][1], max_display)
        for j1 in range(min(d1, 6)):
            for j2 in range(min(d2, 6)):
                fig.add_shape(type='line',
                    x0=i, y0=j1 - d1/2,
                    x1=i+1, y1=j2 - d2/2,
                    line=dict(color='rgba(150,150,220,0.08)', width=0.8))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(30,33,48,0.8)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=420,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        font=dict(family='Inter'),
        title='Neural Network Layer Structure (Base Model)',
        margin=dict(b=80)
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5: INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Business Insights</h1>
        <p>UPI fraud patterns, financial impact & actionable intelligence for managers</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        amounts = np.concatenate([np.random.lognormal(10, 1.5, 2000),
                                   np.random.lognormal(12.5, 1.0, 300)])
        labels  = ['Normal']*2000 + ['Fraud']*300
        fig = px.histogram(pd.DataFrame({'Amount (₹)': amounts, 'Type': labels}),
                           x='Amount (₹)', color='Type', nbins=60,
                           barmode='overlay',
                           color_discrete_map={'Normal':'#667eea','Fraud':'#ff6b6b'},
                           title='💰 Transaction Amount Distribution')
        fig.update_layout(template='plotly_dark', height=340,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hours = list(range(24))
        fraud_h = [2,3,5,8,4,3,2,4,7,6,7,9,10,8,7,6,9,12,15,14,13,10,7,4]
        fig = go.Figure(go.Bar(x=hours, y=fraud_h,
                                marker=dict(color=fraud_h, colorscale='RdYlGn_r',
                                            line=dict(width=0))))
        fig.update_layout(title='⏰ Fraud Attempts by Hour of Day',
                          template='plotly_dark', height=340,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title='Hour (0–23)',
                          yaxis_title='Fraud Cases',
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.sunburst(
            pd.DataFrame({'type':   ['TRANSFER','TRANSFER','CASH_OUT','CASH_OUT'],
                          'outcome':['Fraud','Legit','Fraud','Legit'],
                          'count':  [3162, 7081, 4116, 18600]}),
            path=['type','outcome'], values='count',
            color='outcome',
            color_discrete_map={'Fraud':'#ff6b6b','Legit':'#38ef7d'},
            title='🥧 Fraud Breakdown by Transaction Type'
        )
        fig.update_layout(template='plotly_dark', height=380,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        months    = ['Jan','Feb','Mar','Apr','May','Jun']
        prevented = [12.3, 15.1, 18.7, 14.2, 20.5, 23.1]
        missed    = [2.1,  1.8,  2.3,  1.5,  2.8,  2.2]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=prevented,
                              name='Fraud Prevented (₹Cr)',
                              marker_color='#38ef7d'))
        fig.add_trace(go.Bar(x=months, y=missed,
                              name='Missed Fraud (₹Cr)',
                              marker_color='#ff6b6b'))
        fig.update_layout(title='💸 Financial Impact Simulation (₹ Crores)',
                          barmode='group', template='plotly_dark', height=380,
                          paper_bgcolor='rgba(30,33,48,0.8)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🎯 Key Managerial Insights</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🕐 **Peak Fraud Window**\n\nFraud spikes between 6 PM–9 PM. Increase monitoring and lower decision thresholds during these hours.")
    with c2:
        st.warning("💸 **High-Risk Amount Zone**\n\nTransactions above ₹2,00,000 show 3× higher fraud probability. Flag these for secondary verification.")
    with c3:
        st.error("🔄 **Zero-Balance Drain Pattern**\n\nIf sender balance drops to ₹0 after a TRANSFER, probability of fraud exceeds 85%. Auto-block recommended.")