# 🛡️ UPI Fraud Detection System

> **AI-powered real-time fraud detection for UPI transactions using a Deep Learning ANN model — built with TensorFlow, Streamlit, and PaySim synthetic data.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://upi-fraud-4ec9qljxsf4flodz62gqnt.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project was developed as part of the **"Deep Learning for Managers"** course. It demonstrates how Artificial Neural Networks (ANN) can be applied to detect fraudulent UPI (Unified Payments Interface) transactions in real time.

With over **10 billion** monthly UPI transactions in India, fraud detection is a critical business problem. This system provides:
- A trained ANN model achieving **ROC-AUC > 0.99**
- An interactive Streamlit dashboard for live transaction analysis
- Hyperparameter tuning with real-time training visualization
- Business insights and financial impact simulation

---

## 🚀 Live Demo

👉 **[Launch the App](https://upi-fraud-4ec9qljxsf4flodz62gqnt.streamlit.app/)**

---

## 🗂️ Project Structure

```
upi-fraud/
│
├── app.py                  # Main Streamlit application (5 pages)
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (libgomp1)
│
├── ann_model.h5            # Trained ANN model (TensorFlow/Keras)
├── scaler.pkl              # StandardScaler fitted on training data
├── features.json           # Feature names list
├── metrics.json            # Model evaluation metrics
├── training_history.json   # Epoch-wise loss/AUC history
│
├── train_data.npz          # Compressed training data (SMOTE balanced)
└── test_data.npz           # Compressed test data (stratified sample)
```

---

## 🧠 Model Architecture

```
Input Layer     →  12 Features
Dense Layer 1   →  256 neurons + BatchNorm + Dropout(0.4) + ReLU
Dense Layer 2   →  128 neurons + BatchNorm + Dropout(0.3) + ReLU
Dense Layer 3   →  64  neurons + Dropout(0.2) + ReLU
Dense Layer 4   →  32  neurons + Dropout(0.1) + ReLU
Output Layer    →  1 neuron (Sigmoid)
```

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9983** |
| Precision | 0.3680 |
| Recall | **0.9927** |
| F1-Score | 0.5370 |

> High recall is prioritized — catching fraud matters more than false alarms in financial systems.

---

## 📊 Dataset

**PaySim Synthetic Mobile Money Dataset** — [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

- 6.3 million transactions simulated from real mobile money logs
- Binary classification: `isFraud` (0 = Normal, 1 = Fraud)
- Class imbalance: 99.87% normal, 0.13% fraud
- Fraud occurs **only** in `TRANSFER` and `CASH_OUT` transaction types

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender balance before |
| `newbalanceOrig` | Sender balance after |
| `oldbalanceDest` | Receiver balance before |
| `newbalanceDest` | Receiver balance after |
| `errorBalanceOrig` | Balance discrepancy (sender) |
| `errorBalanceDest` | Balance discrepancy (receiver) |
| `balanceDropRatio` | Fraction of sender balance transferred |
| `isLargeTransaction` | Amount > ₹2,00,000 |
| `zeroBalanceAfter` | Sender balance hits ₹0 after transfer |
| `destBalanceIncrease` | Net increase in receiver balance |
| `type_encoded` | TRANSFER=1, CASH_OUT=0 |

---

## ⚖️ Handling Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied **only to training data** to create a balanced 50/50 split. The test set was kept untouched to reflect real-world distribution.

---

## 🖥️ App Pages

### 🏠 Dashboard
- 6 KPI cards (ROC-AUC, Precision, Recall, F1, Fraud Caught/Missed)
- Training loss & AUC curves
- Confusion matrix heatmap
- Fraud rate by transaction type

### 🔍 Fraud Detector
- Input form: transaction type, amount, sender/receiver balances
- Real-time fraud probability with animated gauge chart
- Risk factor breakdown bar chart
- 4 quick test scenarios (balance drain, large transfer, normal payment, small UPI)

### ⚙️ Hypertune Model
- Customize: neurons per layer, dropout, learning rate, batch size, threshold
- Live epoch-by-epoch training chart
- Early stopping with patience=8
- Side-by-side comparison: base vs tuned (metrics, confusion matrices, radar chart)
- Threshold sensitivity analysis

### 📊 Model Analytics
- Precision-Recall vs Threshold curve
- Performance radar chart
- ANN architecture node diagram

### 📈 Business Insights
- Transaction amount distribution (Normal vs Fraud)
- Fraud attempts by hour of day
- Sunburst chart: fraud by transaction type
- Financial impact simulation (₹ Crores prevented vs missed)
- 3 key managerial takeaways

---

## 🔑 Key Business Insights

| Finding | Detail |
|---------|--------|
| 🕐 **Peak Fraud Window** | 6 PM – 9 PM shows highest fraud attempts |
| 💸 **High-Risk Amount** | Transactions > ₹2,00,000 have 3× fraud probability |
| 🔄 **Zero-Balance Drain** | Sender balance → ₹0 after TRANSFER = 85%+ fraud probability |
| 🔁 **Riskiest Type** | TRANSFER (31% fraud rate) vs CASH_OUT (18%) |

---

## ⚙️ Installation & Local Run

```bash
# Clone the repository
git clone https://github.com/parvjhamb123-sketch/upi-fraud.git
cd upi-fraud

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit==1.43.0
tensorflow-cpu==2.20.0
scikit-learn==1.6.1
imbalanced-learn==0.13.0
pandas==2.2.3
numpy==2.2.0
plotly==5.24.1
joblib==1.4.2
```

---

## 🔮 Decision Threshold — The Key Business Lever

The default threshold is **0.5**, but this is a business decision:

| Threshold | Effect |
|-----------|--------|
| 🔽 Lower (0.3) | Catch more fraud — but more false alarms for legit customers |
| 🔼 Higher (0.7) | Fewer false alarms — but some fraud slips through |

The Hypertune page lets you explore this trade-off interactively.

---

## 👨‍💻 Author

**Parv Jhamb**
- Course: Deep Learning for Managers
- Dataset: [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Deployed on: [Streamlit Cloud](https://streamlit.io/cloud)

---

## 📄 License

This project is for educational purposes. Dataset is synthetic and does not contain real financial data.
