# Predictive Maintenance & Anomaly Detection in 6G-Integrated Smart Manufacturing

An end-to-end machine learning project for monitoring smart
manufacturing systems, detecting abnormal machine behavior, predicting
operational efficiency, and estimating maintenance risk through an
interactive Streamlit dashboard.

## 📌 Project Overview

Modern smart manufacturing environments generate continuous machine,
production, sensor, quality, and network data. This project combines
**Isolation Forest**, **Random Forest**, rolling statistical features, a
maintenance risk engine, **SHAP explainability**, and **Streamlit** to
support machine-health monitoring.

The project is designed around a **6G-integrated smart manufacturing
context**, using indicators such as temperature, vibration, power
consumption, network latency, packet loss, production speed, and error
rate.

## 🎯 Objectives

1.  Understand and preprocess manufacturing data.
2.  Detect abnormal machine operating conditions.
3.  Predict machine efficiency status.
4.  Combine anomaly, efficiency, sensor, and network indicators into a
    maintenance risk score.
5.  Provide interpretable machine-level insights.
6.  Build an interactive monitoring dashboard.

## 🏭 Key Features

### Anomaly Detection

An **Isolation Forest** model identifies unusual machine operating
patterns using sensor, production, quality, and network features. A
normalized **0--100 Anomaly Score** is produced for dashboard
monitoring.

### Efficiency Classification

A **Random Forest Classifier** predicts machine efficiency status. The
workflow includes train/test evaluation, accuracy, precision, recall,
F1-score, confusion matrix, training/testing comparison, baseline
comparison, and feature importance.

### Maintenance Risk Engine

The final maintenance score combines:

  Component           Weight
  ----------------- --------
  Anomaly Score          50%
  Efficiency Risk        25%
  Sensor Risk            15%
  Network Risk           10%

Risk priorities:

       Score Priority
  ---------- -----------
      `< 25` Normal
     `25–49` Monitor
     `50–74` Warning
     `75–89` High Risk
    `90–100` Critical

### Explainable AI

**SHAP (SHapley Additive exPlanations)** is used to understand which
features contribute to model predictions and machine-level risk
analysis.

### Streamlit Dashboard

The dashboard provides machine selection, operation-mode filtering,
system overview, anomaly monitoring, machine risk analysis,
recommendations, alerts, interactive charts, and explainable-AI
insights.

## 🧠 Machine Learning Architecture

``` text
Manufacturing Dataset
        ↓
Data Cleaning & Preparation
        ↓
Rolling Feature Engineering
        ↓
 ┌───────────────┬─────────────────┐
 ↓               ↓
Isolation       Random Forest
Forest           Classification
 ↓               ↓
Anomaly Score   Efficiency Risk
 └───────────────┬─────────────────┘
                 ↓
       Sensor & Network Risk
                 ↓
      Maintenance Risk Engine
                 ↓
       Risk Priority & Alerts
                 ↓
       Streamlit Dashboard
                 ↓
       SHAP Explainability
```

## 🛠️ Technology Stack

  Technology         Purpose
  ------------------ ---------------------------
  Python             Core programming
  Pandas             Data processing
  NumPy              Numerical operations
  Scikit-learn       Machine learning
  Isolation Forest   Anomaly detection
  Random Forest      Efficiency classification
  Joblib             Model serialization
  SHAP               Explainable AI
  Plotly             Interactive visualization
  Matplotlib         Visualization
  Seaborn            Exploratory visualization
  Streamlit          Dashboard
  Git                Version control
  GitHub             Repository hosting

## 📂 Project Structure

``` text
Predictive-Maintenance-6G/
│
├── app/
│   ├── app.py
│   ├── pipeline.py
│   └── style.css
│
├── data/
│   └── Thales_Group_Manufacturing UNIFIED MENTOR.csv
│
├── models/
│   ├── anomaly_features.pkl
│   ├── anomaly_scaler.pkl
│   ├── anomaly_score_scaler.pkl
│   ├── classification_features.pkl
│   ├── isolation_forest.pkl
│   └── random_forest.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

### Clone the repository

``` bash
git clone https://github.com/SamsulHoque2525/Predictive-Maintenance-6G.git
cd Predictive-Maintenance-6G
```

### Create a virtual environment

Windows:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

``` powershell
pip install -r requirements.txt
```

### Run the dashboard

``` powershell
streamlit run app/app.py
```

## 📊 Model Evaluation

The supervised model is evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   Training vs testing accuracy
-   Baseline comparison
-   Feature importance

Exact metric values should be taken from the final notebook output
rather than estimated.

## 🔎 Explainable AI

SHAP helps investigate:

-   Which features contributed to a prediction
-   Which sensor conditions influence efficiency
-   Why a machine received a particular prediction
-   Which operational signals deserve attention

## 🚨 Maintenance Risk Workflow

``` text
Anomaly Risk
     +
Efficiency Risk
     +
Sensor Risk
     +
Network Risk
     ↓
Maintenance Risk Score
     ↓
Risk Priority
     ↓
Maintenance Recommendation
```

## 📈 Dashboard Screenshots

Recommended screenshots can be stored in:

``` text
images/
├── dashboard-overview.png
├── anomaly-detection.png
├── maintenance-risk.png
└── shap-explainability.png
```

## 🚀 Future Scope

-   Real-time IoT sensor streaming
-   Live 5G/6G network telemetry integration
-   Automated maintenance scheduling
-   Remaining Useful Life (RUL) prediction
-   LSTM/Transformer-based predictive maintenance
-   Cloud deployment
-   Edge-AI inference
-   Automated alerts
-   Industrial digital-twin integration

## 🎓 Skills Demonstrated

-   Machine Learning
-   Unsupervised Learning
-   Supervised Classification
-   Feature Engineering
-   Anomaly Detection
-   Predictive Maintenance
-   Explainable AI
-   Data Visualization
-   Streamlit Development
-   Model Deployment
-   Git/GitHub

## 👨‍💻 Author

**Samsul Hoque**

GitHub: https://github.com/SamsulHoque2525

## 📜 License

No license has been added to the repository at this stage.
