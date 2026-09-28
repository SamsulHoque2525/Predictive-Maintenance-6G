
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance 6G",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CUSTOM CSS
# =========================================================

with open("app/style.css", "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )



DATA_PATH = "data/Thales_Group_Manufacturing UNIFIED MENTOR.csv"

df = pd.read_csv(DATA_PATH)

# Load trained models
isolation_forest = joblib.load(
    "models/isolation_forest.pkl"
)

anomaly_scaler = joblib.load(
    "models/anomaly_scaler.pkl"
)
anomaly_score_scaler = joblib.load(
    "models/anomaly_score_scaler.pkl"
)

random_forest = joblib.load(
    "models/random_forest.pkl"
)

classification_features = joblib.load(
    "models/classification_features.pkl"
)

anomaly_features = joblib.load(
    "models/anomaly_features.pkl"
)

# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <h1 style="
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    ">
        Predictive Maintenance
    </h1>

    <h3 style="
        font-weight: 500;
        margin-top: 0;
    ">
        6G-Integrated Smart Manufacturing Intelligence
    </h3>

    <p style="
        font-size: 16px;
        opacity: 0.75;
    ">
        Real-time anomaly detection, machine health monitoring,
        maintenance risk prediction and explainable AI.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.sidebar.title("⚙️ Dashboard Controls")

machine_list = sorted(
    df["Machine_ID"].unique()
)

selected_machine = st.sidebar.selectbox(
    "Select Machine",
    ["All Machines"] + machine_list
)

operation_modes = sorted(
    df["Operation_Mode"].dropna().unique()
)

selected_mode = st.sidebar.multiselect(
    "Operation Mode",
    operation_modes,
    default=operation_modes
)

filtered_df = df.copy()

if selected_machine != "All Machines":
    filtered_df = filtered_df[
        filtered_df["Machine_ID"] == selected_machine
    ]

if selected_mode:
    filtered_df = filtered_df[
        filtered_df["Operation_Mode"].isin(
            selected_mode
        )
    ]
    
    # =========================================================
# SIDEBAR SYSTEM STATUS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 📡 System Status"
)

st.sidebar.success(
    "● ML Models Online"
)

st.sidebar.info(
    f"🏭 {df['Machine_ID'].nunique()} Machines"
)

st.sidebar.info(
    f"📊 {len(df):,} Records"
)

st.sidebar.info(
    "🤖 Isolation Forest + Random Forest"
)
    
    
    
st.header("📊 System Overview")
total_records = len(filtered_df)

total_machines = filtered_df["Machine_ID"].nunique()

active_records = (
    filtered_df["Operation_Mode"] == "Active"
).sum()

maintenance_records = (
    filtered_df["Operation_Mode"] == "Maintenance"
).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{total_records:,}"
    )

with col2:
    st.metric(
        "Machines",
        total_machines
    )

with col3:
    st.metric(
        "Active Records",
        f"{active_records:,}"
    )

with col4:
    st.metric(
        "Maintenance Records",
        f"{maintenance_records:,}"
    )
    
    st.subheader("Machine Summary")

machine_summary = (
    filtered_df
    .groupby("Machine_ID")
    .agg(
        Records=("Machine_ID", "size"),
        Avg_Temperature=("Temperature_C", "mean"),
        Avg_Vibration=("Vibration_Hz", "mean"),
        Avg_Power=("Power_Consumption_kW", "mean"),
        Avg_Error_Rate=("Error_Rate_%", "mean")
    )
    .round(2)
)

st.dataframe(
    machine_summary,
    use_container_width=True
)

st.subheader("🌡️ Temperature by Machine")

temperature_chart = px.bar(
    machine_summary.reset_index(),
    x="Machine_ID",
    y="Avg_Temperature",
    title="Average Temperature by Machine"
)

st.plotly_chart(
    temperature_chart,
    use_container_width=True
)

st.subheader("⚙️ Vibration by Machine")

vibration_chart = px.bar(
    machine_summary.reset_index(),
    x="Machine_ID",
    y="Avg_Vibration",
    title="Average Vibration by Machine"
)

st.plotly_chart(
    vibration_chart,
    use_container_width=True
)

st.subheader("📈 Efficiency Status")

efficiency_counts = (
    filtered_df["Efficiency_Status"]
    .value_counts()
    .reset_index()
)

efficiency_counts.columns = [
    "Efficiency_Status",
    "Count"
]

efficiency_chart = px.pie(
    efficiency_counts,
    names="Efficiency_Status",
    values="Count",
    title="Efficiency Status Distribution"
)

st.plotly_chart(
    efficiency_chart,
    use_container_width=True
)

# =========================================================
# ANOMALY DETECTION
# =========================================================

st.header("🚨 Anomaly Detection")

st.write(
    "The Isolation Forest model identifies unusual "
    "machine operating conditions based on sensor, "
    "production and network parameters."
)


# =========================================================
# PREPARE DATA FOR ANOMALY DETECTION
# =========================================================

anomaly_df = df.copy()


# ---------------------------------------------------------
# CREATE DATETIME
# ---------------------------------------------------------

anomaly_df["Date"] = pd.to_datetime(
    anomaly_df["Date"],
    errors="coerce"
)

anomaly_df["Timestamp"] = pd.to_datetime(
    anomaly_df["Timestamp"],
    errors="coerce"
)


anomaly_df["Datetime"] = pd.to_datetime(
    anomaly_df["Date"].dt.strftime("%Y-%m-%d")
    + " "
    + anomaly_df["Timestamp"].dt.strftime("%H:%M:%S"),
    errors="coerce"
)


# ---------------------------------------------------------
# SORT DATA
# ---------------------------------------------------------

anomaly_df = anomaly_df.sort_values(
    by=["Machine_ID", "Datetime"]
).reset_index(drop=True)


# =========================================================
# CREATE SAME ROLLING FEATURES USED DURING TRAINING
# =========================================================

WINDOW = 5


anomaly_df["Temperature_Rolling_Mean"] = (
    anomaly_df
    .groupby("Machine_ID")["Temperature_C"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=1
        ).mean()
    )
)


anomaly_df["Vibration_Rolling_Mean"] = (
    anomaly_df
    .groupby("Machine_ID")["Vibration_Hz"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=1
        ).mean()
    )
)


anomaly_df["Power_Rolling_Mean"] = (
    anomaly_df
    .groupby("Machine_ID")["Power_Consumption_kW"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=1
        ).mean()
    )
)


anomaly_df["Error_Rate_Rolling_Mean"] = (
    anomaly_df
    .groupby("Machine_ID")["Error_Rate_%"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=1
        ).mean()
    )
)


anomaly_df["Production_Speed_Rolling_Mean"] = (
    anomaly_df
    .groupby("Machine_ID")[
        "Production_Speed_units_per_hr"
    ]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=1
        ).mean()
    )
)


anomaly_df["Temperature_Rolling_Std"] = (
    anomaly_df
    .groupby("Machine_ID")["Temperature_C"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=2
        ).std()
    )
)


anomaly_df["Vibration_Rolling_Std"] = (
    anomaly_df
    .groupby("Machine_ID")["Vibration_Hz"]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=2
        ).std()
    )
)


anomaly_df["Power_Rolling_Std"] = (
    anomaly_df
    .groupby("Machine_ID")[
        "Power_Consumption_kW"
    ]
    .transform(
        lambda x:
        x.rolling(
            WINDOW,
            min_periods=2
        ).std()
    )
)

# =========================================================
# SELECT ANOMALY FEATURES
# =========================================================

X_anomaly_app = anomaly_df[
    anomaly_features
].copy()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

X_anomaly_app = X_anomaly_app.fillna(
    X_anomaly_app.median()
)


# =========================================================
# SCALE FEATURES
# =========================================================

X_anomaly_scaled_app = (
    anomaly_scaler.transform(
        X_anomaly_app
    )
)


# =========================================================
# PREDICT ANOMALIES
# =========================================================

anomaly_predictions_app = (
    isolation_forest.predict(
        X_anomaly_scaled_app
    )
)


anomaly_df["Anomaly_Label"] = (
    anomaly_predictions_app
)


anomaly_df["Anomaly_Status"] = (
    anomaly_df["Anomaly_Label"]
    .map({
        1: "Normal",
        -1: "Anomaly"
    })
)


# =========================================================
# ANOMALY SCORE
# =========================================================

raw_anomaly_score = (
    -isolation_forest.decision_function(
        X_anomaly_scaled_app
    )
)

anomaly_df["Anomaly_Score"] = (
    anomaly_score_scaler.transform(
        raw_anomaly_score.reshape(-1, 1)
    ).flatten()
)

anomaly_df["Anomaly_Score"] = (
    anomaly_df["Anomaly_Score"]
    .round(2)
)
# =========================================================
# ANOMALY SUMMARY
# =========================================================

st.header("🚨 Anomaly Intelligence")

st.write(
    "Machine-level anomaly monitoring based on sensor, "
    "production and network behavior."
)


# =========================================================
# ANOMALY STATISTICS
# =========================================================

total_anomalies = (
    anomaly_df["Anomaly_Status"] == "Anomaly"
).sum()

total_normal = (
    anomaly_df["Anomaly_Status"] == "Normal"
).sum()

anomaly_rate = (
    total_anomalies
    / len(anomaly_df)
    * 100
)


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Observations",
        f"{len(anomaly_df):,}"
    )


with col2:

    st.metric(
        "Normal Observations",
        f"{total_normal:,}"
    )


with col3:

    st.metric(
        "Detected Anomalies",
        f"{total_anomalies:,}"
    )


with col4:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


# =========================================================
# NORMAL VS ANOMALY
# =========================================================

st.subheader("📊 Normal vs Anomalous Observations")


anomaly_counts = (
    anomaly_df["Anomaly_Status"]
    .value_counts()
    .reset_index()
)

anomaly_counts.columns = [
    "Status",
    "Count"
]


anomaly_chart = px.pie(
    anomaly_counts,
    names="Status",
    values="Count",
    hole=0.60
)


anomaly_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="center",
        x=0.5
    )
)


st.plotly_chart(
    anomaly_chart,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# MACHINE-WISE ANOMALY ANALYSIS
# =========================================================

st.subheader("🏭 Machine-wise Anomaly Analysis")


machine_anomaly = (
    anomaly_df
    .groupby("Machine_ID")
    .agg(
        Total_Observations=(
            "Machine_ID",
            "size"
        ),

        Anomalies=(
            "Anomaly_Label",
            lambda x: (x == -1).sum()
        ),

        Average_Anomaly_Score=(
            "Anomaly_Score",
            "mean"
        )
    )
)


machine_anomaly["Anomaly_Rate_%"] = (
    machine_anomaly["Anomalies"]
    / machine_anomaly["Total_Observations"]
    * 100
)


machine_anomaly = (
    machine_anomaly
    .sort_values(
        "Anomaly_Rate_%",
        ascending=False
    )
    .round(2)
)


# =========================================================
# MACHINE ANOMALY CHART
# =========================================================

machine_anomaly_chart = px.bar(
    machine_anomaly.reset_index(),
    x="Machine_ID",
    y="Anomaly_Rate_%",
    title="Anomaly Rate by Machine"
)


machine_anomaly_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#CBD5E1"
    ),
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    ),
    xaxis_title="Machine ID",
    yaxis_title="Anomaly Rate (%)"
)


st.plotly_chart(
    machine_anomaly_chart,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# MACHINE ANOMALY TABLE
# =========================================================

st.subheader("🔎 Machine Anomaly Summary")


display_machine_anomaly = machine_anomaly.reset_index()


display_machine_anomaly.columns = [
    "Machine ID",
    "Total Observations",
    "Anomalies",
    "Average Anomaly Score",
    "Anomaly Rate (%)"
]


st.dataframe(
    display_machine_anomaly,
    use_container_width=True,
    height=350
)


# =========================================================
# ANOMALOUS RECORDS
# =========================================================

st.subheader("⚠️ Detected Anomalous Records")


anomalous_records = anomaly_df[
    anomaly_df["Anomaly_Status"] == "Anomaly"
].copy()


anomalous_records = anomalous_records.sort_values(
    "Anomaly_Score",
    ascending=False
)


display_columns = [
    "Machine_ID",
    "Datetime",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Efficiency_Status",
    "Anomaly_Score",
    "Anomaly_Status"
]


st.dataframe(
    anomalous_records[display_columns],
    use_container_width=True,
    height=400
)

# =========================================================
# MAINTENANCE RISK ENGINE
# =========================================================

st.header("🛠️ Maintenance Risk Intelligence")

st.write(
    "The maintenance risk engine combines anomaly behavior, "
    "predicted efficiency, sensor conditions and network "
    "conditions to estimate the maintenance priority of "
    "each observation."
)


# =========================================================
# CREATE MAINTENANCE DATAFRAME
# =========================================================

maintenance_df = anomaly_df.copy()

X_classification_app = maintenance_df[classification_features].copy()
X_classification_app = X_classification_app.fillna(
    X_classification_app.median()
)

predicted_efficiency = random_forest.predict(X_classification_app)

maintenance_df["Predicted_Efficiency"] = predicted_efficiency

prediction_probabilities = random_forest.predict_proba(
    X_classification_app
)

maintenance_df["Prediction_Confidence"] = (
    prediction_probabilities.max(axis=1) * 100
).round(2)

# =========================================================
# 1. EFFICIENCY RISK
# =========================================================

efficiency_risk_map = {
    "High": 0,
    "Medium": 50,
    "Low": 100
}


maintenance_df["Efficiency_Risk_Score"] = (
    maintenance_df[
        "Predicted_Efficiency"
    ].map(efficiency_risk_map)
)


# =========================================================
# 2. SENSOR RISK THRESHOLDS
# =========================================================

temperature_threshold = (
    maintenance_df["Temperature_C"]
    .quantile(0.95)
)

vibration_threshold = (
    maintenance_df["Vibration_Hz"]
    .quantile(0.95)
)

error_threshold = (
    maintenance_df["Error_Rate_%"]
    .quantile(0.95)
)

latency_threshold = (
    maintenance_df["Network_Latency_ms"]
    .quantile(0.95)
)

packet_loss_threshold = (
    maintenance_df["Packet_Loss_%"]
    .quantile(0.95)
)


# =========================================================
# 3. SENSOR RISK FLAGS
# =========================================================

maintenance_df["Temperature_Risk"] = (
    maintenance_df["Temperature_C"]
    >= temperature_threshold
).astype(int)


maintenance_df["Vibration_Risk"] = (
    maintenance_df["Vibration_Hz"]
    >= vibration_threshold
).astype(int)


maintenance_df["Error_Risk"] = (
    maintenance_df["Error_Rate_%"]
    >= error_threshold
).astype(int)


maintenance_df["Latency_Risk"] = (
    maintenance_df["Network_Latency_ms"]
    >= latency_threshold
).astype(int)


maintenance_df["Packet_Loss_Risk"] = (
    maintenance_df["Packet_Loss_%"]
    >= packet_loss_threshold
).astype(int)


# =========================================================
# 4. SENSOR RISK SCORE
# =========================================================

sensor_risk = (
    maintenance_df["Temperature_Risk"]
    +
    maintenance_df["Vibration_Risk"]
    +
    maintenance_df["Error_Risk"]
) / 3 * 100


# =========================================================
# 5. NETWORK RISK SCORE
# =========================================================

network_risk = (
    maintenance_df["Latency_Risk"]
    +
    maintenance_df["Packet_Loss_Risk"]
) / 2 * 100


# =========================================================
# 6. MAINTENANCE RISK SCORE
# =========================================================

maintenance_df["Maintenance_Risk_Score"] = (
    0.50 * maintenance_df["Anomaly_Score"]
    +
    0.25 * maintenance_df[
        "Efficiency_Risk_Score"
    ]
    +
    0.15 * sensor_risk
    +
    0.10 * network_risk
)


maintenance_df[
    "Maintenance_Risk_Score"
] = (
    maintenance_df[
        "Maintenance_Risk_Score"
    ]
    .clip(0, 100)
    .round(2)
)


# =========================================================
# 7. MAINTENANCE PRIORITY
# =========================================================

def maintenance_priority(score):

    if score < 25:
        return "Normal"

    elif score < 50:
        return "Monitor"

    elif score < 75:
        return "Warning"

    elif score < 90:
        return "High Risk"

    else:
        return "Critical"


maintenance_df[
    "Maintenance_Priority"
] = (
    maintenance_df[
        "Maintenance_Risk_Score"
    ].apply(
        maintenance_priority
    )
)


# =========================================================
# 8. MAINTENANCE RECOMMENDATION
# =========================================================

def maintenance_recommendation(row):

    if row["Maintenance_Priority"] == "Critical":

        return (
            "Immediate inspection recommended"
        )

    elif row["Maintenance_Priority"] == "High Risk":

        return (
            "Schedule maintenance inspection soon"
        )

    elif row["Maintenance_Priority"] == "Warning":

        return (
            "Monitor machine condition closely"
        )

    elif row["Maintenance_Priority"] == "Monitor":

        return (
            "Continue monitoring sensor trends"
        )

    else:

        return (
            "Normal operation"
        )


maintenance_df[
    "Maintenance_Recommendation"
] = (
    maintenance_df.apply(
        maintenance_recommendation,
        axis=1
    )
)


# =========================================================
# RISK SUMMARY
# =========================================================

priority_counts = (
    maintenance_df[
        "Maintenance_Priority"
    ]
    .value_counts()
    .reindex(
        [
            "Normal",
            "Monitor",
            "Warning",
            "High Risk",
            "Critical"
        ],
        fill_value=0
    )
)


# =========================================================
# RISK KPI CARDS
# =========================================================

normal_count = priority_counts[
    "Normal"
]

monitor_count = priority_counts[
    "Monitor"
]

warning_count = priority_counts[
    "Warning"
]

high_risk_count = priority_counts[
    "High Risk"
]

critical_count = priority_counts[
    "Critical"
]


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Normal",
        f"{normal_count:,}"
    )


with col2:

    st.metric(
        "Monitor",
        f"{monitor_count:,}"
    )


with col3:

    st.metric(
        "Warning",
        f"{warning_count:,}"
    )


with col4:

    st.metric(
        "High Risk",
        f"{high_risk_count:,}"
    )


with col5:

    st.metric(
        "Critical",
        f"{critical_count:,}"
    )


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.subheader(
    "📊 Maintenance Priority Distribution"
)


risk_chart_df = (
    priority_counts
    .reset_index()
)

risk_chart_df.columns = [
    "Priority",
    "Count"
]


risk_chart = px.bar(
    risk_chart_df,
    x="Priority",
    y="Count",
    title="Maintenance Risk Distribution"
)


risk_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#CBD5E1"
    ),
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    ),
    xaxis_title="Maintenance Priority",
    yaxis_title="Number of Observations"
)


st.plotly_chart(
    risk_chart,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# RISK SCORE DISTRIBUTION
# =========================================================

st.subheader(
    "📈 Maintenance Risk Score"
)


risk_score_chart = px.histogram(
    maintenance_df,
    x="Maintenance_Risk_Score",
    nbins=30,
    title="Maintenance Risk Score Distribution"
)


risk_score_chart.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#CBD5E1"
    ),
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    ),
    xaxis_title="Risk Score",
    yaxis_title="Observations"
)


st.plotly_chart(
    risk_score_chart,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ============================================================
# MACHINE RISK MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">Machine Risk Monitoring</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Identify machines requiring the most attention based on their maintenance risk.'
    '</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Machine-level risk aggregation
# ------------------------------------------------------------

machine_risk = (
    maintenance_df
    .groupby("Machine_ID")
    .agg(
        Maintenance_Risk=("Maintenance_Risk_Score", "mean"),
        Maximum_Risk=("Maintenance_Risk_Score", "max"),
        Average_Anomaly=("Anomaly_Score", "mean"),
        Anomaly_Count=(
            "Anomaly_Status",
            lambda x: (x == "Anomaly").sum()
        ),
        Total_Readings=("Machine_ID", "count")
    )
    .reset_index()
)

# Percentage of anomalous readings
machine_risk["Anomaly_Rate"] = (
    machine_risk["Anomaly_Count"]
    / machine_risk["Total_Readings"]
    * 100
).round(2)

# Sort highest-risk machines first
machine_risk = machine_risk.sort_values(
    "Maintenance_Risk",
    ascending=False
).reset_index(drop=True)

# ------------------------------------------------------------
# Risk category
# ------------------------------------------------------------

def machine_risk_category(score):

    if score < 25:
        return "Normal"

    elif score < 50:
        return "Monitor"

    elif score < 75:
        return "Warning"

    elif score < 90:
        return "High Risk"

    else:
        return "Critical"


machine_risk["Risk_Level"] = (
    machine_risk["Maintenance_Risk"]
    .apply(machine_risk_category)
)

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------

total_machines = len(machine_risk)

high_risk_machines = (
    machine_risk["Risk_Level"]
    .isin(["High Risk", "Critical"])
    .sum()
)

critical_machines = (
    machine_risk["Risk_Level"] == "Critical"
).sum()

average_machine_risk = (
    machine_risk["Maintenance_Risk"].mean()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Machines",
        total_machines
    )

with col2:
    st.metric(
        "High/Critical",
        high_risk_machines
    )

with col3:
    st.metric(
        "Critical Machines",
        critical_machines
    )

with col4:
    st.metric(
        "Avg Machine Risk",
        f"{average_machine_risk:.1f}"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Top 10 highest-risk machines
# ------------------------------------------------------------

st.markdown("### Top 10 Highest-Risk Machines")

top_risk = machine_risk.head(10).copy()

fig_machine_risk = px.bar(
    top_risk,
    x="Maintenance_Risk",
    y="Machine_ID",
    orientation="h",
    text="Maintenance_Risk",
    title="Machines Requiring Attention",
    labels={
        "Maintenance_Risk": "Maintenance Risk Score",
        "Machine_ID": "Machine"
    }
)

fig_machine_risk.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_machine_risk.update_layout(
    height=450,
    yaxis=dict(categoryorder="total ascending"),
    xaxis=dict(range=[0, 100]),
    margin=dict(l=20, r=40, t=60, b=20)
)

st.plotly_chart(
    fig_machine_risk,
    use_container_width=True
)

# ------------------------------------------------------------
# Machine risk table
# ------------------------------------------------------------

st.markdown("### Machine Risk Summary")

display_machine_risk = machine_risk[
    [
        "Machine_ID",
        "Maintenance_Risk",
        "Risk_Level",
        "Average_Anomaly",
        "Anomaly_Count",
        "Anomaly_Rate",
        "Total_Readings"
    ]
].copy()

display_machine_risk.columns = [
    "Machine",
    "Maintenance Risk",
    "Risk Level",
    "Avg Anomaly Score",
    "Anomaly Count",
    "Anomaly Rate (%)",
    "Readings"
]

st.dataframe(
    display_machine_risk,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# MACHINE RISK MONITORING
# =========================================================

st.subheader(
    "🏭 Machine Risk Monitoring"
)

st.write(
    "Machine-level risk analysis based on maintenance risk, "
    "anomaly behavior, and abnormal readings."
)

# ---------------------------------------------------------
# Calculate machine-level risk
# ---------------------------------------------------------

machine_risk = (
    maintenance_df
    .groupby("Machine_ID")
    .agg(
        Maintenance_Risk_Score=(
            "Maintenance_Risk_Score",
            "mean"
        ),
        Maximum_Risk=(
            "Maintenance_Risk_Score",
            "max"
        ),
        Average_Anomaly_Score=(
            "Anomaly_Score",
            "mean"
        ),
        Anomaly_Count=(
            "Anomaly_Status",
            lambda x: (x == "Anomaly").sum()
        ),
        Total_Readings=(
            "Machine_ID",
            "count"
        )
    )
    .reset_index()
)

# ---------------------------------------------------------
# Anomaly rate
# ---------------------------------------------------------

machine_risk["Anomaly_Rate"] = (
    machine_risk["Anomaly_Count"]
    / machine_risk["Total_Readings"]
    * 100
).round(2)

# ---------------------------------------------------------
# Risk category
# ---------------------------------------------------------

def get_machine_risk_level(score):

    if score < 25:
        return "Normal"

    elif score < 50:
        return "Monitor"

    elif score < 75:
        return "Warning"

    elif score < 90:
        return "High Risk"

    else:
        return "Critical"


machine_risk["Risk_Level"] = (
    machine_risk["Maintenance_Risk_Score"]
    .apply(get_machine_risk_level)
)

# ---------------------------------------------------------
# Sort highest-risk machines first
# ---------------------------------------------------------

machine_risk = machine_risk.sort_values(
    "Maintenance_Risk_Score",
    ascending=False
).reset_index(drop=True)

# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------

total_machines = len(machine_risk)

high_risk_count = machine_risk[
    machine_risk["Risk_Level"].isin(
        ["High Risk", "Critical"]
    )
].shape[0]

critical_count = machine_risk[
    machine_risk["Risk_Level"] == "Critical"
].shape[0]

average_risk = machine_risk[
    "Maintenance_Risk_Score"
].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Machines",
        total_machines
    )

with col2:
    st.metric(
        "High / Critical",
        high_risk_count
    )

with col3:
    st.metric(
        "Critical",
        critical_count
    )

with col4:
    st.metric(
        "Average Risk",
        f"{average_risk:.1f}"
    )

# ---------------------------------------------------------
# Top 10 machines
# ---------------------------------------------------------

st.markdown(
    "### Top 10 Highest-Risk Machines"
)

top_machine_risk = machine_risk.head(10)

fig_machine_risk = px.bar(
    top_machine_risk,
    x="Maintenance_Risk_Score",
    y="Machine_ID",
    orientation="h",
    text="Maintenance_Risk_Score",
    labels={
        "Maintenance_Risk_Score":
            "Maintenance Risk Score",
        "Machine_ID":
            "Machine"
    }
)

fig_machine_risk.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)

fig_machine_risk.update_layout(
    height=450,
    xaxis=dict(
        range=[0, 100]
    ),
    yaxis=dict(
        categoryorder="total ascending"
    ),
    margin=dict(
        l=20,
        r=40,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig_machine_risk,
    use_container_width=True
)

# ---------------------------------------------------------
# Machine risk table
# ---------------------------------------------------------

st.markdown(
    "### Machine Risk Details"
)

machine_risk_display = machine_risk[
    [
        "Machine_ID",
        "Maintenance_Risk_Score",
        "Maximum_Risk",
        "Average_Anomaly_Score",
        "Anomaly_Count",
        "Anomaly_Rate",
        "Risk_Level"
    ]
].copy()

machine_risk_display.columns = [
    "Machine",
    "Average Risk",
    "Maximum Risk",
    "Avg Anomaly Score",
    "Anomaly Count",
    "Anomaly Rate (%)",
    "Risk Level"
]

st.dataframe(
    machine_risk_display,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# INDIVIDUAL MACHINE ANALYSIS
# =========================================================

st.subheader(
    "🔍 Individual Machine Analysis"
)

st.write(
    "Select a machine to analyze its historical sensor "
    "behavior, anomaly activity, and maintenance risk."
)

# ---------------------------------------------------------
# Machine selector
# ---------------------------------------------------------

machine_list = sorted(
    maintenance_df["Machine_ID"]
    .dropna()
    .unique()
)

selected_machine = st.selectbox(
    "Select Machine",
    machine_list
)

# ---------------------------------------------------------
# Filter selected machine
# ---------------------------------------------------------

selected_machine_df = maintenance_df[
    maintenance_df["Machine_ID"] == selected_machine
].copy()

selected_machine_df = selected_machine_df.sort_values(
    "Datetime"
)

# ---------------------------------------------------------
# Machine KPIs
# ---------------------------------------------------------

machine_avg_risk = (
    selected_machine_df[
        "Maintenance_Risk_Score"
    ].mean()
)

machine_max_risk = (
    selected_machine_df[
        "Maintenance_Risk_Score"
    ].max()
)

machine_avg_anomaly = (
    selected_machine_df[
        "Anomaly_Score"
    ].mean()
)

machine_anomalies = (
    selected_machine_df[
        "Anomaly_Status"
    ] == "Anomaly"
).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Risk",
        f"{machine_avg_risk:.2f}"
    )

with col2:
    st.metric(
        "Maximum Risk",
        f"{machine_max_risk:.2f}"
    )

with col3:
    st.metric(
        "Average Anomaly Score",
        f"{machine_avg_anomaly:.2f}"
    )

with col4:
    st.metric(
        "Anomalous Readings",
        machine_anomalies
    )

# ---------------------------------------------------------
# Risk history
# ---------------------------------------------------------

st.markdown(
    "### Maintenance Risk History"
)

fig_risk_history = px.line(
    selected_machine_df,
    x="Datetime",
    y="Maintenance_Risk_Score",
    markers=True,
    labels={
        "Datetime": "Time",
        "Maintenance_Risk_Score":
            "Maintenance Risk Score"
    }
)

fig_risk_history.update_layout(
    height=400,
    yaxis=dict(
        range=[0, 100]
    ),
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig_risk_history,
    use_container_width=True
)

# =========================================================
# EXPLAINABLE AI - RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "🧠 Explainable AI"
)

st.write(
    "Feature importance shows which input variables "
    "contribute most to the Random Forest efficiency prediction."
)

# ---------------------------------------------------------
# Get feature importance from trained Random Forest
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": classification_features,
    "Importance": random_forest.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

# Convert to percentage
feature_importance["Importance (%)"] = (
    feature_importance["Importance"] * 100
).round(2)

# ---------------------------------------------------------
# Display top features
# ---------------------------------------------------------

top_features = feature_importance.head(10)

fig_feature_importance = px.bar(
    top_features,
    x="Importance (%)",
    y="Feature",
    orientation="h",
    text="Importance (%)",
    title="Top 10 Most Important Features",
    labels={
        "Importance (%)": "Importance (%)",
        "Feature": "Feature"
    }
)

fig_feature_importance.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_feature_importance.update_layout(
    height=500,
    yaxis=dict(
        categoryorder="total ascending"
    ),
    margin=dict(
        l=20,
        r=50,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_feature_importance,
    use_container_width=True
)

# ---------------------------------------------------------
# Complete feature importance table
# ---------------------------------------------------------

st.markdown(
    "### Feature Importance Details"
)

st.dataframe(
    feature_importance[
        [
            "Feature",
            "Importance (%)"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

st.markdown(
    "### 🔎 SHAP Prediction Explanation"
)

st.write(
    "SHAP explains how individual input features influence "
    "the Random Forest prediction for the selected machine."
)

# ---------------------------------------------------------
# Prepare selected machine data
# ---------------------------------------------------------

shap_data = selected_machine_df[
    classification_features
].copy()

shap_data = shap_data.fillna(
    shap_data.median()
)

# ---------------------------------------------------------
# Create SHAP Tree Explainer
# ---------------------------------------------------------

explainer = shap.TreeExplainer(
    random_forest
)

# ---------------------------------------------------------
# Calculate SHAP values
# ---------------------------------------------------------

shap_values = explainer.shap_values(
    shap_data
)

# ---------------------------------------------------------
# Select latest reading of selected machine
# ---------------------------------------------------------

latest_index = len(shap_data) - 1

latest_features = shap_data.iloc[
    latest_index
]

# ---------------------------------------------------------
# Handle SHAP output format
# ---------------------------------------------------------

latest_prediction = random_forest.predict(
    latest_features.to_frame().T
)[0]

class_index = list(
    random_forest.classes_
).index(latest_prediction)

shap_array = np.asarray(shap_values)

# SHAP can return:
# 1. List of arrays for different classes
# 2. 2D array: samples × features
# 3. 3D array: samples × features × classes

if isinstance(shap_values, list):

    latest_shap_values = np.asarray(
        shap_values[class_index][latest_index]
    ).reshape(-1)

elif shap_array.ndim == 3:

    latest_shap_values = (
        shap_array[
            latest_index,
            :,
            class_index
        ]
        .reshape(-1)
    )

elif shap_array.ndim == 2:

    latest_shap_values = (
        shap_array[
            latest_index
        ]
        .reshape(-1)
    )

else:

    latest_shap_values = (
        shap_array
        .reshape(-1)
    )

# Safety check
if len(latest_shap_values) != len(
    classification_features
):

    st.error(
        "SHAP output does not match the "
        "number of classification features."
    )

    st.stop()

# ---------------------------------------------------------
# Create explanation dataframe
# ---------------------------------------------------------

shap_explanation = pd.DataFrame({
    "Feature": classification_features,
    "SHAP_Value": latest_shap_values,
    "Feature_Value": latest_features.values
})

shap_explanation["Absolute_SHAP"] = (
    shap_explanation["SHAP_Value"]
    .abs()
)

shap_explanation = (
    shap_explanation
    .sort_values(
        "Absolute_SHAP",
        ascending=False
    )
    .head(10)
)

# ---------------------------------------------------------
# SHAP bar chart
# ---------------------------------------------------------

fig_shap = px.bar(
    shap_explanation.sort_values(
        "SHAP_Value"
    ),
    x="SHAP_Value",
    y="Feature",
    orientation="h",
    text="SHAP_Value",
    title="Top Features Influencing the Latest Prediction",
    labels={
        "SHAP_Value":
            "SHAP Contribution",
        "Feature":
            "Feature"
    }
)

fig_shap.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig_shap.update_layout(
    height=500,
    margin=dict(
        l=20,
        r=50,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig_shap,
    use_container_width=True
)

# ---------------------------------------------------------
# SHAP details
# ---------------------------------------------------------

st.markdown(
    "### SHAP Feature Contributions"
)

shap_display = shap_explanation[
    [
        "Feature",
        "Feature_Value",
        "SHAP_Value"
    ]
].copy()

shap_display.columns = [
    "Feature",
    "Current Value",
    "SHAP Contribution"
]

st.dataframe(
    shap_display,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# Sensor trends
# ---------------------------------------------------------

st.markdown(
    "### Sensor Trends"
)

sensor_options = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr"
]

selected_sensor = st.selectbox(
    "Select Sensor",
    sensor_options
)

fig_sensor = px.line(
    selected_machine_df,
    x="Datetime",
    y=selected_sensor,
    markers=True,
    labels={
        "Datetime": "Time",
        selected_sensor: selected_sensor
    }
)

fig_sensor.update_layout(
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig_sensor,
    use_container_width=True
)

# ---------------------------------------------------------
# Anomaly history
# ---------------------------------------------------------

st.markdown(
    "### Anomaly Score History"
)

fig_anomaly_history = px.line(
    selected_machine_df,
    x="Datetime",
    y="Anomaly_Score",
    markers=True,
    labels={
        "Datetime": "Time",
        "Anomaly_Score":
            "Anomaly Score"
    }
)

fig_anomaly_history.update_layout(
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    )
)

st.plotly_chart(
    fig_anomaly_history,
    use_container_width=True
)

# =========================================================
# HIGH-PRIORITY MAINTENANCE ALERTS
# =========================================================

st.subheader(
    "🚨 Maintenance Alerts"
)

st.write(
    "Machines and observations requiring maintenance "
    "attention based on the calculated maintenance risk."
)

# ---------------------------------------------------------
# Create alert dataframe
# ---------------------------------------------------------

maintenance_alerts = maintenance_df[
    maintenance_df["Maintenance_Priority"].isin(
        [
            "Warning",
            "High Risk",
            "Critical"
        ]
    )
].copy()

maintenance_alerts = maintenance_alerts.sort_values(
    "Maintenance_Risk_Score",
    ascending=False
)

# ---------------------------------------------------------
# Alert statistics
# ---------------------------------------------------------

total_alerts = len(maintenance_alerts)

warning_alerts = (
    maintenance_alerts["Maintenance_Priority"]
    == "Warning"
).sum()

high_risk_alerts = (
    maintenance_alerts["Maintenance_Priority"]
    == "High Risk"
).sum()

critical_alerts = (
    maintenance_alerts["Maintenance_Priority"]
    == "Critical"
).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Alerts",
        f"{total_alerts:,}"
    )

with col2:
    st.metric(
        "Warning",
        f"{warning_alerts:,}"
    )

with col3:
    st.metric(
        "High Risk",
        f"{high_risk_alerts:,}"
    )

with col4:
    st.metric(
        "Critical",
        f"{critical_alerts:,}"
    )

# ---------------------------------------------------------
# Critical alerts
# ---------------------------------------------------------

critical_df = maintenance_alerts[
    maintenance_alerts["Maintenance_Priority"]
    == "Critical"
].copy()

if len(critical_df) > 0:

    st.markdown(
        "### 🔴 Critical Maintenance Alerts"
    )

    critical_columns = [
        "Machine_ID",
        "Datetime",
        "Maintenance_Risk_Score",
        "Anomaly_Score",
        "Predicted_Efficiency",
        "Temperature_C",
        "Vibration_Hz",
        "Error_Rate_%",
        "Maintenance_Recommendation"
    ]

    st.dataframe(
        critical_df[critical_columns],
        use_container_width=True,
        height=350,
        hide_index=True
    )

else:

    st.success(
        "No critical maintenance alerts detected."
    )

# ---------------------------------------------------------
# High-risk and warning alerts
# ---------------------------------------------------------

st.markdown(
    "### 🟠 Other Maintenance Alerts"
)

alert_columns = [
    "Machine_ID",
    "Datetime",
    "Maintenance_Risk_Score",
    "Maintenance_Priority",
    "Anomaly_Status",
    "Anomaly_Score",
    "Predicted_Efficiency",
    "Temperature_C",
    "Vibration_Hz",
    "Error_Rate_%",
    "Maintenance_Recommendation"
]

st.dataframe(
    maintenance_alerts[alert_columns],
    use_container_width=True,
    height=450,
    hide_index=True
)