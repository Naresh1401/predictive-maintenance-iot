"""
Streamlit Dashboard for Predictive Maintenance.

Provides interactive views:
- Real-time sensor monitoring
- Machine health heatmap
- RUL predictions timeline
- Anomaly detection alerts
- Maintenance scheduling calendar
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_sensor_data():
    path = PROJECT_ROOT / "data" / "raw" / "sensor_readings.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_data
def load_failure_data():
    path = PROJECT_ROOT / "data" / "raw" / "failure_events.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["failure_timestamp"])
    return df


@st.cache_data
def load_maintenance_data():
    path = PROJECT_ROOT / "data" / "raw" / "maintenance_logs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["maintenance_timestamp"])
    return df


@st.cache_data
def load_features_data():
    path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def render_sidebar():
    st.sidebar.title("🏭 Predictive Maintenance")
    st.sidebar.markdown("---")

    sensor_df = load_sensor_data()
    if sensor_df is None:
        st.sidebar.warning("No data found. Run `make data` first.")
        return None, None, None

    machines = sorted(sensor_df["machine_id"].unique())
    selected_machines = st.sidebar.multiselect(
        "Select Machines", machines, default=machines[:5]
    )

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(
            sensor_df["timestamp"].min().date(),
            sensor_df["timestamp"].max().date(),
        ),
    )

    view = st.sidebar.radio(
        "Dashboard View",
        ["Sensor Monitoring", "Health Heatmap", "RUL Predictions", "Anomaly Alerts", "Maintenance Calendar"],
    )

    return selected_machines, date_range, view


def render_sensor_monitoring(sensor_df: pd.DataFrame, machines: list, date_range: tuple):
    st.header("📊 Real-Time Sensor Monitoring")

    sensors = ["temperature", "vibration", "pressure", "rpm", "current", "acoustic_emission"]
    selected_sensor = st.selectbox("Select Sensor", sensors, index=1)

    filtered = sensor_df[
        (sensor_df["machine_id"].isin(machines))
        & (sensor_df["timestamp"].dt.date >= date_range[0])
        & (sensor_df["timestamp"].dt.date <= date_range[1])
    ]

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    # Subsample for plotting performance
    if len(filtered) > 50000:
        filtered = filtered.groupby("machine_id").apply(
            lambda x: x.iloc[::max(1, len(x) // 5000)]
        ).reset_index(drop=True)

    fig = px.line(
        filtered, x="timestamp", y=selected_sensor,
        color="machine_id", title=f"{selected_sensor.replace('_', ' ').title()} Over Time",
    )
    fig.update_layout(height=500, xaxis_title="Time", yaxis_title=selected_sensor.replace("_", " ").title())
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    latest = filtered.groupby("machine_id")[selected_sensor].last()
    col1.metric("Mean (Latest)", f"{latest.mean():.2f}")
    col2.metric("Max (Latest)", f"{latest.max():.2f}")
    col3.metric("Min (Latest)", f"{latest.min():.2f}")
    col4.metric("Std Dev", f"{latest.std():.2f}")


def render_health_heatmap(features_df: pd.DataFrame, machines: list):
    st.header("🟢 Machine Health Heatmap")

    if features_df is None:
        st.warning("No processed features found. Run `make features` first.")
        return

    filtered = features_df[features_df["machine_id"].isin(machines)]
    if filtered.empty:
        st.warning("No data for selected machines.")
        return

    # Get latest health indicator per machine per day
    filtered = filtered.copy()
    filtered["date"] = filtered["timestamp"].dt.date
    daily_health = filtered.groupby(["machine_id", "date"])["health_indicator"].mean().reset_index()

    pivot = daily_health.pivot(index="machine_id", columns="date", values="health_indicator")
    # Show last 30 days
    if pivot.shape[1] > 30:
        pivot = pivot.iloc[:, -30:]

    fig = px.imshow(
        pivot.values,
        labels=dict(x="Date", y="Machine", color="Health Index"),
        x=[str(d) for d in pivot.columns],
        y=list(pivot.index),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Machine Health Indicator (Higher = More Degraded)",
    )
    fig.update_layout(height=max(400, len(machines) * 30))
    st.plotly_chart(fig, use_container_width=True)


def render_rul_predictions(features_df: pd.DataFrame, machines: list, date_range: tuple):
    st.header("⏱️ RUL Predictions Timeline")

    if features_df is None:
        st.warning("No processed features found. Run `make features` first.")
        return

    filtered = features_df[
        (features_df["machine_id"].isin(machines))
        & (features_df["timestamp"].dt.date >= date_range[0])
        & (features_df["timestamp"].dt.date <= date_range[1])
    ]

    if filtered.empty:
        st.warning("No data for selected filters.")
        return

    # RUL timeline
    if len(filtered) > 50000:
        filtered = filtered.groupby("machine_id").apply(
            lambda x: x.iloc[::max(1, len(x) // 5000)]
        ).reset_index(drop=True)

    fig = px.line(
        filtered, x="timestamp", y="rul_hours", color="machine_id",
        title="Remaining Useful Life Over Time",
    )
    fig.add_hline(y=24, line_dash="dash", line_color="red", annotation_text="Critical (24h)")
    fig.add_hline(y=72, line_dash="dash", line_color="orange", annotation_text="Warning (72h)")
    fig.update_layout(height=500, yaxis_title="RUL (hours)")
    st.plotly_chart(fig, use_container_width=True)

    # Current RUL summary
    st.subheader("Current RUL Status")
    latest_rul = features_df[features_df["machine_id"].isin(machines)].groupby("machine_id")["rul_hours"].last()

    cols = st.columns(min(5, len(latest_rul)))
    for i, (machine, rul) in enumerate(latest_rul.items()):
        col = cols[i % len(cols)]
        if rul < 24:
            status = "🔴 CRITICAL"
        elif rul < 72:
            status = "🟠 WARNING"
        elif rul < 168:
            status = "🟡 MONITOR"
        else:
            status = "🟢 HEALTHY"
        col.metric(machine, f"{rul:.0f}h", status)


def render_anomaly_alerts(features_df: pd.DataFrame, machines: list):
    st.header("🚨 Anomaly Detection Alerts")

    if features_df is None:
        st.warning("No processed features found. Run `make features` first.")
        return

    filtered = features_df[
        (features_df["machine_id"].isin(machines))
        & (features_df["is_anomaly"] == 1)
    ]

    if filtered.empty:
        st.success("No anomalies detected for selected machines.")
        return

    st.error(f"⚠️ {len(filtered):,} anomaly readings detected across {filtered['machine_id'].nunique()} machines")

    # Anomalies by machine
    anomaly_counts = filtered["machine_id"].value_counts().reset_index()
    anomaly_counts.columns = ["machine_id", "anomaly_count"]
    fig = px.bar(
        anomaly_counts, x="machine_id", y="anomaly_count",
        title="Anomaly Count by Machine", color="anomaly_count",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly timeline
    filtered_sub = filtered.copy()
    filtered_sub["date"] = filtered_sub["timestamp"].dt.date
    daily_anomalies = filtered_sub.groupby(["date", "machine_id"]).size().reset_index(name="count")

    fig2 = px.scatter(
        daily_anomalies, x="date", y="machine_id", size="count",
        color="count", color_continuous_scale="Reds",
        title="Anomaly Detection Timeline",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Recent anomalies table
    st.subheader("Most Recent Anomalies")
    recent = filtered.sort_values("timestamp", ascending=False).head(20)
    display_cols = ["timestamp", "machine_id", "failure_label", "health_indicator"]
    display_cols = [c for c in display_cols if c in recent.columns]
    st.dataframe(recent[display_cols].reset_index(drop=True), use_container_width=True)


def render_maintenance_calendar(maintenance_df: pd.DataFrame, failure_df: pd.DataFrame, machines: list):
    st.header("📅 Maintenance Scheduling Calendar")

    if maintenance_df is None:
        st.warning("No maintenance data found. Run `make data` first.")
        return

    filtered_maint = maintenance_df[maintenance_df["machine_id"].isin(machines)]

    # Calendar heatmap: maintenance events per day
    filtered_maint = filtered_maint.copy()
    filtered_maint["date"] = filtered_maint["maintenance_timestamp"].dt.date
    daily_counts = filtered_maint.groupby(["date", "maintenance_type"]).size().reset_index(name="count")

    fig = px.scatter(
        daily_counts, x="date", y="count", color="maintenance_type",
        size="count", title="Maintenance Events Over Time",
        labels={"count": "Events", "date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost summary
    st.subheader("Maintenance Cost Summary")
    col1, col2, col3 = st.columns(3)

    preventive = filtered_maint[filtered_maint["maintenance_type"] == "preventive"]
    corrective = filtered_maint[filtered_maint["maintenance_type"] == "corrective"]

    col1.metric("Total Maintenance Events", len(filtered_maint))
    col2.metric("Preventive Cost", f"${preventive['cost_usd'].sum():,.0f}")
    col3.metric("Corrective Cost", f"${corrective['cost_usd'].sum():,.0f}")

    # Upcoming scheduled maintenance (simulated)
    st.subheader("Recommended Maintenance Schedule (Next 14 Days)")
    today = datetime(2026, 1, 1)
    schedule = []
    rng = np.random.RandomState(SEED)
    for machine in machines[:10]:
        next_date = today + timedelta(days=int(rng.randint(1, 15)))
        priority = rng.choice(["High", "Medium", "Low"])
        schedule.append({
            "Machine": machine,
            "Recommended Date": next_date.strftime("%Y-%m-%d"),
            "Type": "Preventive",
            "Priority": priority,
            "Estimated Duration": f"{rng.randint(2, 8)}h",
            "Estimated Cost": f"${rng.randint(300, 2000):,}",
        })
    st.dataframe(pd.DataFrame(schedule), use_container_width=True)


def main():
    selected_machines, date_range, view = render_sidebar()

    if selected_machines is None:
        st.title("🏭 Predictive Maintenance Dashboard")
        st.info("Generate data first using `make data`, then refresh this page.")
        return

    sensor_df = load_sensor_data()
    features_df = load_features_data()
    failure_df = load_failure_data()
    maintenance_df = load_maintenance_data()

    if len(date_range) == 2:
        date_start, date_end = date_range
    else:
        date_start = date_range[0]
        date_end = date_range[0]

    if view == "Sensor Monitoring":
        render_sensor_monitoring(sensor_df, selected_machines, (date_start, date_end))
    elif view == "Health Heatmap":
        render_health_heatmap(features_df, selected_machines)
    elif view == "RUL Predictions":
        render_rul_predictions(features_df, selected_machines, (date_start, date_end))
    elif view == "Anomaly Alerts":
        render_anomaly_alerts(features_df, selected_machines)
    elif view == "Maintenance Calendar":
        render_maintenance_calendar(maintenance_df, failure_df, selected_machines)


if __name__ == "__main__":
    main()
