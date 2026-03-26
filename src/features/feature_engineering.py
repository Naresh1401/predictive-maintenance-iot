"""
Feature Engineering for IoT Sensor Data.

Constructs time-domain, frequency-domain, and health indicator features
from raw sensor readings for predictive maintenance modeling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq

SEED = 42
np.random.seed(SEED)

ROLLING_WINDOWS = {
    "24h": 24,
    "7d": 168,
}

KEY_SENSORS = [
    "temperature", "vibration", "pressure", "rpm", "voltage", "current",
    "acoustic_emission", "oil_temperature", "coolant_flow", "bearing_frequency",
    "power_consumption", "torque", "motor_current_phase_imbalance",
]


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling statistics (mean, std, max, min, skew, kurtosis) for key sensors."""
    features = {}
    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            continue
        for window_name, window_size in ROLLING_WINDOWS.items():
            rolled = df[sensor].rolling(window=window_size, min_periods=1)
            features[f"{sensor}_mean_{window_name}"] = rolled.mean()
            features[f"{sensor}_std_{window_name}"] = rolled.std().fillna(0)
            features[f"{sensor}_max_{window_name}"] = rolled.max()
            features[f"{sensor}_min_{window_name}"] = rolled.min()
            features[f"{sensor}_skew_{window_name}"] = rolled.skew().fillna(0)
            features[f"{sensor}_kurt_{window_name}"] = rolled.kurt().fillna(0)
    return pd.DataFrame(features, index=df.index)


def compute_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate of change (first difference and percent change) for key sensors."""
    features = {}
    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            continue
        features[f"{sensor}_diff_1h"] = df[sensor].diff().fillna(0)
        features[f"{sensor}_pct_change_1h"] = df[sensor].pct_change().fillna(0).replace(
            [np.inf, -np.inf], 0
        )
        # 6-hour rate of change
        features[f"{sensor}_diff_6h"] = df[sensor].diff(periods=6).fillna(0)
    return pd.DataFrame(features, index=df.index)


def compute_frequency_features(df: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
    """Compute frequency-domain features using FFT over sliding windows."""
    features = {
        f"{sensor}_peak_freq": np.zeros(len(df)) for sensor in KEY_SENSORS if sensor in df.columns
    }
    features.update(
        {f"{sensor}_spectral_energy": np.zeros(len(df)) for sensor in KEY_SENSORS if sensor in df.columns}
    )
    features.update(
        {f"{sensor}_spectral_entropy": np.zeros(len(df)) for sensor in KEY_SENSORS if sensor in df.columns}
    )

    sample_rate = 1.0  # 1 sample per hour
    freqs = rfftfreq(window_size, d=1.0 / sample_rate)

    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            continue
        values = df[sensor].values
        for i in range(window_size, len(values)):
            window = values[i - window_size: i]
            # Detrend
            window = window - np.mean(window)
            fft_vals = np.abs(rfft(window))
            if fft_vals.sum() > 0:
                features[f"{sensor}_peak_freq"][i] = freqs[np.argmax(fft_vals[1:]) + 1]
                features[f"{sensor}_spectral_energy"][i] = np.sum(fft_vals ** 2)
                # Spectral entropy
                psd = fft_vals ** 2
                psd_norm = psd / (psd.sum() + 1e-10)
                psd_norm = psd_norm[psd_norm > 0]
                features[f"{sensor}_spectral_entropy"][i] = -np.sum(
                    psd_norm * np.log2(psd_norm + 1e-10)
                )

    return pd.DataFrame(features, index=df.index)


def compute_health_indicator(df: pd.DataFrame) -> pd.Series:
    """
    Construct a composite Health Indicator (HI) from multi-sensor fusion.

    HI = 0 (healthy) to 1 (near failure), based on z-score deviations.
    """
    z_scores = pd.DataFrame(index=df.index)
    weights = {
        "temperature": 1.0, "vibration": 1.5, "pressure": 0.8,
        "acoustic_emission": 1.2, "oil_temperature": 0.9,
        "current": 1.0, "bearing_frequency": 1.3,
        "motor_current_phase_imbalance": 1.1, "torque": 0.7,
    }
    total_weight = 0
    weighted_sum = pd.Series(0.0, index=df.index)

    for sensor, weight in weights.items():
        if sensor not in df.columns:
            continue
        col = df[sensor]
        z = (col - col.mean()) / (col.std() + 1e-10)
        z_abs = z.abs()
        weighted_sum += z_abs * weight
        total_weight += weight

    hi = weighted_sum / (total_weight + 1e-10)
    # Normalize to 0-1 using min-max
    hi = (hi - hi.min()) / (hi.max() - hi.min() + 1e-10)
    return hi


def compute_time_since_maintenance(
    df: pd.DataFrame, maintenance_df: pd.DataFrame, machine_id: str
) -> pd.Series:
    """Compute hours since last maintenance for a machine."""
    machine_maint = maintenance_df[maintenance_df["machine_id"] == machine_id].copy()
    ts_col = "maintenance_timestamp"
    if ts_col in machine_maint.columns:
        machine_maint[ts_col] = pd.to_datetime(machine_maint[ts_col])
        maint_times = sorted(machine_maint[ts_col].tolist())
    else:
        maint_times = []

    timestamps = pd.to_datetime(df["timestamp"])
    hours_since = np.zeros(len(df))
    last_maint = timestamps.iloc[0]

    for i, ts in enumerate(timestamps):
        while maint_times and maint_times[0] <= ts:
            last_maint = maint_times.pop(0)
        hours_since[i] = (ts - last_maint).total_seconds() / 3600.0

    return pd.Series(hours_since, index=df.index, name="hours_since_maintenance")


def compute_operating_condition_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin continuous operating conditions into categorical features."""
    features = {}
    if "rpm" in df.columns:
        features["rpm_bin"] = pd.cut(
            df["rpm"], bins=[0, 1450, 1500, 1550, 1700],
            labels=["low", "normal_low", "normal_high", "high"]
        ).cat.codes
    if "temperature" in df.columns:
        features["temp_bin"] = pd.cut(
            df["temperature"], bins=[0, 65, 75, 80, 200],
            labels=["cold", "normal", "warm", "hot"]
        ).cat.codes
    if "power_consumption" in df.columns:
        features["power_bin"] = pd.cut(
            df["power_consumption"], bins=[0, 9, 13, 16, 100],
            labels=["low", "normal", "high", "critical"]
        ).cat.codes
    return pd.DataFrame(features, index=df.index)


def compute_rul_target(
    df: pd.DataFrame, failure_df: pd.DataFrame, machine_id: str
) -> pd.Series:
    """
    Compute Remaining Useful Life (RUL) target variable.

    For each timestamp, RUL = hours until next failure.
    If no upcoming failure, RUL is capped at max_rul.
    """
    max_rul = 720  # 30 days cap

    machine_failures = failure_df[failure_df["machine_id"] == machine_id].copy()
    if "failure_timestamp" in machine_failures.columns:
        machine_failures["failure_timestamp"] = pd.to_datetime(
            machine_failures["failure_timestamp"]
        )
        failure_times = sorted(machine_failures["failure_timestamp"].tolist())
    else:
        failure_times = []

    timestamps = pd.to_datetime(df["timestamp"])
    rul = np.full(len(df), max_rul, dtype=np.float64)

    for i, ts in enumerate(timestamps):
        # Find next failure after this timestamp
        next_failures = [ft for ft in failure_times if ft > ts]
        if next_failures:
            hours_to_failure = (next_failures[0] - ts).total_seconds() / 3600.0
            rul[i] = min(hours_to_failure, max_rul)

    return pd.Series(rul, index=df.index, name="rul_hours")


def compute_failure_label(
    df: pd.DataFrame, failure_df: pd.DataFrame, machine_id: str,
    pre_failure_window: int = 48,
) -> pd.DataFrame:
    """
    Create failure classification labels.

    Labels: 0=normal, 1=bearing, 2=motor, 3=overheating.
    A reading is labeled as failure-type if within pre_failure_window hours before failure.
    """
    label_map = {"normal": 0, "bearing": 1, "motor": 2, "overheating": 3}
    machine_failures = failure_df[failure_df["machine_id"] == machine_id].copy()

    labels = np.zeros(len(df), dtype=int)
    is_anomaly = np.zeros(len(df), dtype=int)
    timestamps = pd.to_datetime(df["timestamp"])

    if "failure_timestamp" in machine_failures.columns:
        machine_failures["failure_timestamp"] = pd.to_datetime(
            machine_failures["failure_timestamp"]
        )
        for _, row in machine_failures.iterrows():
            ft = row["failure_timestamp"]
            f_type = row["failure_type"]
            window_start = ft - pd.Timedelta(hours=pre_failure_window)

            mask = (timestamps >= window_start) & (timestamps <= ft)
            labels[mask] = label_map[f_type]
            is_anomaly[mask] = 1

    result = pd.DataFrame({
        "failure_label": labels,
        "is_anomaly": is_anomaly,
    }, index=df.index)
    return result


def engineer_features_for_machine(
    machine_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    machine_id: str,
) -> pd.DataFrame:
    """Run full feature engineering pipeline for a single machine."""
    df = machine_df.copy().reset_index(drop=True)

    # Rolling statistics
    rolling_feats = compute_rolling_features(df)

    # Rate of change
    roc_feats = compute_rate_of_change(df)

    # Frequency features (expensive — subsample for large datasets)
    freq_feats = compute_frequency_features(df, window_size=24)

    # Health indicator
    health_indicator = compute_health_indicator(df)

    # Time since maintenance
    time_since_maint = compute_time_since_maintenance(df, maintenance_df, machine_id)

    # Operating condition bins
    op_bins = compute_operating_condition_bins(df)

    # Target variables
    rul = compute_rul_target(df, failure_df, machine_id)
    failure_labels = compute_failure_label(df, failure_df, machine_id)

    # Combine all features
    result = pd.concat([
        df[["timestamp", "machine_id"]],
        df[list(set(df.columns) - {"timestamp", "machine_id"})],
        rolling_feats,
        roc_feats,
        freq_feats,
        pd.DataFrame({"health_indicator": health_indicator}),
        pd.DataFrame(time_since_maint),
        op_bins,
        pd.DataFrame(rul),
        failure_labels,
    ], axis=1)

    return result


def main():
    """Run feature engineering on all machines and save processed data."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    sensor_df = pd.read_csv(raw_dir / "sensor_readings.csv")
    failure_df = pd.read_csv(raw_dir / "failure_events.csv")
    maintenance_df = pd.read_csv(raw_dir / "maintenance_logs.csv")

    machine_ids = sorted(sensor_df["machine_id"].unique())
    print(f"Processing {len(machine_ids)} machines...")

    all_features = []
    for i, machine_id in enumerate(machine_ids):
        machine_data = sensor_df[sensor_df["machine_id"] == machine_id].copy()
        features = engineer_features_for_machine(
            machine_data, failure_df, maintenance_df, machine_id
        )
        all_features.append(features)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(machine_ids)} machines...")

    processed_df = pd.concat(all_features, ignore_index=True)

    # Save processed dataset
    processed_df.to_csv(processed_dir / "features.csv", index=False)
    print(f"  Saved {len(processed_df):,} rows to features.csv")

    # Save feature list
    feature_cols = [
        c for c in processed_df.columns
        if c not in ["timestamp", "machine_id", "rul_hours", "failure_label", "is_anomaly"]
    ]
    pd.Series(feature_cols).to_csv(processed_dir / "feature_columns.csv", index=False)
    print(f"  {len(feature_cols)} features saved to feature_columns.csv")

    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
