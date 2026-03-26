"""
Synthetic IoT Sensor Data Generator for Predictive Maintenance.

Generates realistic sensor readings for 100 industrial machines with
degradation patterns, failure events, and maintenance logs.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# --- Configuration ---
NUM_MACHINES = 100
NUM_DAYS = 365
READINGS_PER_DAY = 24  # hourly
TOTAL_READINGS = NUM_DAYS * READINGS_PER_DAY  # 8760 per machine

SENSOR_SPECS = {
    "temperature":        {"unit": "°C",      "mean": 72.0,  "std": 5.0,   "range": (60, 85)},
    "vibration":          {"unit": "mm/s",     "mean": 2.0,   "std": 0.5,   "range": (0.5, 4.0)},
    "pressure":           {"unit": "bar",      "mean": 6.5,   "std": 0.5,   "range": (5.0, 8.0)},
    "rpm":                {"unit": "rev/min",  "mean": 1500,  "std": 30,    "range": (1400, 1600)},
    "voltage":            {"unit": "V",        "mean": 400,   "std": 8,     "range": (380, 420)},
    "current":            {"unit": "A",        "mean": 17.5,  "std": 3.0,   "range": (10, 25)},
    "acoustic_emission":  {"unit": "dB",       "mean": 70,    "std": 4,     "range": (60, 80)},
    "oil_temperature":    {"unit": "°C",       "mean": 50,    "std": 4,     "range": (40, 60)},
    "coolant_flow":       {"unit": "L/min",    "mean": 10,    "std": 0.8,   "range": (8, 12)},
    "bearing_frequency":  {"unit": "Hz",       "mean": 35,    "std": 5,     "range": (20, 50)},
    "spindle_load":       {"unit": "%",        "mean": 55,    "std": 8,     "range": (30, 80)},
    "feed_rate":          {"unit": "mm/min",   "mean": 200,   "std": 20,    "range": (150, 250)},
    "torque":             {"unit": "Nm",       "mean": 40,    "std": 6,     "range": (25, 60)},
    "humidity":           {"unit": "%",        "mean": 45,    "std": 5,     "range": (30, 65)},
    "power_consumption":  {"unit": "kW",       "mean": 12,    "std": 2,     "range": (7, 18)},
    "tool_wear":          {"unit": "µm",       "mean": 100,   "std": 30,    "range": (0, 250)},
    "chip_load":          {"unit": "mm",       "mean": 0.15,  "std": 0.03,  "range": (0.05, 0.30)},
    "surface_roughness":  {"unit": "µm",       "mean": 1.6,   "std": 0.3,   "range": (0.8, 3.2)},
    "axis_displacement":  {"unit": "µm",       "mean": 5.0,   "std": 1.5,   "range": (0, 15)},
    "motor_current_phase_imbalance": {"unit": "%", "mean": 2.0, "std": 0.8, "range": (0, 8)},
}

FAILURE_TYPES = ["bearing", "motor", "overheating"]

DEGRADATION_PROFILES = {
    "bearing": {
        "vibration":         {"drift": 0.008, "spike": 3.0},
        "acoustic_emission": {"drift": 0.005, "spike": 2.5},
        "bearing_frequency": {"drift": 0.006, "spike": 2.0},
        "temperature":       {"drift": 0.003, "spike": 1.5},
    },
    "motor": {
        "current":           {"drift": 0.006, "spike": 2.5},
        "voltage":           {"drift": -0.004, "spike": -2.0},
        "rpm":               {"drift": -0.005, "spike": -3.0},
        "motor_current_phase_imbalance": {"drift": 0.007, "spike": 2.0},
        "torque":            {"drift": 0.004, "spike": 1.8},
    },
    "overheating": {
        "temperature":       {"drift": 0.010, "spike": 4.0},
        "oil_temperature":   {"drift": 0.008, "spike": 3.0},
        "coolant_flow":      {"drift": -0.005, "spike": -2.0},
        "power_consumption": {"drift": 0.004, "spike": 2.0},
    },
}


def _generate_base_signal(n_points: int, spec: dict) -> np.ndarray:
    """Generate a base sensor signal with realistic noise and autocorrelation."""
    noise = np.random.normal(0, spec["std"] * 0.3, n_points)
    # Add autocorrelation via cumulative smoothing
    smoothed = np.zeros(n_points)
    smoothed[0] = spec["mean"] + noise[0]
    alpha = 0.05
    for i in range(1, n_points):
        smoothed[i] = (1 - alpha) * smoothed[i - 1] + alpha * (spec["mean"] + noise[i])
    # Add diurnal pattern (slight temp/load variation)
    hours = np.arange(n_points) % 24
    diurnal = np.sin(2 * np.pi * hours / 24) * spec["std"] * 0.15
    signal = smoothed + diurnal
    return signal


def _apply_degradation(
    signal: np.ndarray,
    degradation_start: int,
    failure_point: int,
    drift_rate: float,
    spike_magnitude: float,
    sensor_std: float,
) -> np.ndarray:
    """Apply gradual degradation pattern leading up to failure."""
    degraded = signal.copy()
    deg_length = failure_point - degradation_start
    if deg_length <= 0:
        return degraded

    # Gradual degradation (exponential ramp)
    t = np.arange(deg_length)
    drift = drift_rate * sensor_std * (np.exp(t / deg_length * 3) - 1)
    if spike_magnitude < 0:
        drift = -np.abs(drift)

    degraded[degradation_start:failure_point] += drift

    # Add increasing noise during degradation
    extra_noise = np.random.normal(0, sensor_std * 0.2 * (t / deg_length), deg_length)
    degraded[degradation_start:failure_point] += extra_noise

    # Spike at failure
    if failure_point < len(degraded):
        spike_window = min(6, len(degraded) - failure_point)
        degraded[failure_point: failure_point + spike_window] += (
            spike_magnitude * sensor_std * np.linspace(1, 0.3, spike_window)
        )
    return degraded


def _generate_failure_schedule(rng: np.random.RandomState) -> list[dict]:
    """Generate failure events for a single machine."""
    failures = []
    num_failures = rng.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.30, 0.10])
    occupied = set()

    for _ in range(num_failures):
        for _attempt in range(50):
            failure_hour = rng.randint(720, TOTAL_READINGS - 48)
            # Ensure failures are spaced at least 720h (30 days) apart
            if all(abs(failure_hour - occ) > 720 for occ in occupied):
                occupied.add(failure_hour)
                deg_start = failure_hour - rng.randint(168, 504)  # 7–21 days before
                deg_start = max(0, deg_start)
                failures.append({
                    "failure_hour": failure_hour,
                    "degradation_start": deg_start,
                    "failure_type": rng.choice(FAILURE_TYPES),
                })
                break
    return failures


def generate_machine_metadata() -> pd.DataFrame:
    """Generate machine metadata."""
    rng = np.random.RandomState(SEED)
    manufacturers = ["Siemens", "ABB", "Fanuc", "Bosch", "Mitsubishi", "Haas"]
    models = ["CNC-X200", "CNC-X400", "CNC-X600", "LATHE-100", "MILL-300", "DRILL-500"]
    locations = ["Plant-A", "Plant-B", "Plant-C"]

    records = []
    for mid in range(1, NUM_MACHINES + 1):
        install_year = rng.randint(2015, 2024)
        records.append({
            "machine_id": f"M{mid:03d}",
            "manufacturer": rng.choice(manufacturers),
            "model": rng.choice(models),
            "install_year": install_year,
            "age_years": 2026 - install_year,
            "location": rng.choice(locations),
            "rated_power_kw": rng.choice([15, 20, 25, 30, 40]),
            "operating_hours_at_start": rng.randint(5000, 50000),
        })
    return pd.DataFrame(records)


def generate_sensor_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sensor readings, failure events, and maintenance logs."""
    start_date = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start=start_date, periods=TOTAL_READINGS, freq="h")

    all_sensor_records = []
    all_failure_records = []
    all_maintenance_records = []

    for mid in range(1, NUM_MACHINES + 1):
        machine_id = f"M{mid:03d}"
        rng = np.random.RandomState(SEED + mid)

        # Generate failure schedule for this machine
        failures = _generate_failure_schedule(rng)

        # Generate base signals for all sensors
        signals = {}
        for sensor_name, spec in SENSOR_SPECS.items():
            signals[sensor_name] = _generate_base_signal(TOTAL_READINGS, spec)

        # Apply degradation for each failure event
        for f_event in failures:
            f_type = f_event["failure_type"]
            deg_start = f_event["degradation_start"]
            f_hour = f_event["failure_hour"]

            profile = DEGRADATION_PROFILES[f_type]
            for sensor_name, params in profile.items():
                spec = SENSOR_SPECS[sensor_name]
                signals[sensor_name] = _apply_degradation(
                    signals[sensor_name],
                    deg_start, f_hour,
                    params["drift"], params["spike"], spec["std"],
                )

            # Record failure event
            all_failure_records.append({
                "machine_id": machine_id,
                "failure_timestamp": timestamps[f_hour],
                "failure_type": f_type,
                "degradation_start": timestamps[deg_start],
                "downtime_hours": rng.randint(4, 48),
            })

            # Unscheduled maintenance after failure
            maint_time = min(f_hour + rng.randint(1, 6), TOTAL_READINGS - 1)
            all_maintenance_records.append({
                "machine_id": machine_id,
                "maintenance_timestamp": timestamps[maint_time],
                "maintenance_type": "corrective",
                "description": f"Repair due to {f_type} failure",
                "duration_hours": rng.randint(2, 24),
                "cost_usd": rng.randint(500, 10000),
            })

        # Scheduled maintenance every ~90 days
        for day in range(0, NUM_DAYS, rng.randint(75, 105)):
            hour_idx = day * 24 + rng.randint(8, 16)
            if hour_idx < TOTAL_READINGS:
                all_maintenance_records.append({
                    "machine_id": machine_id,
                    "maintenance_timestamp": timestamps[hour_idx],
                    "maintenance_type": "preventive",
                    "description": "Scheduled preventive maintenance",
                    "duration_hours": rng.randint(1, 8),
                    "cost_usd": rng.randint(200, 2000),
                })

        # Build sensor dataframe rows (subsample to reduce size: every 1h is kept)
        sensor_data = {"timestamp": timestamps, "machine_id": machine_id}
        for sensor_name, signal in signals.items():
            sensor_data[sensor_name] = np.round(signal, 4)

        df_machine = pd.DataFrame(sensor_data)
        all_sensor_records.append(df_machine)

        if mid % 20 == 0:
            print(f"  Generated data for {mid}/{NUM_MACHINES} machines...")

    sensor_df = pd.concat(all_sensor_records, ignore_index=True)
    failure_df = pd.DataFrame(all_failure_records)
    maintenance_df = pd.DataFrame(all_maintenance_records)

    return sensor_df, failure_df, maintenance_df


def main():
    """Generate and save all datasets."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Generating machine metadata...")
    metadata_df = generate_machine_metadata()
    metadata_df.to_csv(raw_dir / "machine_metadata.csv", index=False)
    print(f"  Saved {len(metadata_df)} machines to machine_metadata.csv")

    print("Generating sensor data, failure events, and maintenance logs...")
    sensor_df, failure_df, maintenance_df = generate_sensor_data()

    sensor_df.to_csv(raw_dir / "sensor_readings.csv", index=False)
    print(f"  Saved {len(sensor_df):,} sensor readings to sensor_readings.csv")

    failure_df.to_csv(raw_dir / "failure_events.csv", index=False)
    print(f"  Saved {len(failure_df)} failure events to failure_events.csv")

    maintenance_df.to_csv(raw_dir / "maintenance_logs.csv", index=False)
    print(f"  Saved {len(maintenance_df)} maintenance records to maintenance_logs.csv")

    print("Data generation complete.")


if __name__ == "__main__":
    main()
