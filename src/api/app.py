"""
FastAPI Application for Predictive Maintenance.

Endpoints:
- POST /predict/rul        — Predict remaining useful life
- POST /detect/anomaly     — Detect anomalies in sensor readings
- POST /classify/failure   — Classify failure type
- GET  /health             — Service health check
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_engineering import (
    compute_rolling_features,
    compute_rate_of_change,
    compute_health_indicator,
    KEY_SENSORS,
)

app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered equipment failure prediction using IoT sensor data",
    version="1.0.0",
)

# --- Model loading ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


def _load_feature_columns() -> list[str]:
    path = MODELS_DIR / "feature_columns.csv"
    if not path.exists():
        return []
    return pd.read_csv(path, header=None)[0].tolist()


# Lazy-load models on first request
_models_cache: dict = {}


def get_model(key: str, filename: str):
    if key not in _models_cache:
        data = _load_model(filename)
        if data is None:
            raise HTTPException(status_code=503, detail=f"Model '{key}' not loaded. Run training first.")
        _models_cache[key] = data
    return _models_cache[key]


# --- Request/Response schemas ---

class SensorReading(BaseModel):
    """Single sensor reading from a machine."""
    temperature: float = Field(..., description="Temperature in °C")
    vibration: float = Field(..., description="Vibration in mm/s")
    pressure: float = Field(..., description="Pressure in bar")
    rpm: float = Field(..., description="RPM")
    voltage: float = Field(..., description="Voltage in V")
    current: float = Field(..., description="Current in A")
    acoustic_emission: float = Field(..., description="Acoustic emission in dB")
    oil_temperature: float = Field(..., description="Oil temperature in °C")
    coolant_flow: float = Field(..., description="Coolant flow in L/min")
    bearing_frequency: float = Field(..., description="Bearing frequency in Hz")
    spindle_load: float = Field(55.0, description="Spindle load in %")
    feed_rate: float = Field(200.0, description="Feed rate in mm/min")
    torque: float = Field(40.0, description="Torque in Nm")
    humidity: float = Field(45.0, description="Humidity in %")
    power_consumption: float = Field(12.0, description="Power consumption in kW")
    tool_wear: float = Field(100.0, description="Tool wear in µm")
    chip_load: float = Field(0.15, description="Chip load in mm")
    surface_roughness: float = Field(1.6, description="Surface roughness in µm")
    axis_displacement: float = Field(5.0, description="Axis displacement in µm")
    motor_current_phase_imbalance: float = Field(2.0, description="Phase imbalance in %")


class RULResponse(BaseModel):
    machine_id: str = ""
    predicted_rul_hours: float
    risk_level: str
    model_used: str = "XGBoost"


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    model_used: str = "IsolationForest"


class FailureResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_used: str = "XGBoost"


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str


# --- Helper ---

def sensor_to_array(reading: SensorReading) -> np.ndarray:
    """Convert sensor reading to engineered feature array matching training pipeline."""
    raw_values = {
        "temperature": reading.temperature, "vibration": reading.vibration,
        "pressure": reading.pressure, "rpm": reading.rpm,
        "voltage": reading.voltage, "current": reading.current,
        "acoustic_emission": reading.acoustic_emission,
        "oil_temperature": reading.oil_temperature,
        "coolant_flow": reading.coolant_flow, "bearing_frequency": reading.bearing_frequency,
        "spindle_load": reading.spindle_load, "feed_rate": reading.feed_rate,
        "torque": reading.torque, "humidity": reading.humidity,
        "power_consumption": reading.power_consumption,
        "tool_wear": reading.tool_wear, "chip_load": reading.chip_load,
        "surface_roughness": reading.surface_roughness,
        "axis_displacement": reading.axis_displacement,
        "motor_current_phase_imbalance": reading.motor_current_phase_imbalance,
    }
    # Build a single-row DataFrame with raw sensor columns
    df = pd.DataFrame([raw_values])

    # Compute engineered features (rolling stats use single-row min_periods=1)
    rolling_feats = compute_rolling_features(df)
    rate_feats = compute_rate_of_change(df)
    hi = compute_health_indicator(df)

    # Merge all features
    df = pd.concat([df, rolling_feats, rate_feats], axis=1)
    df["health_indicator"] = hi.values

    # Align to saved training feature columns
    feature_cols = _load_feature_columns()
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_cols]

    return df.values.astype(np.float32)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = list(_models_cache.keys())
    return HealthResponse(
        status="healthy",
        models_loaded=loaded,
        version="1.0.0",
    )


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(reading: SensorReading, machine_id: str = "unknown"):
    """Predict remaining useful life from sensor readings."""
    model_data = get_model("xgboost_rul", "xgboost_rul.joblib")
    model = model_data["model"]
    scaler = model_data["scaler"]

    X = sensor_to_array(reading)
    try:
        X_scaled = scaler.transform(X)
        rul = float(model.predict(X_scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")

    rul = max(0, rul)

    if rul < 24:
        risk = "CRITICAL"
    elif rul < 72:
        risk = "HIGH"
    elif rul < 168:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return RULResponse(
        machine_id=machine_id,
        predicted_rul_hours=round(rul, 1),
        risk_level=risk,
    )


@app.post("/detect/anomaly", response_model=AnomalyResponse)
def detect_anomaly(reading: SensorReading):
    """Detect anomalies in sensor readings."""
    model_data = get_model("isolationforest", "isolationforest_detector.joblib")
    model = model_data["model"]
    scaler = model_data["scaler"]

    X = sensor_to_array(reading)
    try:
        X_scaled = scaler.transform(X)
        label = model.predict(X_scaled)[0]
        score = float(model.decision_function(X_scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Detection error: {str(e)}")

    return AnomalyResponse(
        is_anomaly=bool(label == -1),
        anomaly_score=round(score, 4),
    )


@app.post("/classify/failure", response_model=FailureResponse)
def classify_failure(reading: SensorReading):
    """Classify the type of failure from sensor readings."""
    label_names = {0: "normal", 1: "bearing", 2: "motor", 3: "overheating"}
    model_data = get_model("xgboost_classifier", "xgboost_classifier.joblib")
    model = model_data["model"]
    scaler = model_data["scaler"]

    X = sensor_to_array(reading)
    try:
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Classification error: {str(e)}")

    proba_dict = {label_names.get(i, str(i)): round(float(p), 4) for i, p in enumerate(proba)}

    return FailureResponse(
        predicted_class=label_names.get(pred, "unknown"),
        confidence=round(float(max(proba)), 4),
        probabilities=proba_dict,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
