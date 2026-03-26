# Predictive Maintenance for Industrial Equipment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![Domain](https://img.shields.io/badge/Domain-Manufacturing%20%7C%20IoT-red)

**ML-powered equipment failure prediction using sensor data** — remaining useful life (RUL) estimation, anomaly detection, and maintenance scheduling optimization for industrial manufacturing environments.

## Overview

This project addresses the critical challenge of **unplanned equipment downtime** in manufacturing, which costs the industry an estimated $50B annually. By leveraging IoT sensor data from 100 industrial machines across 20 sensor channels, the system provides:

- **Remaining Useful Life (RUL) Estimation**: Predict when equipment will fail, enabling proactive scheduling
- **Anomaly Detection**: Identify abnormal operating patterns before they escalate to failures
- **Failure Classification**: Categorize failure root causes (bearing, motor, overheating) for targeted maintenance
- **Maintenance Optimization**: Cost-benefit analysis to minimize downtime while controlling maintenance spend

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐               │
│  │ 100 IoT  │  │  Maintenance │  │   Machine       │               │
│  │ Machines │──│  Logs        │──│   Metadata      │               │
│  │ 20 Sensr │  │              │  │                 │               │
│  └────┬─────┘  └──────┬───────┘  └───────┬─────────┘               │
│       └───────────────┼──────────────────┘                         │
│                       ▼                                            │
├─────────────────────────────────────────────────────────────────────┤
│                  FEATURE ENGINEERING LAYER                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │Rolling Stats │ │ Freq Domain  │ │ Health Index  │               │
│  │24h/7d window │ │ FFT Features │ │ Construction  │               │
│  └──────┬───────┘ └──────┬───────┘ └──────┬────────┘               │
│         └────────────────┼────────────────┘                        │
│                          ▼                                         │
├─────────────────────────────────────────────────────────────────────┤
│                     MODEL LAYER                                     │
│  ┌────────────────┐ ┌──────────────┐ ┌────────────────────┐       │
│  │ RUL Prediction │ │   Anomaly    │ │ Failure             │       │
│  │ XGB/LGBM/RF/GB │ │  Detection   │ │ Classification      │       │
│  │ MAE: 12.4 hrs  │ │  IF/OCSVM/AE │ │ RF/XGB multi-class  │       │
│  └────────┬───────┘ └──────┬───────┘ └─────────┬──────────┘       │
│           └────────────────┼───────────────────┘                   │
│                            ▼                                       │
├─────────────────────────────────────────────────────────────────────┤
│                    SERVING LAYER                                    │
│  ┌──────────────────┐      ┌──────────────────────┐               │
│  │  FastAPI REST     │      │  Streamlit Dashboard  │               │
│  │  /predict/rul     │      │  Real-time Monitoring  │               │
│  │  /detect/anomaly  │      │  Health Heatmap        │               │
│  │  /classify/failure│      │  Maintenance Calendar  │               │
│  └──────────────────┘      └──────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## Results

### RUL Prediction Performance

| Model              | MAE (hours) | RMSE (hours) | R² Score |
|---------------------|-------------|--------------|----------|
| XGBoost Regressor   | 12.4        | 18.7         | 0.89     |
| LightGBM Regressor  | 13.1        | 19.2         | 0.88     |
| Random Forest       | 14.8        | 21.3         | 0.85     |
| Gradient Boosting   | 13.6        | 20.1         | 0.87     |

### Anomaly Detection Performance

| Model               | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|-----------|--------|----------|---------|
| Isolation Forest     | 0.91      | 0.87   | 0.89     | 0.94    |
| One-Class SVM        | 0.88      | 0.82   | 0.85     | 0.91    |
| Autoencoder (MLP)    | 0.93      | 0.85   | 0.89     | 0.95    |
| Hotelling T²         | 0.86      | 0.90   | 0.88     | 0.93    |

### Failure Classification Performance

| Model           | Accuracy | Macro-F1 | Bearing-F1 | Motor-F1 | Overheat-F1 |
|-----------------|----------|----------|------------|----------|-------------|
| XGBoost         | 0.94     | 0.91     | 0.93       | 0.90     | 0.89        |
| Random Forest   | 0.92     | 0.89     | 0.91       | 0.88     | 0.87        |

## Project Structure

```
predictive-maintenance-iot/
├── data/
│   ├── raw/                     # Generated sensor data
│   └── processed/               # Engineered features
├── src/
│   ├── data/
│   │   └── generate_data.py     # Synthetic IoT data generator
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── anomaly_detection.py
│   │   ├── rul_prediction.py
│   │   ├── failure_classification.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── api/
│   │   └── app.py               # FastAPI serving
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
├── models/                      # Serialized models
├── results/
│   └── figures/                 # Evaluation plots
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/nareshsampangi/predictive-maintenance-iot.git
cd predictive-maintenance-iot
pip install -r requirements.txt

# Generate data, engineer features, train, and evaluate
make all

# Or step by step:
make data        # Generate synthetic sensor data
make features    # Engineer features
make train       # Train all models
make evaluate    # Generate evaluation plots

# Launch API
make api

# Launch Dashboard
make dashboard
```

## Sensor Data

The system ingests data from **20 sensor channels** per machine:

| Sensor               | Unit    | Normal Range  | Failure Indicator          |
|----------------------|---------|---------------|----------------------------|
| Temperature          | °C      | 60–85         | Gradual rise before failure |
| Vibration            | mm/s    | 0.5–4.0       | Spike before bearing fail  |
| Pressure             | bar     | 5.0–8.0       | Drop indicates leaks       |
| RPM                  | rev/min | 1400–1600     | Instability = motor issues |
| Voltage              | V       | 380–420       | Fluctuation = electrical   |
| Current              | A       | 10–25         | Overcurrent = overload     |
| Acoustic Emission    | dB      | 60–80         | High = bearing defects     |
| Oil Temperature      | °C      | 40–60         | Rise = lubrication failure |
| Coolant Flow         | L/min   | 8–12          | Drop = cooling failure     |
| Bearing Frequency    | Hz      | 20–50         | Shift = wear               |
| ... +10 more sensors |         |               |                            |

## Key Techniques

- **Temporal Feature Engineering**: Rolling statistics, rate-of-change, frequency domain (FFT)
- **Health Indicator Construction**: Composite degradation score from multi-sensor fusion
- **Imbalanced Learning**: Class-weighted models for rare failure events
- **Time-Series Cross-Validation**: Temporal split preserving chronological order
- **Cost-Sensitive Evaluation**: Maintenance cost-benefit analysis

## API Endpoints

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| POST   | `/predict/rul`     | Predict remaining useful life      |
| POST   | `/detect/anomaly`  | Detect sensor anomalies            |
| POST   | `/classify/failure`| Classify failure type              |
| GET    | `/health`          | Service health check               |

## Author

**Naresh Sampangi**

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
