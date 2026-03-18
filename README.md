# Sleep Disorder Prediction

A full end-to-end MLOps project that predicts sleep disorder risk from polysomnography (PSG) data. Takes raw sleep stage predictions, computes clinical sleep metrics, and runs them through a trained machine learning model to estimate disorder risk per subject.

**Live demo:** http://sleepanalysis.duckdns.org:8501

---

## Overview

| Component | Technology |
|---|---|
| ML Training | scikit-learn, LightGBM, MLflow |
| Inference API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Cloud Deployment | AWS EC2 (free tier) |
| Artifact Storage | AWS S3 |

---

## Project Structure

```
sleep_disorder_prediction/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── api/
│   │   └── main.py                  # FastAPI inference endpoint
│   ├── frontend/
│   │   └── streamlit_app.py         # Streamlit UI
│   └── services/
│       └── sleep_analysis_pipeline.py  # Metrics engine + inference pipeline
├── artifacts/
│   └── apnea_experiment_mlflow/
│       ├── best_model.joblib
│       ├── scaler.joblib
│       ├── imputer.joblib
│       ├── metadata.json
│       ├── model_leaderboard.csv
│       ├── random_forest_feature_importance.csv
│       └── lightgbm_feature_importance.csv
└── scripts/
    └── mlflow_train.py              # Training script with MLflow logging
```

---

## How It Works

1. User uploads a CSV containing raw sleep stage predictions (`SubNo`, `SegNo`, `y_pred`)
2. The pipeline computes 20+ clinical sleep metrics per subject (sleep efficiency, REM latency, arousal index, WASO, etc.)
3. Metrics are fed into the trained classifier to produce a disorder risk probability
4. Results are displayed in the Streamlit UI with sleep stage timeline and distribution charts

### Sleep Stage Encoding

| Value | Stage |
|---|---|
| 0 | Awake |
| 1 | Light (N1) |
| 2 | Deep (N3) |
| 3 | REM |

### Risk Levels

| Probability | Risk Level |
|---|---|
| < 0.25 | Low |
| 0.25 – 0.50 | Moderate |
| 0.50 – 0.75 | High |
| > 0.75 | Very High |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata and feature order |
| POST | `/analyze` | Run inference on uploaded CSV |

### `/analyze` request format

| Field | Type | Description |
|---|---|---|
| `file` | CSV file | Columns: `SubNo`, `SegNo`, `y_pred` (or `Class`) |
| `gender` | int | 0 = Female, 1 = Male |
| `age` | int | Subject age |
| `bmi` | float | Subject BMI |

---

## Running Locally

### Prerequisites

- Docker
- Docker Compose

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/JasperT2000/sleep_disorder_mlops.git
cd sleep_disorder_mlops

# 2. Add model artifacts (not included in repo — see Training section)
mkdir -p artifacts/apnea_experiment_mlflow
# copy best_model.joblib, scaler.joblib, imputer.joblib here

# 3. Start the app
mkdir -p logs
docker compose up -d --build

# 4. Open the app
# Streamlit UI  → http://localhost:8501
# API docs      → http://localhost:8000/docs
# Health check  → http://localhost:8000/health
```

---

## Training

The training script uses MLflow to track experiments and saves all artifacts locally.

```bash
python scripts/mlflow_train.py \
  --csv /path/to/your/data.csv \
  --out_dir artifacts/apnea_experiment_mlflow \
  --experiment_name apnea_experiment \
  --threshold 0.25
```

### Training pipeline

1. Feature preprocessing — StandardScaler → KNNImputer
2. Class imbalance handling — RandomOverSampler
3. Model training — RandomForestClassifier (100 estimators)
4. Threshold tuning — default 0.25, optimised for recall on imbalanced clinical data
5. Artifacts saved — `best_model.joblib`, `scaler.joblib`, `imputer.joblib`, `metadata.json`
6. MLflow logging — params, metrics, artifacts, model registry

### Features used

```
gender, age, bmi,
Latency start sleep to REM [min],
Arousal index,
NoREM1%, NoREM2%, NoREM3%, REM%,
sleep time, sleep latency
```

---

## Deployment (AWS EC2)

The app is deployed on an AWS EC2 `t2.micro` instance (free tier) and runs 24/7 via Docker Compose with `restart: unless-stopped`.

### Architecture

```
User → Streamlit (port 8501) → FastAPI (port 8000) → Model artifacts (volume mount)
```

Model artifacts are stored in AWS S3 and pulled to the instance at deploy time:

```bash
aws s3 cp s3://your-bucket/artifacts ./artifacts --recursive
```

---

## Computed Sleep Metrics

| Metric | Description |
|---|---|
| `sleep_efficiency_pct` | % of recording time spent asleep |
| `total_sleep_time_min` | Total minutes asleep |
| `sleep_latency_min` | Minutes from recording start to first sleep |
| `rem_latency_min` | Minutes from sleep onset to first REM |
| `arousal_index` | Awakenings per hour of sleep |
| `waso_min` | Wake After Sleep Onset (minutes) |
| `number_of_awakenings` | Total awakening events |
| `light_sleep_pct` | % of sleep time in N1 |
| `deep_sleep_pct` | % of sleep time in N3 |
| `rem_sleep_pct` | % of sleep time in REM |
| `stage_transitions_per_hour` | Sleep stage changes per hour |
| `longest_sleep_run_min` | Longest uninterrupted sleep period |
| `rem_bouts` | Number of distinct REM episodes |

---

## Tech Stack

- **Python 3.11**
- **FastAPI** — REST inference API
- **Streamlit** — interactive frontend
- **scikit-learn** — preprocessing and Random Forest
- **LightGBM** — gradient boosting (compared during training)
- **MLflow** — experiment tracking and model registry
- **joblib** — model serialization
- **Docker / Docker Compose** — containerization
- **AWS EC2** — cloud hosting
- **AWS S3** — artifact storage

---

## License

MIT
