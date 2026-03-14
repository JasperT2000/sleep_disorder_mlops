import io
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.sleep_analysis_pipeline import SleepDisorderInferencePipeline

app = FastAPI(title="Sleep Disorder Analysis API")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts/apnea_experiment_mlflow"))
LOG_PATH = Path(os.getenv("LOG_PATH", "/app/logs/inference_log.csv"))

pipeline = SleepDisorderInferencePipeline(model_dir=str(ARTIFACTS_DIR))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "threshold": pipeline.threshold,
        "feature_order": pipeline.feature_order,
        "metadata": pipeline.metadata,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gender: int = Form(...),
    age: int = Form(...),
    bmi: float = Form(...),
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded CSV: {e}")

    if "y_pred" not in df.columns and "Class" in df.columns:
        df = df.copy()
        df["y_pred"] = df["Class"]

    required = {"SubNo", "SegNo", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {sorted(missing)}")

    try:
        results = pipeline.run_from_prediction_dataframe(df, gender=gender, age=age, bmi=bmi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    try:
        log_df = pd.DataFrame(results)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists():
            log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            log_df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass

    return JSONResponse(content={"results": results})