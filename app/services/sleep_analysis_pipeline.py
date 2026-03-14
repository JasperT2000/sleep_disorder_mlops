import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EPOCH_SECONDS = 30
DEFAULT_THRESHOLD = 0.25


class SleepMetrics4Class:
    def __init__(self, pred_df: pd.DataFrame):
        self.data = pred_df.copy()

        required = {"SubNo", "SegNo", "y_pred"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.data = self.data.sort_values(["SubNo", "SegNo"]).reset_index(drop=True)

    @staticmethod
    def _runs_of_value(arr, value):
        runs = []
        start = None
        for i, v in enumerate(arr):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(arr) - 1))
        return runs

    @staticmethod
    def _runs_of_sleep(arr):
        runs = []
        start = None
        for i, v in enumerate(arr):
            if v != 0 and start is None:
                start = i
            elif v == 0 and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(arr) - 1))
        return runs

    def compute_metrics_for_subject(self, sub_df: pd.DataFrame) -> dict:
        stages = sub_df["y_pred"].to_numpy()
        total_epochs = len(stages)

        if total_epochs == 0:
            return {}

        total_recording_min = total_epochs * EPOCH_SECONDS / 60.0

        sleep_mask = stages != 0
        sleep_epochs = int(np.sum(sleep_mask))
        awake_epochs = int(np.sum(stages == 0))

        total_sleep_time_min = sleep_epochs * EPOCH_SECONDS / 60.0
        sleep_efficiency_pct = (sleep_epochs / total_epochs) * 100.0 if total_epochs else 0.0
        awake_pct_recording = (awake_epochs / total_epochs) * 100.0 if total_epochs else 0.0

        sleep_indices = np.where(stages != 0)[0]
        sleep_latency_min = sleep_indices[0] * EPOCH_SECONDS / 60.0 if len(sleep_indices) else np.nan

        rem_indices = np.where(stages == 3)[0]
        if len(rem_indices) and len(sleep_indices) and rem_indices[0] >= sleep_indices[0]:
            rem_latency_min = (rem_indices[0] - sleep_indices[0]) * EPOCH_SECONDS / 60.0
        else:
            rem_latency_min = np.nan

        light_epochs = int(np.sum(stages == 1))
        deep_epochs = int(np.sum(stages == 2))
        rem_epochs = int(np.sum(stages == 3))

        light_sleep_pct = (light_epochs / sleep_epochs) * 100.0 if sleep_epochs else 0.0
        deep_sleep_pct = (deep_epochs / sleep_epochs) * 100.0 if sleep_epochs else 0.0
        rem_sleep_pct = (rem_epochs / sleep_epochs) * 100.0 if sleep_epochs else 0.0

        sleep_to_awake = (stages[:-1] != 0) & (stages[1:] == 0)
        number_of_awakenings = int(np.sum(sleep_to_awake))

        sleep_hours = (sleep_epochs * EPOCH_SECONDS) / 3600.0
        arousal_index = number_of_awakenings / sleep_hours if sleep_hours else 0.0

        if len(sleep_indices):
            first_sleep_idx = sleep_indices[0]
            waso_epochs = int(np.sum(stages[first_sleep_idx:] == 0))
        else:
            waso_epochs = 0
        waso_min = waso_epochs * EPOCH_SECONDS / 60.0

        awake_runs = self._runs_of_value(stages, 0)
        awake_runs_after_sleep = [(s, e) for s, e in awake_runs if len(sleep_indices) and s >= sleep_indices[0]]
        awakening_durations_min = [((e - s + 1) * EPOCH_SECONDS) / 60.0 for s, e in awake_runs_after_sleep]
        mean_awakening_duration_min = float(np.mean(awakening_durations_min)) if awakening_durations_min else 0.0
        max_awakening_duration_min = float(np.max(awakening_durations_min)) if awakening_durations_min else 0.0

        sleep_runs = self._runs_of_sleep(stages)
        sleep_run_lengths_min = [((e - s + 1) * EPOCH_SECONDS) / 60.0 for s, e in sleep_runs]
        longest_sleep_run_min = float(np.max(sleep_run_lengths_min)) if sleep_run_lengths_min else 0.0

        rem_runs = self._runs_of_value(stages, 3)
        rem_bouts = len(rem_runs)
        rem_bout_lengths_min = [((e - s + 1) * EPOCH_SECONDS) / 60.0 for s, e in rem_runs]
        mean_rem_bout_min = float(np.mean(rem_bout_lengths_min)) if rem_bout_lengths_min else 0.0

        transitions = int(np.sum(stages[1:] != stages[:-1]))
        stage_transitions_per_hour = transitions / (total_recording_min / 60.0) if total_recording_min > 0 else 0.0

        deep_to_light_ratio = (deep_epochs / light_epochs) if light_epochs > 0 else np.nan
        rem_to_sleep_ratio = (rem_epochs / sleep_epochs) if sleep_epochs > 0 else np.nan

        return {
            "total_recording_min": total_recording_min,
            "total_sleep_time_min": total_sleep_time_min,
            "sleep_efficiency_pct": sleep_efficiency_pct,
            "sleep_latency_min": sleep_latency_min,
            "rem_latency_min": rem_latency_min,
            "light_sleep_pct": light_sleep_pct,
            "deep_sleep_pct": deep_sleep_pct,
            "rem_sleep_pct": rem_sleep_pct,
            "awake_pct_recording": awake_pct_recording,
            "arousal_index": arousal_index,
            "number_of_awakenings": number_of_awakenings,
            "mean_awakening_duration_min": mean_awakening_duration_min,
            "max_awakening_duration_min": max_awakening_duration_min,
            "waso_min": waso_min,
            "stage_transitions": transitions,
            "stage_transitions_per_hour": stage_transitions_per_hour,
            "longest_sleep_run_min": longest_sleep_run_min,
            "rem_bouts": rem_bouts,
            "mean_rem_bout_min": mean_rem_bout_min,
            "deep_to_light_ratio": deep_to_light_ratio,
            "rem_to_sleep_ratio": rem_to_sleep_ratio,
        }


class SleepDisorderInferencePipeline:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        required_files = [
            model_dir / "scaler.joblib",
            model_dir / "imputer.joblib",
            model_dir / "best_model.joblib",
        ]
        missing_files = [str(p) for p in required_files if not p.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing model artifacts: {missing_files}")

        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.imputer = joblib.load(model_dir / "imputer.joblib")
        self.model = joblib.load(model_dir / "best_model.joblib")

        self.metadata = {}
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.threshold = float(self.metadata.get("threshold", DEFAULT_THRESHOLD))
        self.feature_order = self.metadata.get(
            "features",
            [
                "gender",
                "age",
                "Latency start sleep to REM [min]",
                "Arousal index",
                "NoREM1%",
                "NoREM2%",
                "NoREM3%",
                "REM%",
                "sleep time",
                "sleep latency",
                "bmi",
            ],
        )

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.25:
            return "Low"
        if prob < 0.50:
            return "Moderate"
        if prob < 0.75:
            return "High"
        return "Very High"

    def _build_feature_row(self, metrics: dict, gender: int, age: int, bmi: float) -> pd.DataFrame:
        row = {
            "gender": int(gender),
            "age": int(age),
            "Latency start sleep to REM [min]": metrics.get("rem_latency_min", np.nan),
            "Arousal index": metrics.get("arousal_index", np.nan),
            "NoREM1%": metrics.get("light_sleep_pct", np.nan),
            "NoREM2%": 0.0,
            "NoREM3%": metrics.get("deep_sleep_pct", np.nan),
            "REM%": metrics.get("rem_sleep_pct", np.nan),
            "sleep time": metrics.get("total_sleep_time_min", np.nan),
            "sleep latency": metrics.get("sleep_latency_min", np.nan),
            "bmi": float(bmi),
        }

        df = pd.DataFrame([row])
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan

        return df[self.feature_order]

    def run_from_prediction_dataframe(self, pred_df: pd.DataFrame, gender: int, age: int, bmi: float):
        if pred_df.empty:
            raise ValueError("Prediction dataframe is empty.")

        required = {"SubNo", "SegNo", "y_pred"}
        missing = required - set(pred_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        pred_df = pred_df.copy()
        pred_df = pred_df.sort_values(["SubNo", "SegNo"]).reset_index(drop=True)

        metrics_engine = SleepMetrics4Class(pred_df)
        results = []

        for subno, sub_df in pred_df.groupby("SubNo"):
            sub_df = sub_df.sort_values("SegNo")
            metrics = metrics_engine.compute_metrics_for_subject(sub_df)

            if not metrics:
                continue

            X = self._build_feature_row(metrics, gender=gender, age=age, bmi=bmi)

            # Keep same order as training for output consistency.
            X_scaled = self.scaler.transform(X)
            X_proc = self.imputer.transform(X_scaled)

            prob = float(self.model.predict_proba(X_proc)[0, 1])
            pred = int(prob >= self.threshold)

            results.append(
                {
                    "SubNo": int(subno),
                    "disorder_risk_probability": prob,
                    "disorder_risk_prediction": pred,
                    "risk_level": self._risk_level(prob),
                    "threshold_used": self.threshold,
                    **metrics,
                }
            )

        if not results:
            raise ValueError("No valid results generated from the input prediction dataframe.")

        return results