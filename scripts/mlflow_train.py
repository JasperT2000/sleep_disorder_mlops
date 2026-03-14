import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def safe_roc_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return None


def collect_metrics(model_name, y_test, y_pred, y_prob=None):
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_positive": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision_positive": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": safe_roc_auc(y_test, y_prob) if y_prob is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def build_models(random_state: int):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            class_weight="balanced",
        ),
        "lightgbm": LGBMClassifier(
            random_state=random_state,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--experiment_name", default="sleep_disorder_all_features")
    parser.add_argument("--registered_model_name", default="sleep_disorder_predictor")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--primary_metric", default="f1_positive")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.csv)

    target = "disorder"
    drop_cols = {"id", "disorder", "insomnia", "slp_apnea"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)

    X = df[feature_cols].copy()
    y = df[target].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    # Keep preprocessing order same as your existing deployed artifacts
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=2)

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train_proc = imputer.fit_transform(X_train_scaled)
    X_test_proc = imputer.transform(X_test_scaled)

    ros = RandomOverSampler(random_state=args.random_state)
    X_train_res, y_train_res = ros.fit_resample(X_train_proc, y_train)

    models = build_models(args.random_state)

    all_results = []
    best_result = None
    best_model = None
    best_model_name = None
    best_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train_res, y_train_res)

            y_prob = model.predict_proba(X_test_proc)[:, 1]
            y_pred = (y_prob >= args.threshold).astype(int)

            outputs = collect_metrics(model_name, y_test, y_pred, y_prob)

            mlflow.log_param("model_type", model.__class__.__name__)
            mlflow.log_param("target", target)
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_param("threshold", args.threshold)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("tracking_uri", tracking_uri)
            mlflow.log_param("preprocessing_order", "scaler_then_imputer")

            for metric_name in [
                "accuracy",
                "balanced_accuracy",
                "f1_positive",
                "recall_positive",
                "precision_positive",
            ]:
                mlflow.log_metric(metric_name, outputs[metric_name])

            if outputs["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", outputs["roc_auc"])

            mlflow.log_dict(outputs, "results.json")

            if hasattr(model, "feature_importances_"):
                fi_df = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                fi_path = out_dir / f"{model_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(str(fi_path))

            model_info = mlflow.sklearn.log_model(model, artifact_path="model")

            result_row = {
                "run_id": mlflow.active_run().info.run_id,
                "model_name": model_name,
                **outputs,
                "model_uri": model_info.model_uri,
            }
            all_results.append(result_row)

            current_score = outputs.get(args.primary_metric, -1)
            best_score = best_result.get(args.primary_metric, -1) if best_result else -1

            if best_result is None or current_score > best_score:
                best_result = result_row
                best_model = model
                best_model_name = model_name
                best_run_id = mlflow.active_run().info.run_id

    if best_model is None:
        raise RuntimeError("No best model was selected.")

    # Save deployable artifacts for the best model only
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(imputer, out_dir / "imputer.joblib")
    joblib.dump(best_model, out_dir / "best_model.joblib")

    leaderboard_df = pd.DataFrame(all_results).sort_values(
        by=args.primary_metric, ascending=False
    )
    leaderboard_path = out_dir / "model_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    metadata = {
        "target": target,
        "features": feature_cols,
        "best_model_name": best_model_name,
        "best_run_id": best_run_id,
        "threshold": args.threshold,
        "primary_metric": args.primary_metric,
        "preprocessing_order": ["scaler", "imputer"],
        "leaderboard": leaderboard_df[
            ["model_name", "accuracy", "balanced_accuracy", "f1_positive", "recall_positive", "precision_positive", "roc_auc"]
        ].to_dict(orient="records"),
    }

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Log final selection summary in its own MLflow run
    with mlflow.start_run(run_name="best_model_summary"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selected_run_id", best_run_id)
        mlflow.log_param("primary_metric", args.primary_metric)
        mlflow.log_param("threshold", args.threshold)

        mlflow.log_artifact(str(out_dir / "scaler.joblib"))
        mlflow.log_artifact(str(out_dir / "imputer.joblib"))
        mlflow.log_artifact(str(out_dir / "best_model.joblib"))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(leaderboard_path))

        best_model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(
            model_uri=best_model_uri,
            name=args.registered_model_name,
        )

        client = MlflowClient()
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="selected_model",
            value=best_model_name,
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="threshold",
            value=str(args.threshold),
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="preprocessing_order",
            value="scaler_then_imputer",
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registered.version,
            key="primary_metric",
            value=args.primary_metric,
        )

    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(metadata, indent=2))
    print(f"Registered best model: {args.registered_model_name}, version: {registered.version}")


if __name__ == "__main__":
    main()