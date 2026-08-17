import glob
import os
import sys
import warnings

import pickle
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def find_dataset():
    # try common filenames first
    patterns = [
        "Cust_Spend_Data.csv",
        "Cust_Spend_Data(1).csv",
        "Cust_Spend_Data*.csv",
        "*Cust_Spend_Data*.csv",
    ]
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            return found[0]
    # fallback: any csv in workspace
    all_csv = glob.glob("*.csv")
    return all_csv[0] if all_csv else None


def normalize_cols(df):
    # map lowercase stripped names -> original
    mapping = {c.strip().lower(): c for c in df.columns}
    return mapping


def train_pipeline(csv_path=None, save_paths=("customer_spending_model.pkl", "scaler.pkl")):
    """Train and save a model. Returns (model, scaler, feature_names, y_col_name)."""
    warnings.filterwarnings("ignore")
    path = csv_path or find_dataset()
    if not path:
        raise FileNotFoundError("No CSV dataset found in the workspace. Place the CSV and retry.")

    print(f"Using dataset: {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    # basic cleaning
    df = df.copy()
    df = df.drop_duplicates()

    col_map = normalize_cols(df)

    required = [
        "avg_mthly_spend",
        "no_of_visits",
        "apparel_items",
        "fnv_items",
        "staples_items",
    ]

    missing = [r for r in required if r not in col_map]
    if missing:
        raise KeyError(f"The dataset is missing required columns: {missing}. Available: {list(df.columns)}")

    # use original column names
    y_col = col_map["avg_mthly_spend"]
    visits_col = col_map["no_of_visits"]
    apparel_col = col_map["apparel_items"]
    fnv_col = col_map["fnv_items"]
    staples_col = col_map["staples_items"]

    # drop obvious non-feature columns if present
    for drop_candidate in ["cust_id", "name"]:
        if drop_candidate in col_map:
            df = df.drop(columns=[col_map[drop_candidate]])

    # ensure numeric
    num_cols = [visits_col, apparel_col, fnv_col, staples_col, y_col]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # impute numeric missing with median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # feature engineering
    df["Total_Items"] = df[apparel_col] + df[fnv_col] + df[staples_col]
    # avoid division by zero
    df["Items_Per_Visit"] = df["Total_Items"] / df[visits_col].replace({0: np.nan})
    df["Items_Per_Visit"] = df["Items_Per_Visit"].fillna(0)

    features = [
        visits_col,
        apparel_col,
        fnv_col,
        staples_col,
        "Total_Items",
        "Items_Per_Visit",
    ]

    X = df[features]
    y = df[y_col]

    n_samples = len(df)
    if n_samples < 2:
        raise ValueError("Not enough rows to train a model (need at least 2).")

    # determine a safe test size (at least 1 sample)
    test_size = max(1, int(n_samples * 0.2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train and compare several candidate models (as described in the README),
    # then keep whichever generalizes best on the held-out test split.
    # A high-capacity model like RandomForest with default settings tends to
    # overfit badly on very small datasets (this project has ~10 rows), so we
    # can't just assume the "fanciest" model wins -- we have to check.
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

    print("\nModel comparison (test set)")
    for name, res in results.items():
        print(f"- {name}: MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}  R2={res['r2']:.4f}")

    # pick the best model by lowest RMSE on the test split
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best = results[best_name]
    pipeline = best["pipeline"]
    mae, rmse, r2 = best["mae"], best["rmse"], best["r2"]

    print(f"\nSelected best model: {best_name}")
    print(f"- MAE: {mae:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R2: {r2:.4f}")

    # save model, scaler, and full pipeline
    model_path, scaler_path = save_paths
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    pipeline_path = "pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    # save feature list for later use by the app
    features_meta = {
        "features": features,
        "y_col": y_col,
    }
    with open("features.json", "w", encoding="utf-8") as f:
        json.dump(features_meta, f)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "samples": int(n_samples),
        "best_model": best_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "all_models": {
            name: {"mae": float(res["mae"]), "rmse": float(res["rmse"]), "r2": float(res["r2"])}
            for name, res in results.items()
        },
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    return model, scaler, features, y_col, metrics


if __name__ == "__main__":
    train_pipeline()
