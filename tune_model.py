import glob
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_dataset():
    patterns = ["Cust_Spend_Data.csv", "Cust_Spend_Data(1).csv", "Cust_Spend_Data*.csv", "*Cust_Spend_Data*.csv"]
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            return found[0]
    all_csv = glob.glob("*.csv")
    return all_csv[0] if all_csv else None


def load_and_prepare(path):
    df = pd.read_csv(path).drop_duplicates()
    # normalize columns
    col_map = {c.strip().lower(): c for c in df.columns}
    required = ["avg_mthly_spend", "no_of_visits", "apparel_items", "fnv_items", "staples_items"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    y_col = col_map["avg_mthly_spend"]
    visits = col_map["no_of_visits"]
    app = col_map["apparel_items"]
    fnv = col_map["fnv_items"]
    stp = col_map["staples_items"]

    for c in [visits, app, fnv, stp, y_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[[visits, app, fnv, stp, y_col]] = df[[visits, app, fnv, stp, y_col]].fillna(df[[visits, app, fnv, stp, y_col]].median())

    df["Total_Items"] = df[app] + df[fnv] + df[stp]
    df["Items_Per_Visit"] = df["Total_Items"] / df[visits].replace({0: np.nan})
    df["Items_Per_Visit"] = df["Items_Per_Visit"].fillna(0)

    features = [visits, app, fnv, stp, "Total_Items", "Items_Per_Visit"]
    X = df[features]
    y = df[y_col]
    return X, y, features


def tune():
    warnings.filterwarnings("ignore")
    path = find_dataset()
    if not path:
        print("No dataset found")
        return
    X, y, features = load_and_prepare(path)
    n = len(X)
    if n < 4:
        print("Not enough samples for tuning")
        return

    test_size = max(1, int(n * 0.2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42)),
    ])

    param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 5, 10, 20],
        'model__max_features': ['sqrt', 'log2', None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
    print("Starting RandomizedSearchCV...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)

    best = search.best_estimator_
    # evaluate on test set
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'samples': int(n),
        'best_model': 'RandomForest (tuned)',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
    }

    print('Tuning finished. Tuned RandomForest test metrics:')
    print(metrics)

    # Safety check: this script only tunes RandomForest, but run_pipeline.py
    # may have already deployed a different, better-performing model (e.g.
    # LinearRegression on a small dataset). Only overwrite the live
    # pipeline/model/scaler files if the tuned model actually beats what's
    # currently deployed, so tuning can never silently regress production.
    current_rmse = None
    if os.path.exists('metrics.json'):
        try:
            with open('metrics.json', 'r', encoding='utf-8') as f:
                current_rmse = json.load(f).get('rmse')
        except Exception:
            current_rmse = None

    if current_rmse is not None and rmse >= current_rmse:
        print(f"\nTuned RandomForest (RMSE={rmse:.2f}) did not beat the currently "
              f"deployed model (RMSE={current_rmse:.2f}). Keeping the existing "
              f"pipeline.pkl / metrics.json unchanged.")
        return

    # save pipeline, metrics, AND the standalone model/scaler files so they
    # never fall out of sync with the tuned pipeline (app.py falls back to
    # these two files if pipeline.pkl is ever missing).
    with open('pipeline.pkl', 'wb') as f:
        pickle.dump(best, f)
    with open('customer_spending_model.pkl', 'wb') as f:
        pickle.dump(best.named_steps['model'], f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(best.named_steps['scaler'], f)
    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    print('\nTuned model improved on the deployed model -- saved as the new pipeline.')


if __name__ == '__main__':
    tune()
