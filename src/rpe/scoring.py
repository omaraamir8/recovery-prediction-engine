"""Shared scoring logic for the recovery champion model.

Used by the FastAPI service (`rpe.serve.app`) and the `rpe.predict` CLI,
so the workout loop can pull next-day recovery forecasts without needing
a running server.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = PROJECT_ROOT / "warehouse.duckdb"
ARTIFACT = PROJECT_ROOT / "models" / "ridge_recovery.pkl"

LAG_BASE = ["hrv_z", "rhr_z", "resp_z", "strain_trimp", "sleep_debt_h"]


class ScoringError(Exception):
    """Raised when a prediction can't be produced (missing data/model)."""


_bundle: dict | None = None


def band(score: float) -> str:
    if score >= 67:
        return "green"
    if score >= 34:
        return "yellow"
    return "red"


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise ScoringError(
                f"model artifact missing at {ARTIFACT}; "
                "train first: uv run python -m rpe.models.train_recovery"
            )
        _bundle = joblib.load(ARTIFACT)
    return _bundle


def features_for(day: date) -> pd.DataFrame:
    """Assemble the model's feature row for `day` (lags pulled from day-1)."""
    bundle = get_bundle()
    day_before = day - timedelta(days=1)
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute(
        "select * from mart_daily where date_day between ? and ? order by date_day",
        [str(day_before), str(day)],
    ).df()
    con.close()
    if len(df) < 2 or str(df["date_day"].iloc[-1].date()) != str(day):
        raise ScoringError(f"no complete features for {day}")
    row = df.iloc[[-1]].copy()
    prev = df.iloc[[-2]]
    for c in LAG_BASE:
        row[f"{c}_lag1"] = prev[c].to_numpy()
    row["dow"] = pd.to_datetime(row["date_day"]).dt.dayofweek
    X = row[bundle["feature_cols"]]
    return X.fillna(bundle["impute_medians"])


def latest_day() -> date:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    latest = con.execute("select max(date_day) from mart_daily").fetchone()[0]
    con.close()
    return pd.to_datetime(latest).date()


def actual_recovery(day: date) -> dict | None:
    """Actual (proxy-label) recovery for `day`, if labeled."""
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    row = con.execute(
        "select recovery_score, recovery_band from mart_recovery where date_day = ?",
        [str(day)],
    ).fetchone()
    con.close()
    if row is None or row[0] is None:
        return None
    return {"recovery": round(float(row[0]), 1), "band": row[1]}


def predict_day(day: date) -> dict:
    bundle = get_bundle()
    X = features_for(day)
    Xs = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns)
    score = float(bundle["model"].predict(Xs)[0])
    score = max(0.0, min(100.0, score))
    return {
        "date": str(day),
        "predicts_recovery_for": str(day + timedelta(days=1)),
        "predicted_recovery": round(score, 1),
        "predicted_band": band(score),
        "model": "RidgeCV",
        "model_trained_on": bundle["trained_on"],
        "model_test_mae": round(bundle["test_mae"], 2),
        "n_samples": bundle["n_samples"],
    }
