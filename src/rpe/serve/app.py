"""Recovery prediction API.

Serves the champion model (RidgeCV, models/ridge_recovery.pkl) over HTTP.
Scoring logic lives in rpe.scoring (shared with the `rpe.predict` CLI).

Endpoints:
    GET  /health            service + model status
    GET  /predict/latest    next-day recovery prediction for the most recent
                           complete day in the warehouse
    POST /predict           {"date": "2026-09-30"} -> prediction for date+1

Run:  uv run uvicorn rpe.serve.app:app --reload   (needs fastapi, uvicorn)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rpe.scoring import (
    ARTIFACT,
    ScoringError,
    get_bundle,
    latest_day,
    predict_day,
)

app = FastAPI(title="recovery-prediction-engine", version="0.1.0")


class PredictRequest(BaseModel):
    date: date


def _predict(day: date) -> dict:
    try:
        return predict_day(day)
    except ScoringError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health() -> dict:
    ok = ARTIFACT.exists()
    return {"status": "ok" if ok else "degraded",
            "model_loaded": ok,
            "model_test_mae": round(get_bundle()["test_mae"], 2) if ok else None}


@app.get("/predict/latest")
def predict_latest() -> dict:
    return _predict(latest_day())


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    return _predict(req.date)
