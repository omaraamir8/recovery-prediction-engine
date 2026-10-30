"""Train recovery-prediction models: features at date D -> recovery at D+1.

No leakage by construction: the label at D+1 is never visible in D's features,
and all baselines/z-scores in mart_daily use strictly-prior rows.
Validation is chronological (no shuffling) + sklearn TimeSeriesSplit.

Usage: uv run python -m rpe.models.train_recovery
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:  # pragma: no cover - training runs fine without tracking
    mlflow = None
    HAS_MLFLOW = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WAREHOUSE = PROJECT_ROOT / "warehouse.duckdb"

FEATURES = [
    "hrv_z", "rhr_z", "resp_z",
    "sleep_total_h", "sleep_efficiency", "awakenings",
    "sleep_deep_h", "sleep_rem_h",
    "sleep_debt_h", "sleep_debt_7d",
    "strain_trimp", "workout_min", "workout_count",
    # strain phase 2: intensity-weighted load, rolling accumulations,
    # acute:chronic ratio, density, and load x recovery interactions
    "strain_bannister", "strain_peak_intensity",
    "strain_7d", "strain_bannister_7d", "strain_acwr",
    "training_days_7d", "strain_sleep_interaction",
    "active_rel",
    "steps", "active_kcal", "spo2_avg", "wrist_temp_c",
    "hrv_base_30d", "rhr_base_30d",
]


def load_frame() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    df = con.execute(
        """
        select d.*, r.recovery_score
        from mart_daily d
        join mart_recovery r on r.date_day = d.date_day
        order by d.date_day
        """
    ).df()
    con.close()
    df["date_day"] = pd.to_datetime(df["date_day"])
    # lag features (prior-day physiology)
    for col in ["hrv_z", "rhr_z", "resp_z", "strain_trimp", "sleep_debt_h"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    df["dow"] = df["date_day"].dt.dayofweek
    # target: tomorrow's recovery. Drop days where the watch wasn't worn
    # (no label) — supervised rows need both today's and tomorrow's score.
    df["target"] = df["recovery_score"].shift(-1)
    df["today_recovery"] = df["recovery_score"]
    return df.dropna(subset=["target", "today_recovery"]).reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray) -> dict:
    err = y_true - y_pred
    direction_hit = np.sign(y_true - prev) == np.sign(y_pred - prev)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "direction_acc": float(np.mean(direction_hit)),
    }


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    from contextlib import contextmanager

    @contextmanager
    def maybe_run(name: str):
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
            mlflow.set_experiment("recovery-prediction")
            with mlflow.start_run(run_name=name):
                yield
        else:
            print("(mlflow not installed — skipping experiment tracking)")
            yield

    df = load_frame()
    feature_cols = FEATURES + [f"{c}_lag1" for c in
                               ["hrv_z", "rhr_z", "resp_z", "strain_trimp", "sleep_debt_h"]] + ["dow"]
    X = df[feature_cols]
    y = df["target"].to_numpy()
    prev = df["today_recovery"].to_numpy()
    print(f"samples: {len(df)}  features: {len(feature_cols)}")

    n = len(df)
    i_train, i_valid = int(n * 0.6), int(n * 0.8)
    splits = {"train": (0, i_train), "valid": (i_train, i_valid), "test": (i_valid, n)}

    with maybe_run("gbm-v1"):
        # --- baselines (no training) ---
        persist = {k: metrics(y[a:b], prev[a:b], prev[a:b]) for k, (a, b) in splits.items()}
        mean_pred = np.full(n, y[:i_train].mean())
        mean_m = {k: metrics(y[a:b], mean_pred[a:b], prev[a:b]) for k, (a, b) in splits.items()}
        print("persistence baseline:", {k: round(v["mae"], 2) for k, v in persist.items()})
        print("mean baseline:      ", {k: round(v["mae"], 2) for k, v in mean_m.items()})

        # --- ridge (linear, heavy regularization; alpha via time-series CV on train) ---
        # X has NaNs at lag boundaries -> median-impute for the linear model
        X_lin = X.fillna(X.median())
        scaler = StandardScaler().fit(X_lin.iloc[:i_train])
        Xs = pd.DataFrame(scaler.transform(X_lin), index=X.index, columns=X.columns)
        ridge = RidgeCV(alphas=np.logspace(-1, 3, 9),
                        cv=TimeSeriesSplit(n_splits=3)).fit(Xs.iloc[:i_train], y[:i_train])
        print(f"ridge chosen alpha: {ridge.alpha_:.2f}")
        ridge_m = {}
        for k, (a, b) in splits.items():
            m = metrics(y[a:b], ridge.predict(Xs.iloc[a:b]), prev[a:b])
            ridge_m[k] = m
            print(f"ridge {k}: " + " ".join(f"{kk}={vv:.3f}" for kk, vv in m.items()))

        # --- gradient boosting with time-series CV on train ---
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=5, early_stopping=True, random_state=42,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        cv_mae = []
        for tr, va in tscv.split(X.iloc[:i_train]):
            gbm.fit(X.iloc[tr], y[tr])
            cv_mae.append(np.mean(np.abs(y[va] - gbm.predict(X.iloc[va]))))
        print(f"train CV MAE: {np.mean(cv_mae):.2f} (+/- {np.std(cv_mae):.2f})")

        gbm.fit(X.iloc[:i_valid], y[:i_valid])
        gbm_m = {}
        for k, (a, b) in splits.items():
            m = metrics(y[a:b], gbm.predict(X.iloc[a:b]), prev[a:b])
            gbm_m[k] = m
            print(f"gbm {k}: " + " ".join(f"{kk}={vv:.3f}" for kk, vv in m.items()))

        # --- persist the champion (RidgeCV) for serving ---
        import joblib
        from datetime import date
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        champion_path = models_dir / "ridge_recovery.pkl"
        joblib.dump(
            {
                "model": ridge,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "impute_medians": X.median(),
                "trained_on": str(date.today()),
                "test_mae": ridge_m["test"]["mae"],
                "n_samples": n,
            },
            champion_path,
        )
        print("saved champion ->", champion_path)

        if HAS_MLFLOW:
            mlflow.log_params({"model": "HistGradientBoostingRegressor", **gbm.get_params()})
            mlflow.log_param("ridge_alpha", ridge.alpha_)
            for k, m in persist.items():
                mlflow.log_metrics({f"persist_{k}_{kk}": vv for kk, vv in m.items()})
            for k, m in mean_m.items():
                mlflow.log_metrics({f"mean_{k}_{kk}": vv for kk, vv in m.items()})
            for k, m in gbm_m.items():
                mlflow.log_metrics({f"gbm_{k}_{kk}": vv for kk, vv in m.items()})
            for k, m in ridge_m.items():
                mlflow.log_metrics({f"ridge_{k}_{kk}": vv for kk, vv in m.items()})
            mlflow.log_metric("cv_mae_mean", float(np.mean(cv_mae)))
            # champion artifact: joblib bundle (model+scaler+impute medians),
            # the same file the FastAPI service loads
            mlflow.log_artifact(str(champion_path), artifact_path="champion")

        from sklearn.inspection import permutation_importance
        perm = permutation_importance(gbm, X.iloc[i_train:i_valid], y[i_train:i_valid],
                                      n_repeats=10, random_state=42, scoring="neg_mean_absolute_error")
        importances = pd.Series(perm.importances_mean, index=feature_cols)
        print("\ntop features (permutation importance, valid split):\n"
              + importances.sort_values(ascending=False).head(8).to_string())

        # machine-readable summary for the retrain loop / accuracy log
        import json
        summary = {
            "date": str(date.today()),
            "n_samples": n,
            "ridge_test": {k: round(float(v), 3) for k, v in ridge_m["test"].items()},
            "gbm_test": {k: round(float(v), 3) for k, v in gbm_m["test"].items()},
            "persist_test_mae": round(float(persist["test"]["mae"]), 3),
            "mean_test_mae": round(float(mean_m["test"]["mae"]), 3),
        }
        (models_dir / "last_train_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
