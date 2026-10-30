"""Retrain loop: ingest -> dbt -> train all models -> champion promotion.

This is the "efficient re-training & optimization loop": as new Apple Health
data lands, the pipeline refreshes Bronze/Silver/Gold, retrains RidgeCV,
gradient boosting, and the LSTM, logs everything to MLflow, and promotes a
new champion only if it honestly beats the incumbent (test MAE, with a small
noise tolerance). Every run appends to models/accuracy_log.jsonl so accuracy
over time is visible — the goal is a rising curve as n grows, and eventually
graduating from Ridge to more complex models when the data supports it.

Promotion policy (v1):
  - Ridge challenger replaces the champion iff new_test_mae <= old_test_mae + 0.5
  - GBM/LSTM are trained + logged every run but NOT auto-promoted; promoting a
    complex model is a manual review decision until n is large enough that the
    test set is convincing (revisit past ~200 samples).
  - last 3 champion bundles are kept as backups under models/history/

Usage:  uv run python -m rpe.models.retrain
Intended schedule: weekly (cron), e.g. Sunday evening after the week's data.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS = PROJECT_ROOT / "models"
HISTORY = MODELS / "history"
ACC_LOG = MODELS / "accuracy_log.jsonl"
CHAMPION = MODELS / "ridge_recovery.pkl"
PROMOTE_TOLERANCE = 0.5  # MAE points of noise tolerance before rejecting


def run(cmd: list[str], cwd: Path, env_extra: dict | None = None) -> None:
    import os
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main() -> None:
    py = sys.executable  # project venv python
    MODELS.mkdir(exist_ok=True)
    HISTORY.mkdir(exist_ok=True)

    old_mae = joblib.load(CHAMPION)["test_mae"] if CHAMPION.exists() else float("inf")
    old_n = joblib.load(CHAMPION)["n_samples"] if CHAMPION.exists() else 0
    print(f"incumbent champion: test MAE {old_mae:.2f} (n={old_n})")

    # 1. fresh data -> Bronze
    run([py, "-m", "rpe.ingest.healthkit"], cwd=PROJECT_ROOT)
    # 2. dbt Silver/Gold (explicit venv path: cron runs with a clean env)
    dbt_bin = PROJECT_ROOT / ".venv" / "bin" / "dbt"
    run([str(dbt_bin), "build", "--profiles-dir", "."], cwd=PROJECT_ROOT / "dbt")
    # 3. train Ridge + GBM (logs to MLflow, writes new champion bundle)
    if CHAMPION.exists():
        backup = HISTORY / f"ridge_recovery_{date.today()}_mae{old_mae:.2f}.pkl"
        shutil.copy2(CHAMPION, backup)
    run([py, "-m", "rpe.models.train_recovery"], cwd=PROJECT_ROOT)
    # 4. train LSTM challenger (logged, not auto-promoted)
    run([py, "-m", "rpe.models.train_lstm"], cwd=PROJECT_ROOT)

    new = joblib.load(CHAMPION)
    new_mae, new_n = new["test_mae"], new["n_samples"]
    print(f"challenger: test MAE {new_mae:.2f} (n={new_n})")

    if new_mae <= old_mae + PROMOTE_TOLERANCE:
        print(f"PROMOTED (tolerance {PROMOTE_TOLERANCE})")
        promoted = True
    else:
        print(f"REJECTED: worse than incumbent by {new_mae - old_mae:.2f} > {PROMOTE_TOLERANCE}; "
              "restoring previous champion")
        newest_backup = sorted(HISTORY.glob("ridge_recovery_*.pkl"))[-1]
        shutil.copy2(newest_backup, CHAMPION)
        promoted = False

    # keep only last 3 backups
    for old in sorted(HISTORY.glob("ridge_recovery_*.pkl"))[:-3]:
        old.unlink()

    final = joblib.load(CHAMPION)
    summ = json.loads((MODELS / "last_train_summary.json").read_text())
    lstm_summ = json.loads((MODELS / "last_lstm_summary.json").read_text())
    entry = {
        "date": str(date.today()),
        "n_samples": new_n,
        "ridge_test_mae": summ["ridge_test"]["mae"],
        "ridge_test_dir_acc": summ["ridge_test"]["direction_acc"],
        "gbm_test_mae": summ["gbm_test"]["mae"],
        "lstm_test_mae": lstm_summ["lstm_test"]["mae"],
        "persist_test_mae": summ["persist_test_mae"],
        "mean_test_mae": summ["mean_test_mae"],
        "champion_test_mae": round(final["test_mae"], 3),
        "champion_n": final["n_samples"],
        "promoted": promoted,
    }
    with ACC_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print("logged ->", ACC_LOG)
    print("done.")


if __name__ == "__main__":
    main()
