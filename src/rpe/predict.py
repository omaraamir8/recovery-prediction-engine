"""CLI: next-day recovery forecast, no server needed.

Usage:
    uv run python -m rpe.predict              # latest complete day -> tomorrow
    uv run python -m rpe.predict --date 2026-09-30
    uv run python -m rpe.predict --json        # machine-readable output

This is what the workout loop calls: after Omar logs a workout, pull the
forecast for tomorrow's recovery and factor it into training feedback.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from rpe.scoring import (
    ScoringError,
    actual_recovery,
    band,
    latest_day,
    predict_day,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="next-day recovery forecast")
    ap.add_argument("--date", help="base date YYYY-MM-DD (default: latest complete day)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    day = pd.to_datetime(args.date).date() if args.date else latest_day()
    try:
        pred = predict_day(day)
    except ScoringError as e:
        raise SystemExit(f"cannot predict: {e}")

    actual = actual_recovery(day)
    out = dict(pred)
    out["actual_recovery_today"] = actual["recovery"] if actual else None
    out["actual_band_today"] = actual["band"] if actual else None
    if actual:
        delta = out["predicted_recovery"] - actual["recovery"]
        out["predicted_direction"] = "up" if delta > 0 else "down" if delta < 0 else "flat"
        out["predicted_delta"] = round(delta, 1)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    tomorrow = date.fromisoformat(out["predicts_recovery_for"])
    print(f"today {out['date']}: actual recovery "
          f"{out['actual_recovery_today']} ({out['actual_band_today']})"
          if actual else f"today {out['date']}: (unlabeled)")
    print(f"tomorrow {tomorrow}: predicted {out['predicted_recovery']} "
          f"({out['predicted_band']}) — trending {out.get('predicted_direction', '?')} "
          f"({out.get('predicted_delta', '?'):+} pts)")
    print(f"model: {out['model']} trained {out['model_trained_on']} "
          f"(test MAE {out['model_test_mae']}, n={out['n_samples']})")


if __name__ == "__main__":
    main()
