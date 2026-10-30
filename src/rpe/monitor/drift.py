"""Feature/label drift check: recent 14 days vs the training-period baseline.

Compares per-feature means; flags anything shifted by more than 2 training
standard deviations. Also watches the recovery label itself. Exits non-zero
when drift is detected so CI / Dagster can alert on it.

Usage: uv run python -m rpe.monitor.drift [--days 14] [--threshold 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rpe.models.train_recovery import FEATURES, load_frame  # noqa: E402

WATCH = ["hrv_z", "rhr_z", "resp_z", "sleep_debt_h", "strain_trimp",
         "steps", "wrist_temp_c", "sleep_total_h"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--threshold", type=float, default=2.0)
    args = ap.parse_args()

    df = load_frame().sort_values("date_day").reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(n * 0.6)]
    recent = df.iloc[-args.days:]

    report = {"window_days": args.days, "threshold_std": args.threshold,
              "train_end": str(train["date_day"].iloc[-1].date()),
              "recent_start": str(recent["date_day"].iloc[0].date()),
              "flags": []}
    for col in WATCH + ["recovery_score"]:
        mu, sd = train[col].mean(), train[col].std()
        if not sd or pd_isna(sd):
            continue
        shift = (recent[col].mean() - mu) / sd
        flag = abs(shift) > args.threshold
        report["flags"].append({"feature": col, "shift_std": round(float(shift), 2),
                                "drifted": bool(flag)})
    drifted = [f for f in report["flags"] if f["drifted"]]
    print(json.dumps(report, indent=2))
    if drifted:
        print(f"\nDRIFT DETECTED in {len(drifted)} feature(s): "
              + ", ".join(f["feature"] for f in drifted))
        sys.exit(1)
    print("\nno drift detected.")


def pd_isna(x) -> bool:
    return x != x  # NaN-safe without importing pandas here


if __name__ == "__main__":
    main()
