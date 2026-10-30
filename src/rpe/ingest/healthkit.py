"""Bronze ingestion: Apple HealthKit -> immutable Parquet tables.

Usage: uv run python -m rpe.ingest.healthkit
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BRONZE = PROJECT_ROOT / "bronze"
START_DATE = "2026-06-16"
END_DATE = date.today().isoformat()


def _run_health_cli(args: list[str]) -> dict:
    proc = subprocess.run(
        ["health-cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"health-cli failed: {proc.stderr[-2000:]}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(proc.stdout)


def ingest_daily_metrics() -> pd.DataFrame:
    payload = _run_health_cli([
        "query", "metrics", "--provider", "healthkit",
        "--start-date", START_DATE, "--end-date", END_DATE,
        "--interval", "daily",
    ])
    records = payload.get("records", [])
    df = pd.DataFrame(records)
    out = BRONZE / "daily_metrics.parquet"
    df.to_parquet(out, index=False)
    print(f"daily_metrics: {len(df)} rows -> {out}")
    return df


def ingest_sessions(category: str) -> pd.DataFrame:
    payload = _run_health_cli([
        "query", "sessions", "--provider", "healthkit",
        "--category", category,
        "--start-date", START_DATE, "--end-date", END_DATE,
    ])
    records = payload.get("records", [])
    df = pd.DataFrame(records)
    out = BRONZE / f"{category}_sessions.parquet"
    df.to_parquet(out, index=False)
    print(f"{category}_sessions: {len(df)} rows -> {out}")
    return df


def main() -> None:
    BRONZE.mkdir(parents=True, exist_ok=True)
    dm = ingest_daily_metrics()
    sl = ingest_sessions("sleep")
    wo = ingest_sessions("workout")

    manifest = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "tables": {
            "daily_metrics": {"rows": len(dm), "columns": list(dm.columns)},
            "sleep_sessions": {"rows": len(sl), "columns": list(sl.columns)},
            "workout_sessions": {"rows": len(wo), "columns": list(wo.columns)},
        },
    }
    (BRONZE / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    main()
