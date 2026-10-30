"""Dagster orchestration for the recovery-prediction-engine pipeline.

Assets:
    bronze_parquet      raw HealthKit pulls -> bronze/*.parquet (immutable)
    dbt_marts           dbt build: staging -> mart_daily / mart_recovery
    champion_model      train_recovery.py -> models/ridge_recovery.pkl
    drift_report        rpe.monitor.drift -> fails loudly on drift

Schedule: daily 06:00 local — ingest yesterday, rebuild marts, retrain weekly,
drift-check daily.

Run:  uv sync --extra orchestration && dagster dev -m orchestration.definitions
(Not installed in the base env; dagster is a heavy optional extra.)

NOTE: do NOT add `from __future__ import annotations` to this module.
Dagster validates the asset `context` parameter against the *raw* annotation
object, so a stringified "AssetExecutionContext" fails its type check.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dagster import (
        AssetExecutionContext,
        Definitions,
        ScheduleDefinition,
        asset,
        define_asset_job,
    )
    HAS_DAGSTER = True
except ImportError:  # pragma: no cover
    HAS_DAGSTER = False


def _run(cmd: list[str], cwd: Path, context: "AssetExecutionContext") -> None:
    context.log.info("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=sys.stdout, stderr=sys.stderr)


if HAS_DAGSTER:
    VENV_PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    @asset(group_name="ingest")
    def bronze_parquet(context: AssetExecutionContext) -> None:
        """Pull HealthKit -> immutable Bronze Parquet + manifest."""
        _run([VENV_PY, "-m", "rpe.ingest.healthkit"], PROJECT_ROOT, context)

    @asset(deps=[bronze_parquet], group_name="transform")
    def dbt_marts(context: AssetExecutionContext) -> None:
        """dbt build: Bronze -> Silver staging -> Gold marts + tests."""
        _run([VENV_PY, "-m", "dbt.cli.main", "build", "--profiles-dir", "."],
             PROJECT_ROOT / "dbt", context)

    @asset(deps=[dbt_marts], group_name="model")
    def champion_model(context: AssetExecutionContext) -> None:
        """Train baselines/Ridge/GBM -> models/ridge_recovery.pkl."""
        _run([VENV_PY, "-m", "rpe.models.train_recovery"], PROJECT_ROOT, context)

    @asset(deps=[dbt_marts], group_name="monitor")
    def drift_report(context: AssetExecutionContext) -> None:
        """Fail loudly if recent features drifted off the training baseline."""
        _run([VENV_PY, "-m", "rpe.monitor.drift", "--days", "14"],
             PROJECT_ROOT, context)

    daily_job = define_asset_job(
        name="daily_recovery_pipeline",
        selection=[bronze_parquet, dbt_marts, drift_report],
    )
    weekly_retrain_job = define_asset_job(
        name="weekly_retrain",
        selection=[bronze_parquet, dbt_marts, champion_model, drift_report],
    )

    defs = Definitions(
        assets=[bronze_parquet, dbt_marts, champion_model, drift_report],
        jobs=[daily_job, weekly_retrain_job],
        schedules=[
            ScheduleDefinition(job=daily_job, cron_schedule="0 6 * * *"),
            ScheduleDefinition(job=weekly_retrain_job, cron_schedule="0 6 * * 0"),
        ],
    )
