# Recovery Prediction Engine

A personal recovery-prediction system trained on 100+ days of my own wearable
biometric data (Apple Watch via HealthKit) — designed and built the way a
production ML system at a wearable company would be: medallion data platform,
dbt transforms, chronological validation, experiment tracking, serving, and
drift monitoring.

Inspired by the recovery-tracking approach pioneered by wearable companies like WHOOP — the architecture and goal, not their proprietary formulas or data.

## What it does

1. **Recovery scoring** — a daily 0–100 recovery label from HRV, resting HR,
   sleep, and respiratory rate vs. rolling personal baselines. This is a
   **self-designed proxy** built on the same four inputs commonly described in published wearable
   recovery methodology — it is not any vendor's proprietary formula,
   and I don't claim it is.
2. **Next-day recovery prediction** — features at day D → recovery at D+1.
   Baselines → RidgeCV → gradient boosting → (PyTorch LSTM), validated with
   strict chronological splits and `TimeSeriesSplit`. No shuffling, no leakage:
   every baseline/z-score uses strictly-prior rows only.
3. **Serving** — the champion model behind a FastAPI service (`/predict`,
   `/predict/latest`, `/health`), Dockerized.
4. **Monitoring** — dbt data tests on every build plus a feature-drift check
   (recent 14 days vs. training baseline) that fails loudly in CI/Dagster.

## Results (honest ones)

86 usable days → chronological 60/20/20 split → held-out test set of 18 days.
Target: next-day recovery score (0–100).

| model | test MAE | test RMSE | direction acc. |
|---|---|---|---|
| Persistence (tomorrow = today) | 21.02 | 24.01 | 0.00 |
| Mean (train mean) | 13.98 | 16.85 | 0.72 |
| **RidgeCV** (α via time-series CV) | **12.81** | **15.64** | **0.83** |
| HistGradientBoosting | 17.79 | 21.46 | 0.78 |
| LSTM (7-day windows, 60 seqs) | 17.40 | 20.31 | 0.50 |

What this actually says:

- **Simple wins at this sample size.** Heavily-regularized linear regression
  beats gradient boosting (which overfits: train MAE 4.3 → test 17.8) and
  barely edges the mean baseline. With n=86, model complexity is the enemy —
  a finding, not a failure.
- **Direction is learnable, magnitude is noisy.** The model gets the *direction*
  of tomorrow's change right ~83% of the time (vs 72% for naive mean-reversion),
  but point predictions carry ±13 MAE. More data and intraday features would be
  the next levers, not fancier architectures.
- The label distribution is sane: 36 green (avg 79), 44 yellow (avg 53),
  12 red (avg 16) across 108 days.
- **Strain phase (2026-10-07):** added 8 strain features — a Bannister-style
  exponential TRIMP from his own workout HR data (explicitly not any vendor's
  strain formula), peak session intensity, 7d/28d rolling loads, acute:chronic
  ratio, training density, strain x sleep-debt interaction, and relative daily
  load. Same-window test: Ridge 19.19 → **17.12**, and `strain_bannister_7d`
  ranked #3 in permutation importance. The champion (12.81, easier test
  window) stands — the newest test window is simply harder for every model,
  so no promotion. Honest, not gamed.

Along the way the pipeline caught a real bug: `sleep_efficiency` arrives from
HealthKit on a 0–100 scale but the label formula treated it as 0–1, pinning the
sleep component at 100 for every night. The dbt build didn't catch it (tests
passed) — inspecting the component distributions did. Fixed, rebuilt, revalidated.

## Production-style architecture

Medallion layout (Bronze → Silver → Gold), modeled on how production wearable platforms run their data
platform (data lake + Snowflake-centered warehouse, dbt transforms, orchestrated
ML pipelines):

| Production (wearable platform) | This project |
|---|---|
| Snowflake warehouse | DuckDB (local columnar warehouse, portable SQL) |
| Data lake (Iceberg) | Bronze Parquet layer, immutable |
| dbt transforms | dbt + dbt-duckdb, 10/10 tests green |
| Airflow/Dagster orchestration | Dagster: daily pipeline + weekly retrain |
| Python scientific stack, PyTorch | Same (sklearn, PyTorch LSTM) |
| SageMaker / model registry | MLflow tracking + registry |
| Model serving | FastAPI + Docker |
| Datadog monitoring | dbt data tests + feature-drift checks |

## Layout

```
bronze/            # raw ingested data (Parquet, immutable)
src/rpe/           # ingest | features | labels | models | serve | monitor
dbt/               # staging models, marts (mart_daily, mart_recovery), tests
orchestration/     # Dagster: daily pipeline + weekly retrain definitions
dashboards/        # generated static recovery dashboard (recovery.html)
models/            # champion artifact (ridge_recovery.pkl)
warehouse.duckdb   # the analytical warehouse (gitignored)
```

## Quickstart

```bash
uv sync --extra transform --extra ml --extra serve   # dbt, sklearn/mlflow, API
make ingest      # HealthKit -> bronze/
make dbt         # Bronze -> Silver -> Gold marts (+ tests)
make train       # baselines -> RidgeCV -> GBM, logs to MLflow
make serve       # FastAPI on :8000  (/predict/latest)
uv run python -m rpe.monitor.drift      # feature-drift check
uv run python -m rpe.monitor.dashboard  # regenerate dashboards/recovery.html
uv run python -m rpe.predict            # tomorrow's forecast, no server needed
uv run python -m rpe.models.retrain     # full loop: ingest -> dbt -> train all
                                        # -> promote champion iff test MAE improves
                                        # -> append to models/accuracy_log.jsonl
```

Serve from Docker (the model artifact and warehouse are gitignored personal
data, so they're mounted at runtime, not baked into the image):

```bash
docker build -t recovery-prediction-engine .
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/warehouse.duckdb:/app/warehouse.duckdb" \
  recovery-prediction-engine
# -> GET /health, /predict/latest, POST /predict
```

PyTorch LSTM (temporal model over 7-day windows — architecture demo at this
sample size; the RidgeCV/GBM comparison is the serious benchmark):

```bash
# CPU-only torch (the default CUDA bundle is ~2GB of dead weight without a GPU)
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu
uv run python -m rpe.models.train_lstm
```

## Data

Apple HealthKit sync from Apple Watch, 2026-06-16 → present (~108 days):
HRV (SDNN), resting HR, respiratory rate, SpO2, sleep stages, wrist temperature,
workouts with HR. Personal data — Bronze/Silver/Gold, the warehouse, and
`models/` are gitignored and never committed.

## Limitations

- The recovery label is my own proxy, not any vendor's ground truth — the models
  predict *my formula's* output, which is only as good as the formula.
- n=86 training days: uncertainty bands are wide; treat every metric as ±large.
- Single subject (me). Nothing here generalizes to other people.
- Strain is a TRIMP-style proxy from workout summaries (17 workouts), not a
  reconstruction of any vendor's proprietary strain.
