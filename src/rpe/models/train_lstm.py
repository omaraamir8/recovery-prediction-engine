"""Temporal recovery model: LSTM over 14-day feature sequences -> next-day recovery.

Same supervised task as train_recovery.py, but the model sees a 14-day window
of daily physiology (HRV/RHR z-scores, sleep debt, strain, ...) and learns
temporal dynamics end-to-end instead of relying on hand-built lag features.

Honest framing: with ~70 usable sequences this is an architecture demo, not a
production-grade model. The gradient-boosting benchmark in train_recovery.py
remains the serious baseline. Kept small on purpose: 1-layer LSTM, dropout,
early stopping, weight decay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rpe.models.train_recovery import FEATURES, load_frame, metrics  # noqa: E402

SEQ_LEN = 7  # shorter windows survive the scattered unworn-day gaps

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """14-day windows of features -> target of the window's last day.

    Only keeps windows over contiguous calendar days (no gaps from unworn days).
    """
    X_list, y_list, prev_list = [], [], []
    dates = df["date_day"].to_numpy()
    feats = df[FEATURES].to_numpy(dtype=np.float32)
    for i in range(len(df) - seq_len + 1):
        window_dates = dates[i:i + seq_len]
        if (window_dates[-1] - window_dates[0]).astype("timedelta64[D]").astype(int) != seq_len - 1:
            continue
        X_list.append(feats[i:i + seq_len])
        j = i + seq_len - 1  # last day of the window; target[j] = recovery(j+1)
        y_list.append(df["target"].iloc[j])
        prev_list.append(df["today_recovery"].iloc[j])
    X = np.stack(X_list).astype(np.float32)
    return X, np.array(y_list, dtype=np.float32), np.array(prev_list, dtype=np.float32)


class RecoveryLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(self.drop(h_n[-1])).squeeze(-1)


def main() -> None:
    if not HAS_TORCH:
        sys.exit("torch is not installed yet — run: uv sync --extra ml")
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_frame()
    # median-impute feature NaNs (unlike GBM, the LSTM can't ingest NaN)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    X, y, prev = build_sequences(df)
    print(f"sequences: {len(X)}  seq_len={SEQ_LEN}  features={X.shape[2]}")

    n = len(X)
    i_train, i_valid = int(n * 0.6), int(n * 0.8)
    splits = {"train": (0, i_train), "valid": (i_train, i_valid), "test": (i_valid, n)}

    # standardize with train-window stats only (no leakage)
    mu = X[:i_train].mean(axis=(0, 1), keepdims=True)
    sd = X[:i_train].std(axis=(0, 1), keepdims=True) + 1e-8
    Xs = (X - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(torch.from_numpy(Xs[:i_train]), torch.from_numpy(y[:i_train]))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    X_valid = torch.from_numpy(Xs[i_train:i_valid]).to(device)
    y_valid = torch.from_numpy(y[i_train:i_valid]).to(device)

    model = RecoveryLSTM(X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_mae, best_state, patience, wait = float("inf"), None, 25, 0
    for epoch in range(500):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_mae = float(torch.mean(torch.abs(model(X_valid) - y_valid)))
        if va_mae < best_mae - 1e-4:
            best_mae, best_state, wait = va_mae, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at epoch {epoch}, best valid MAE={best_mae:.2f}")
                break
    if best_state is None:  # never improved; keep final weights, say so honestly
        print("warning: validation never improved — reporting final-epoch weights")
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    results = {}
    model.eval()
    with torch.no_grad():
        for k, (a, b) in splits.items():
            preds = model(torch.from_numpy(Xs[a:b]).to(device)).cpu().numpy()
            m = metrics(y[a:b], preds, prev[a:b])
            results[k] = m
            print(f"lstm {k}: " + " ".join(f"{kk}={vv:.3f}" for kk, vv in m.items()))

    # same baselines, on the same sequence test rows, for a fair comparison
    persist_test = metrics(y[i_valid:], prev[i_valid:], prev[i_valid:])
    mean_test = metrics(y[i_valid:], np.full(n - i_valid, y[:i_train].mean()), prev[i_valid:])
    print(f"persistence test: MAE={persist_test['mae']:.2f}")
    print(f"mean baseline test: MAE={mean_test['mae']:.2f}")

    out = PROJECT_ROOT / "models" / "lstm_recovery.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                "features": FEATURES, "seq_len": SEQ_LEN}, out)
    print("saved", out)

    # machine-readable summary for the retrain loop / accuracy log
    import json
    from datetime import date
    (PROJECT_ROOT / "models" / "last_lstm_summary.json").write_text(json.dumps({
        "date": str(date.today()),
        "n_sequences": n,
        "lstm_test": {k: round(float(v), 3) for k, v in results["test"].items()},
    }, indent=2))

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment("recovery-prediction")
        with mlflow.start_run(run_name="lstm-v1"):
            mlflow.log_params({"seq_len": SEQ_LEN, "hidden": 64, "dropout": 0.3,
                               "lr": 1e-3, "weight_decay": 1e-4, "n_sequences": n})
            for k, m in results.items():
                mlflow.log_metrics({f"lstm_{k}_{kk}": vv for kk, vv in m.items()})
            mlflow.log_metrics({"persist_test_mae": persist_test["mae"],
                                "mean_test_mae": mean_test["mae"]})
            # log the raw state-dict bundle (same file the API could load);
            # skip mlflow.pytorch.log_model (pt2 tracing needs input examples)
            mlflow.log_artifact(str(out), artifact_path="lstm")


if __name__ == "__main__":
    main()
