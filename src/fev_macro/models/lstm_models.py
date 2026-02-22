from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from .base import (
    BaseModel,
    get_history_by_item,
    get_item_order,
    get_task_horizon,
    get_task_id_column,
    to_prediction_dataset,
)
from .random_forest import _drift_fallback

_TORCH_IMPORT_ERROR = (
    "torch is required for lstm_univariate/lstm_multivariate. "
    "Install torch>=2.2.0 to use these models."
)


def _import_torch_or_raise() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(_TORCH_IMPORT_ERROR) from exc

    return torch, nn, DataLoader, TensorDataset


def _set_deterministic_seed(torch_mod: Any, seed: int) -> None:
    np.random.seed(int(seed))
    torch_mod.manual_seed(int(seed))
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(int(seed))

    if hasattr(torch_mod.backends, "cudnn"):
        torch_mod.backends.cudnn.benchmark = False
        torch_mod.backends.cudnn.deterministic = True

    try:
        torch_mod.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _resolve_device(torch_mod: Any, requested: str) -> Any:
    request = str(requested).strip().lower()
    if request == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")

    if request.startswith("cuda") and not torch_mod.cuda.is_available():
        return torch_mod.device("cpu")

    try:
        return torch_mod.device(request)
    except Exception:
        return torch_mod.device("cpu")


def _rows_by_item(dataset: Dataset, id_col: str) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for rec in dataset:
        key = rec[id_col] if id_col in rec else "__single_series__"
        rows[key] = rec
    return rows


def _to_numeric_array(value: Any, expected_len: int) -> np.ndarray:
    n = max(int(expected_len), 0)
    if n == 0:
        return np.empty(0, dtype=float)

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        s = pd.to_numeric(pd.Series(value), errors="coerce")
    elif value is None:
        s = pd.Series([np.nan] * n, dtype=float)
    else:
        s = pd.to_numeric(pd.Series([value] * n), errors="coerce")

    if len(s) < n:
        if len(s) == 0:
            s = pd.Series([np.nan] * n, dtype=float)
        else:
            s = pd.concat([s, pd.Series([s.iloc[-1]] * (n - len(s)), dtype=float)], ignore_index=True)
    elif len(s) > n:
        s = s.iloc[:n]

    return s.astype(float).ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def _sanitize_array(values: np.ndarray) -> np.ndarray:
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.astype(float).ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def _fallback_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    if y.size == 1:
        return np.repeat(float(y[-1]), horizon).astype(float)
    return _drift_fallback(y.astype(float), horizon=horizon)


def _safe_mean_std(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis)

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean, std


def _build_univariate_samples(y: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(y) - lookback - horizon + 1
    if sample_count <= 0:
        return np.empty((0, lookback, 1), dtype=float), np.empty((0, horizon), dtype=float)

    X = np.empty((sample_count, lookback, 1), dtype=float)
    Y = np.empty((sample_count, horizon), dtype=float)

    for i, t in enumerate(range(lookback, len(y) - horizon + 1)):
        X[i, :, 0] = y[t - lookback : t]
        Y[i, :] = y[t : t + horizon]

    return X, Y


def _build_multivariate_samples(
    features: np.ndarray,
    target: np.ndarray,
    known_covariates: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    sample_count = len(target) - lookback - horizon + 1
    if sample_count <= 0:
        empty_x = np.empty((0, lookback, features.shape[1]), dtype=float)
        empty_y = np.empty((0, horizon), dtype=float)
        if known_covariates.shape[1] == 0:
            return empty_x, empty_y, None
        empty_k = np.empty((0, horizon * known_covariates.shape[1]), dtype=float)
        return empty_x, empty_y, empty_k

    X = np.empty((sample_count, lookback, features.shape[1]), dtype=float)
    Y = np.empty((sample_count, horizon), dtype=float)

    known_width = horizon * known_covariates.shape[1]
    K = np.empty((sample_count, known_width), dtype=float) if known_width > 0 else None

    for i, t in enumerate(range(lookback, len(target) - horizon + 1)):
        X[i, :, :] = features[t - lookback : t, :]
        Y[i, :] = target[t : t + horizon]
        if K is not None:
            K[i, :] = known_covariates[t : t + horizon, :].reshape(-1)

    return X, Y, K


_LSTM_FORECASTER_CLASS: type[Any] | None = None


def _get_lstm_forecaster_class(torch_mod: Any, nn_mod: Any) -> type[Any]:
    global _LSTM_FORECASTER_CLASS

    if _LSTM_FORECASTER_CLASS is None:

        class _LSTMForecaster(nn_mod.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
                horizon: int,
                known_future_size: int,
            ) -> None:
                super().__init__()
                lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
                self.encoder = nn_mod.LSTM(
                    input_size=int(input_size),
                    hidden_size=int(hidden_size),
                    num_layers=int(num_layers),
                    dropout=lstm_dropout,
                    batch_first=True,
                )
                self.hidden_dropout = nn_mod.Dropout(p=max(0.0, float(dropout)))
                self.known_future_size = int(known_future_size)
                self.head = nn_mod.Linear(int(hidden_size) + self.known_future_size, int(horizon))

            def forward(self, x_seq: Any, known_future: Any | None = None) -> Any:
                _, (h_n, _) = self.encoder(x_seq)
                encoded = self.hidden_dropout(h_n[-1])

                if self.known_future_size > 0:
                    if known_future is None:
                        known_future = encoded.new_zeros((encoded.shape[0], self.known_future_size))
                    encoded = torch_mod.cat([encoded, known_future], dim=1)

                return self.head(encoded)

        _LSTM_FORECASTER_CLASS = _LSTMForecaster

    return _LSTM_FORECASTER_CLASS


class _BaseLSTMModel(BaseModel):
    def __init__(
        self,
        *,
        name: str,
        seed: int,
        lookback: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        lr: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        min_train_samples: int,
        weight_decay: float,
        grad_clip: float,
        device: str,
    ) -> None:
        super().__init__(name=name)
        self.seed = int(seed)
        self.lookback = max(1, int(lookback))
        self.hidden_size = max(1, int(hidden_size))
        self.num_layers = max(1, int(num_layers))
        self.dropout = max(0.0, float(dropout))
        self.lr = float(lr)
        self.batch_size = max(1, int(batch_size))
        self.max_epochs = max(1, int(max_epochs))
        self.patience = max(1, int(patience))
        self.min_train_samples = max(1, int(min_train_samples))
        self.weight_decay = max(0.0, float(weight_decay))
        self.grad_clip = max(0.0, float(grad_clip))
        self.device = str(device)

        self._torch, self._nn, self._DataLoader, self._TensorDataset = _import_torch_or_raise()

    def _fit_and_forecast_one(
        self,
        *,
        X_samples: np.ndarray,
        Y_samples: np.ndarray,
        X_last: np.ndarray,
        known_future_samples: np.ndarray | None,
        known_future_inference: np.ndarray | None,
        horizon: int,
        y_fallback: np.ndarray,
        seed_offset: int,
    ) -> np.ndarray:
        horizon = int(horizon)
        fallback = _fallback_forecast(y_fallback, horizon=horizon)

        if X_samples.ndim != 3 or Y_samples.ndim != 2:
            return fallback

        sample_count = X_samples.shape[0]
        if sample_count < max(2, self.min_train_samples):
            return fallback

        val_size = max(1, int(np.ceil(0.2 * sample_count)))
        train_size = sample_count - val_size
        if train_size < 1:
            return fallback

        X_train_flat = X_samples[:train_size].reshape(-1, X_samples.shape[-1])
        X_mean, X_std = _safe_mean_std(X_train_flat, axis=0)

        y_train_flat = Y_samples[:train_size].reshape(-1)
        y_mean_arr, y_std_arr = _safe_mean_std(y_train_flat, axis=None)
        y_mean = float(np.asarray(y_mean_arr).reshape(-1)[0])
        y_std = float(np.asarray(y_std_arr).reshape(-1)[0])

        X_scaled = (X_samples - X_mean.reshape(1, 1, -1)) / X_std.reshape(1, 1, -1)
        X_last_scaled = (X_last - X_mean.reshape(1, -1)) / X_std.reshape(1, -1)
        Y_scaled = (Y_samples - y_mean) / y_std

        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_last_scaled = np.nan_to_num(X_last_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Y_scaled = np.nan_to_num(Y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if known_future_samples is None:
            K_scaled = np.zeros((sample_count, 0), dtype=float)
            K_inference_scaled = np.zeros((1, 0), dtype=float)
        else:
            known_future_samples = np.asarray(known_future_samples, dtype=float)
            if known_future_samples.ndim != 2 or known_future_samples.shape[0] != sample_count:
                return fallback

            K_train = known_future_samples[:train_size]
            K_mean, K_std = _safe_mean_std(K_train, axis=0)
            K_scaled = (known_future_samples - K_mean.reshape(1, -1)) / K_std.reshape(1, -1)

            if known_future_inference is None:
                known_future_inference = np.zeros(known_future_samples.shape[1], dtype=float)

            known_future_inference = np.asarray(known_future_inference, dtype=float).reshape(-1)
            if known_future_inference.size != known_future_samples.shape[1]:
                known_future_inference = _to_numeric_array(
                    known_future_inference,
                    expected_len=known_future_samples.shape[1],
                )

            K_inference_scaled = (known_future_inference.reshape(1, -1) - K_mean.reshape(1, -1)) / K_std.reshape(1, -1)
            K_scaled = np.nan_to_num(K_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            K_inference_scaled = np.nan_to_num(K_inference_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            torch_mod = self._torch
            nn_mod = self._nn
            run_seed = self.seed + int(seed_offset)
            _set_deterministic_seed(torch_mod, run_seed)

            device = _resolve_device(torch_mod, self.device)
            model_cls = _get_lstm_forecaster_class(torch_mod=torch_mod, nn_mod=nn_mod)
            model = model_cls(
                input_size=int(X_samples.shape[-1]),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                horizon=horizon,
                known_future_size=int(K_scaled.shape[1]),
            ).to(device)

            optimizer = torch_mod.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            criterion = nn_mod.MSELoss()

            x_train = torch_mod.as_tensor(X_scaled[:train_size], dtype=torch_mod.float32)
            y_train = torch_mod.as_tensor(Y_scaled[:train_size], dtype=torch_mod.float32)
            k_train = torch_mod.as_tensor(K_scaled[:train_size], dtype=torch_mod.float32)

            x_val = torch_mod.as_tensor(X_scaled[train_size:], dtype=torch_mod.float32)
            y_val = torch_mod.as_tensor(Y_scaled[train_size:], dtype=torch_mod.float32)
            k_val = torch_mod.as_tensor(K_scaled[train_size:], dtype=torch_mod.float32)

            generator = torch_mod.Generator()
            generator.manual_seed(run_seed)

            train_loader = self._DataLoader(
                self._TensorDataset(x_train, y_train, k_train),
                batch_size=min(self.batch_size, train_size),
                shuffle=True,
                generator=generator,
            )
            val_loader = self._DataLoader(
                self._TensorDataset(x_val, y_val, k_val),
                batch_size=min(self.batch_size, val_size),
                shuffle=False,
            )

            best_state: dict[str, Any] | None = None
            best_val_loss = float("inf")
            epochs_without_improve = 0

            for _ in range(self.max_epochs):
                model.train()
                for x_batch, y_batch, k_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    k_batch = k_batch.to(device)

                    optimizer.zero_grad()
                    y_hat = model(x_batch, k_batch)
                    loss = criterion(y_hat, y_batch)
                    loss.backward()

                    if self.grad_clip > 0.0:
                        torch_mod.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                    optimizer.step()

                model.eval()
                val_losses: list[float] = []
                with torch_mod.no_grad():
                    for x_batch, y_batch, k_batch in val_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        k_batch = k_batch.to(device)
                        y_hat = model(x_batch, k_batch)
                        loss = criterion(y_hat, y_batch)
                        val_losses.append(float(loss.detach().cpu().item()))

                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                if not np.isfinite(val_loss):
                    val_loss = float("inf")

                if val_loss + 1e-10 < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= self.patience:
                        break

            if best_state is None:
                return fallback

            model.load_state_dict(best_state)
            model.eval()

            x_last_tensor = torch_mod.as_tensor(X_last_scaled.reshape(1, *X_last_scaled.shape), dtype=torch_mod.float32).to(device)
            k_last_tensor = torch_mod.as_tensor(K_inference_scaled, dtype=torch_mod.float32).to(device)

            with torch_mod.no_grad():
                y_hat_scaled = model(x_last_tensor, k_last_tensor).detach().cpu().numpy().reshape(-1)

            forecast = y_hat_scaled * y_std + y_mean
            forecast = np.asarray(forecast, dtype=float).reshape(-1)
            if forecast.size != horizon or not np.isfinite(forecast).all():
                return fallback
            return forecast
        except Exception:
            return fallback


class LSTMUnivariateModel(_BaseLSTMModel):
    def __init__(
        self,
        seed: int = 0,
        lookback: int = 16,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        min_train_samples: int = 24,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            name="lstm_univariate",
            seed=seed,
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_train_samples=min_train_samples,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            device=device,
        )

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        preds: dict[Any, np.ndarray] = {}
        for idx, item_id in enumerate(item_order):
            if item_id not in history:
                raise ValueError(f"No history available for item_id={item_id}")

            y = _sanitize_array(np.asarray(history[item_id], dtype=float))
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            X_samples, Y_samples = _build_univariate_samples(y=y, lookback=self.lookback, horizon=horizon)
            if X_samples.shape[0] < self.min_train_samples:
                preds[item_id] = _fallback_forecast(y, horizon=horizon)
                continue

            X_last = y[-self.lookback :].reshape(self.lookback, 1)

            preds[item_id] = self._fit_and_forecast_one(
                X_samples=X_samples,
                Y_samples=Y_samples,
                X_last=X_last,
                known_future_samples=None,
                known_future_inference=None,
                horizon=horizon,
                y_fallback=y,
                seed_offset=idx,
            )

        return to_prediction_dataset(preds, item_order)


class LSTMMultivariateModel(_BaseLSTMModel):
    def __init__(
        self,
        seed: int = 0,
        lookback: int = 16,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        min_train_samples: int = 24,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            name="lstm_multivariate",
            seed=seed,
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_train_samples=min_train_samples,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            device=device,
        )

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        id_col = get_task_id_column(task)
        past_rows = _rows_by_item(dataset=past_data, id_col=id_col)
        future_rows = _rows_by_item(dataset=future_data, id_col=id_col)

        past_covars = [c for c in list(getattr(task, "past_dynamic_columns", [])) if c in past_data.column_names]
        known_covars = [
            c
            for c in list(getattr(task, "known_dynamic_columns", []))
            if c in past_data.column_names or c in future_data.column_names
        ]

        past_covars = list(dict.fromkeys(past_covars))
        known_covars = list(dict.fromkeys(known_covars))
        seq_covars = list(dict.fromkeys(past_covars + known_covars))

        preds: dict[Any, np.ndarray] = {}
        for idx, item_id in enumerate(item_order):
            if item_id not in history:
                raise ValueError(f"No history available for item_id={item_id}")

            y = _sanitize_array(np.asarray(history[item_id], dtype=float))
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            past_row = past_rows.get(item_id)
            if past_row is None:
                raise ValueError(f"Missing past row for item_id={item_id}")

            cov_by_name: dict[str, np.ndarray] = {
                cov: _to_numeric_array(past_row.get(cov), expected_len=len(y)) for cov in seq_covars
            }

            if seq_covars:
                cov_matrix = np.column_stack([_sanitize_array(cov_by_name[cov]) for cov in seq_covars]).astype(float)
                seq_features = np.column_stack([y, cov_matrix]).astype(float)
            else:
                seq_features = y.reshape(-1, 1).astype(float)

            if known_covars:
                known_cov_hist = np.column_stack([_sanitize_array(cov_by_name[cov]) for cov in known_covars]).astype(float)
            else:
                known_cov_hist = np.empty((len(y), 0), dtype=float)

            X_samples, Y_samples, K_samples = _build_multivariate_samples(
                features=seq_features,
                target=y,
                known_covariates=known_cov_hist,
                lookback=self.lookback,
                horizon=horizon,
            )

            if X_samples.shape[0] < self.min_train_samples:
                preds[item_id] = _fallback_forecast(y, horizon=horizon)
                continue

            X_last = seq_features[-self.lookback :, :]

            if known_covars:
                future_row = future_rows.get(item_id)
                known_future_cols: list[np.ndarray] = []
                for cov in known_covars:
                    future_source = (future_row or {}).get(cov)
                    fut_arr = _to_numeric_array(future_source, expected_len=horizon)
                    if future_row is None or cov not in future_row or future_source is None:
                        fut_arr = np.repeat(known_cov_hist[-1, known_covars.index(cov)], horizon)
                    known_future_cols.append(_sanitize_array(fut_arr))

                known_future_inference = np.column_stack(known_future_cols).reshape(-1).astype(float)
            else:
                known_future_inference = None

            preds[item_id] = self._fit_and_forecast_one(
                X_samples=X_samples,
                Y_samples=Y_samples,
                X_last=X_last,
                known_future_samples=K_samples,
                known_future_inference=known_future_inference,
                horizon=horizon,
                y_fallback=y,
                seed_offset=idx,
            )

        return to_prediction_dataset(preds, item_order)
