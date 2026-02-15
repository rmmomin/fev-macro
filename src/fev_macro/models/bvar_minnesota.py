from __future__ import annotations

from typing import Any

import numpy as np
from datasets import Dataset

from .base import BaseModel, get_history_by_item, get_item_order, get_task_horizon, get_task_id_column, to_prediction_dataset
from .random_forest import _drift_fallback, _rows_by_item, _to_numeric_array


PREFERRED_BVAR_COVARIATES: tuple[str, ...] = (
    "UNRATE",
    "FEDFUNDS",
    "CPIAUCSL",
    "INDPRO",
    "PAYEMS",
    "HOUST",
    "GS10",
    "M2REAL",
    "PCECC96",
    "IPFINAL",
    "TB3MS",
    "MORTGAGE30US",
    "SPCS20RSA",
    "WPSFD49207",
    "USSTHPI",
    "EXUSEU",
    "OILPRICEx",
    "PCESVx",
    "PCDGx",
    "RSAFSx",
)


class _MinnesotaBVARBase(BaseModel):
    """Bayesian VAR with Minnesota-style shrinkage implemented via penalized regression."""

    def __init__(
        self,
        name: str,
        max_covariates: int,
        lags: int = 2,
        lambda_shrink: float = 5.0,
        cross_weight: float = 2.0,
        lag_decay: float = 1.5,
    ) -> None:
        super().__init__(name=name)
        self.max_covariates = int(max_covariates)
        self.lags = int(lags)
        self.lambda_shrink = float(lambda_shrink)
        self.cross_weight = float(cross_weight)
        self.lag_decay = float(lag_decay)

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        history = get_history_by_item(past_data, task)

        id_col = get_task_id_column(task)
        known_covars = [
            c for c in list(getattr(task, "known_dynamic_columns", [])) if c in past_data.column_names or c in future_data.column_names
        ]
        past_covars = [c for c in list(getattr(task, "past_dynamic_columns", [])) if c in past_data.column_names]
        lag_covars = list(dict.fromkeys(past_covars + known_covars))

        selected_covars = _select_covariates(
            available=lag_covars,
            preferred=PREFERRED_BVAR_COVARIATES,
            max_covariates=self.max_covariates,
        )

        past_rows = _rows_by_item(dataset=past_data, id_col=id_col)
        future_rows = _rows_by_item(dataset=future_data, id_col=id_col)

        preds: dict[Any, np.ndarray] = {}
        for item_id in item_order:
            y = np.asarray(history[item_id], dtype=float)
            if y.size == 0:
                raise ValueError(f"No history available for item_id={item_id}")

            past_row = past_rows.get(item_id)
            future_row = future_rows.get(item_id)
            if past_row is None:
                raise ValueError(f"Missing past row for item_id={item_id}")

            past_cov = {col: _to_numeric_array(past_row.get(col), expected_len=len(y)) for col in selected_covars}
            future_known_cov = {
                col: _to_numeric_array((future_row or {}).get(col), expected_len=horizon)
                for col in selected_covars
                if col in known_covars
            }

            preds[item_id] = self._forecast_one(
                y=y,
                selected_covars=selected_covars,
                past_cov=past_cov,
                future_known_cov=future_known_cov,
                horizon=horizon,
            )

        return to_prediction_dataset(preds, item_order)

    def _forecast_one(
        self,
        y: np.ndarray,
        selected_covars: list[str],
        past_cov: dict[str, np.ndarray],
        future_known_cov: dict[str, np.ndarray],
        horizon: int,
    ) -> np.ndarray:
        if y.size < max(16, self.lags + 4):
            return _drift_fallback(y, horizon=horizon)

        panel = np.column_stack([y, *[past_cov[c] for c in selected_covars]]).astype(float)
        if panel.shape[0] <= self.lags + 2:
            return _drift_fallback(y, horizon=horizon)

        betas = _fit_minnesota_bvar(
            panel=panel,
            lags=self.lags,
            lambda_shrink=self.lambda_shrink,
            cross_weight=self.cross_weight,
            lag_decay=self.lag_decay,
        )
        if betas is None:
            return _drift_fallback(y, horizon=horizon)

        extended = panel.copy()
        for step in range(horizon):
            x = _build_var_regressor(extended=extended, lags=self.lags)
            yhat = np.array([float(x @ betas[:, eq]) for eq in range(betas.shape[1])], dtype=float)

            # Override exogenous variables with known future values when available.
            for j, cov in enumerate(selected_covars, start=1):
                fut = future_known_cov.get(cov)
                if fut is None:
                    continue
                val = float(fut[step])
                if np.isfinite(val):
                    yhat[j] = val

            if not np.isfinite(yhat).all():
                return _drift_fallback(y, horizon=horizon)

            extended = np.vstack([extended, yhat.reshape(1, -1)])

        fc = extended[-horizon:, 0].astype(float)
        if not np.isfinite(fc).all():
            return _drift_fallback(y, horizon=horizon)
        return fc


class BVARMinnesota8Model(_MinnesotaBVARBase):
    def __init__(self) -> None:
        super().__init__(name="bvar_minnesota_8", max_covariates=7, lags=2, lambda_shrink=6.0)


class BVARMinnesota20Model(_MinnesotaBVARBase):
    def __init__(self) -> None:
        super().__init__(name="bvar_minnesota_20", max_covariates=19, lags=2, lambda_shrink=7.0)


def _select_covariates(available: list[str], preferred: tuple[str, ...], max_covariates: int) -> list[str]:
    ordered: list[str] = []
    avail_set = set(available)

    for c in preferred:
        if c in avail_set and c not in ordered:
            ordered.append(c)
        if len(ordered) >= max_covariates:
            return ordered

    for c in available:
        if c not in ordered:
            ordered.append(c)
        if len(ordered) >= max_covariates:
            break

    return ordered


def _fit_minnesota_bvar(
    panel: np.ndarray,
    lags: int,
    lambda_shrink: float,
    cross_weight: float,
    lag_decay: float,
) -> np.ndarray | None:
    T, K = panel.shape
    if T <= lags:
        return None

    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    for t in range(lags, T):
        X_rows.append(_build_var_regressor(panel[:t], lags=lags))
        Y_rows.append(panel[t])

    X = np.vstack(X_rows)  # (N, P)
    Y = np.vstack(Y_rows)  # (N, K)
    P = X.shape[1]

    betas = np.zeros((P, K), dtype=float)
    for eq in range(K):
        prior_mean = np.zeros(P, dtype=float)
        prior_mean[0] = 0.0  # intercept

        # own first lag centered near random-walk behavior
        own_lag1_idx = 1 + eq
        prior_mean[own_lag1_idx] = 1.0

        w = np.zeros(P, dtype=float)
        w[0] = 0.2
        for lag in range(1, lags + 1):
            for var in range(K):
                idx = 1 + (lag - 1) * K + var
                decay = float(lag) ** lag_decay
                if var == eq and lag == 1:
                    w[idx] = 0.3 * decay
                elif var == eq:
                    w[idx] = 1.0 * decay
                else:
                    w[idx] = cross_weight * decay

        XtX = X.T @ X
        Xty = X.T @ Y[:, eq]
        D = np.diag((lambda_shrink * (w**2)).astype(float))
        rhs = Xty + D @ prior_mean

        try:
            beta = np.linalg.solve(XtX + D, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX + D) @ rhs

        betas[:, eq] = beta

    return betas


def _build_var_regressor(extended: np.ndarray, lags: int) -> np.ndarray:
    K = extended.shape[1]
    feats = [1.0]  # intercept
    for lag in range(1, lags + 1):
        feats.extend(extended[-lag, :].tolist())
    return np.asarray(feats, dtype=float).reshape(1, 1 + lags * K).ravel()
