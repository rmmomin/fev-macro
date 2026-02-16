from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import Dataset

from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from .base import (
    BaseModel,
    get_item_order,
    get_task_horizon,
    get_task_id_column,
    get_task_timestamp_column,
    get_task_target_column,
    to_prediction_dataset,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

_SINGLE_ID = "__single_series__"


def _rows_by_item(dataset: Dataset, id_col: str) -> dict[Any, dict[str, Any]]:
    """Return a per-item dict of the raw records (sequence-style Dataset)."""
    rows: dict[Any, dict[str, Any]] = {}
    for rec in dataset:
        key = rec[id_col] if id_col in rec else _SINGLE_ID
        rows[key] = rec
    return rows


def _to_datetime_index(values: Any) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        return values
    if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        out = pd.to_datetime(pd.Series(values), errors="coerce")
        out = out.dropna()
        return pd.DatetimeIndex(out.to_list())
    # scalar
    out = pd.to_datetime([values], errors="coerce")
    return pd.DatetimeIndex([x for x in out if pd.notna(x)])


def _to_float_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float)
    if isinstance(values, (list, tuple, pd.Series)):
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.asarray([values], dtype=float)


def _prepare_vintage_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalize a wide vintage panel to MultiIndex(vintage, timestamp) for fast slicing.

    Expected columns:
      - vintage: str / datetime-like
      - timestamp: str / datetime-like
      - remaining columns: series IDs
    """
    if "vintage" not in panel.columns or "timestamp" not in panel.columns:
        raise ValueError("Vintage panel must have 'vintage' and 'timestamp' columns")
    df = panel.copy()
    df["vintage"] = pd.to_datetime(df["vintage"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["vintage", "timestamp"])
    df = df.set_index(["vintage", "timestamp"]).sort_index()
    return df


def _latest_vintage_at_or_before(panel_mi: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Timestamp:
    if panel_mi.empty:
        raise ValueError("Vintage panel is empty.")
    vintages = panel_mi.index.get_level_values("vintage").unique().sort_values()
    # If everything is after cutoff, fall back to earliest available vintage to avoid crash.
    eligible = vintages[vintages <= cutoff]
    if len(eligible) == 0:
        return pd.Timestamp(vintages.min())
    return pd.Timestamp(eligible.max())


def _as_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.to_period("M").to_timestamp("MS")


def _as_quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.to_period("Q").start_time.to_period("M").to_timestamp("MS")


def _apply_nowcast_transform(s: pd.Series, transform: str) -> pd.Series:
    """Apply NY Fed-style transform codes from model_spec_FRED.csv.

    - lin: level (typically diffusion indices)
    - chg: first difference
    - pch: percent change (log-diff when possible, else pct_change)
    """
    transform = str(transform).strip().lower()
    if transform == "lin":
        return s
    if transform == "chg":
        return s.diff()
    if transform == "pch":
        s_num = pd.to_numeric(s, errors="coerce")
        # Prefer log-diff for strictly positive series
        if (s_num > 0).all():
            return 100.0 * np.log(s_num).diff()
        return 100.0 * s_num.pct_change()
    # Unknown / passthrough
    return s


def _annualized_qoq_logdiff(q: pd.Series) -> pd.Series:
    q_num = pd.to_numeric(q, errors="coerce")
    if (q_num > 0).all():
        return 400.0 * np.log(q_num).diff()
    # Fallback: annualized simple growth
    return 400.0 * (q_num / q_num.shift(1) - 1.0)


def _safe_forecast_fallback(y: np.ndarray, horizon: int) -> np.ndarray:
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(float(finite[-1]), horizon).astype(float)


def _empty_prepared_panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="vintage"), pd.DatetimeIndex([], name="timestamp")]
    )
    return pd.DataFrame(index=idx)


def _load_vintage_panel_or_empty(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return _empty_prepared_panel()

    panel_path = Path(path).expanduser().resolve()
    if not panel_path.exists():
        return _empty_prepared_panel()

    try:
        raw = pd.read_parquet(panel_path)
        return _prepare_vintage_panel(raw)
    except Exception:
        return _empty_prepared_panel()


# ---------------------------------------------------------------------
# NY Fed Staff Nowcast (FRED-only spec) — block DFM via DynamicFactorMQ
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NYFedSpecRow:
    series_id: str
    frequency: str  # "m" or "q"
    transform: str  # "lin" | "chg" | "pch" | "pca"
    blocks: tuple[str, ...]  # e.g., ("Global", "Labor")


# Spec transcribed from FRBNY-RG/New-York-Nowcast `model_spec_FRED.csv`.
# Blocks correspond to Block0_Global, Block1_Soft, Block2_Nominal, Block3_Labor.
NYFED_FRED_SPEC: tuple[NYFedSpecRow, ...] = (
    NYFedSpecRow("PAYEMS", "m", "chg", ("Global", "Labor")),
    NYFedSpecRow("JTSJOL", "m", "chg", ("Global", "Labor")),
    NYFedSpecRow("UNRATE", "m", "chg", ("Global", "Labor")),
    NYFedSpecRow("ADPMNUSNERSA", "m", "chg", ("Global", "Labor")),
    NYFedSpecRow("GACDISA066MSFRBNY", "m", "lin", ("Soft",)),
    NYFedSpecRow("GACDFSA066MSFRBPHI", "m", "lin", ("Soft",)),
    NYFedSpecRow("INDPRO", "m", "pch", ("Global",)),
    NYFedSpecRow("DGORDER", "m", "pch", ("Global",)),
    NYFedSpecRow("WHLSLRIMSA", "m", "pch", ("Global",)),
    NYFedSpecRow("BUSINV", "m", "pch", ("Global",)),
    NYFedSpecRow("AMDMVS", "m", "pch", ("Global",)),
    NYFedSpecRow("AMDMUO", "m", "pch", ("Global",)),
    NYFedSpecRow("AMDMTI", "m", "pch", ("Global",)),
    NYFedSpecRow("HSN1F", "m", "pch", ("Global",)),
    NYFedSpecRow("HOUST", "m", "pch", ("Global",)),
    NYFedSpecRow("TTLCONS", "m", "pch", ("Global",)),
    NYFedSpecRow("PERMIT", "m", "pch", ("Global",)),
    NYFedSpecRow("RSAFS", "m", "pch", ("Global",)),
    NYFedSpecRow("PCEC96", "m", "pch", ("Global",)),
    NYFedSpecRow("DSPIC96", "m", "pch", ("Global",)),
    NYFedSpecRow("BOPTEXP", "m", "pch", ("Global",)),
    NYFedSpecRow("BOPTIMP", "m", "pch", ("Global",)),
    NYFedSpecRow("IR", "m", "pch", ("Global", "Nominal")),
    NYFedSpecRow("IQ", "m", "pch", ("Global", "Nominal")),
    NYFedSpecRow("CPIAUCSL", "m", "pch", ("Global", "Nominal")),
    NYFedSpecRow("CPILFESL", "m", "pch", ("Global", "Nominal")),
    NYFedSpecRow("PCEPI", "m", "pch", ("Global", "Nominal")),
    NYFedSpecRow("PCEPILFE", "m", "pch", ("Global", "Nominal")),
    # Quarterly:
    NYFedSpecRow("PRS85006112", "q", "pca", ("Global", "Labor")),
    NYFedSpecRow("A261RX1Q020SBEA", "q", "pca", ("Global",)),
    NYFedSpecRow("GDPC1", "q", "pca", ("Global",)),
)


class NYFedNowcastMQDFMModel(BaseModel):
    """NY Fed Staff Nowcast (FRED-only) style mixed-frequency block DFM.

    This is a *Python* analog that uses `statsmodels`' DynamicFactorMQ (Banbura & Modugno 2014;
    Bok et al. 2017). It uses the NY Fed GitHub repo's `model_spec_FRED.csv` variable list and
    block structure, but runs entirely in Python.

    Notes
    -----
    - Requires a **monthly** vintage panel (FRED-MD-style) and optionally a **quarterly** panel.
    - Works best when the monthly panel preserves the ragged-edge missing values (do NOT
      forward-fill the end-of-sample).
    """

    def __init__(
        self,
        md_vintage_panel_path: str,
        qd_vintage_panel_path: str | None = None,
        *,
        spec: Iterable[NYFedSpecRow] = NYFED_FRED_SPEC,
        target_series_id: str = "GDPC1",
        include_other_quarterlies: bool = False,
        factor_orders: int = 2,
        idiosyncratic_ar1: bool = True,
        max_em_iter: int = 200,
        min_monthly_vars: int = 8,
    ) -> None:
        super().__init__(name="nyfed_nowcast_mqdfm")
        self.target_series_id = str(target_series_id)
        self.include_other_quarterlies = bool(include_other_quarterlies)
        self.factor_orders = int(factor_orders)
        self.idiosyncratic_ar1 = bool(idiosyncratic_ar1)
        self.max_em_iter = int(max_em_iter)
        self.min_monthly_vars = int(min_monthly_vars)

        self.md_vintage_panel_path = Path(md_vintage_panel_path).expanduser().resolve()
        self.qd_vintage_panel_path = (
            Path(qd_vintage_panel_path).expanduser().resolve() if qd_vintage_panel_path is not None else None
        )
        self._spec = tuple(spec)
        self.reload_vintage_panels()

    def reload_vintage_panels(self) -> None:
        self._md = _load_vintage_panel_or_empty(self.md_vintage_panel_path)
        self._qd = _load_vintage_panel_or_empty(self.qd_vintage_panel_path)

    def _build_endog(
        self,
        *,
        train_ts: pd.DatetimeIndex,
        y: np.ndarray,
        cutoff: pd.Timestamp,
    ) -> tuple[pd.DataFrame, int, str]:
        """Construct endog DataFrame for DynamicFactorMQ."""
        cutoff = pd.Timestamp(cutoff)
        start_month = _as_quarter_start(pd.Timestamp(train_ts[0]))
        end_month = _as_month_start(cutoff)
        if self._md.empty:
            raise ValueError("Monthly vintage panel unavailable for NYFed nowcast model.")

        # Select the latest MD vintage that would have been available as of cutoff.
        v_md = _latest_vintage_at_or_before(self._md, cutoff)
        md_v = self._md.xs(v_md, level="vintage")

        # Monthly variables from the NY Fed spec
        m_spec = [r for r in self._spec if r.frequency == "m"]
        monthly_ids = [r.series_id for r in m_spec if r.series_id in md_v.columns]
        if len(monthly_ids) < self.min_monthly_vars:
            # Not enough data to run the DFM sensibly.
            raise ValueError(
                f"Only {len(monthly_ids)} NYFed monthly variables found in MD panel at vintage={v_md.date()}"
            )

        idx = pd.date_range(start=start_month, end=end_month, freq="MS")
        X_m = md_v.reindex(idx)[monthly_ids].copy()

        # Apply per-series transforms
        for r in m_spec:
            if r.series_id in X_m.columns:
                X_m[r.series_id] = _apply_nowcast_transform(X_m[r.series_id], r.transform)

        # Quarterly: target is always included and sourced from `y`
        y_name = self.target_series_id
        y_m = pd.Series(index=idx, dtype=float)
        # Fill quarterly values at quarter-end months
        for q_dt, q_val in zip(train_ts, y, strict=False):
            if not np.isfinite(q_val):
                continue
            m_dt = _as_month_start(pd.Timestamp(q_dt))
            if m_dt in y_m.index:
                y_m.loc[m_dt] = float(q_val)

        endog = X_m.copy()
        endog[y_name] = y_m

        # Optional: include the other quarterlies from the spec if QD panel is provided
        if self.include_other_quarterlies and not self._qd.empty:
            v_qd = _latest_vintage_at_or_before(self._qd, cutoff)
            qd_v = self._qd.xs(v_qd, level="vintage")
            q_spec = [r for r in self._spec if r.frequency == "q" and r.series_id != self.target_series_id]
            for r in q_spec:
                if r.series_id not in qd_v.columns:
                    continue
                q_series = qd_v[r.series_id].copy()
                # Transform to annualized QoQ growth (the NY Fed spec labels these as QoQ % chg. AR)
                q_g = _annualized_qoq_logdiff(q_series) if r.transform == "pca" else q_series
                q_m = pd.Series(index=idx, dtype=float)
                for q_dt, q_val in q_g.items():
                    if not np.isfinite(q_val):
                        continue
                    m_dt = _as_month_start(pd.Timestamp(q_dt))
                    if m_dt in q_m.index:
                        q_m.loc[m_dt] = float(q_val)
                endog[r.series_id] = q_m

        # Ensure ordering: monthly variables first, quarterly last
        monthly_cols = [c for c in endog.columns if c in X_m.columns]
        quarterly_cols = [c for c in endog.columns if c not in monthly_cols]
        endog = endog[monthly_cols + quarterly_cols]

        return endog, len(monthly_cols), y_name

    def _build_factor_map(self, endog_columns: list[str]) -> dict[str, list[str]]:
        spec_map = {r.series_id: r for r in self._spec}
        factors_map: dict[str, list[str]] = {}
        for col in endog_columns:
            if col in spec_map:
                blocks = list(spec_map[col].blocks)
            elif col == self.target_series_id:
                blocks = ["Global"]
            else:
                blocks = ["Global"]
            # Filter empty blocks
            blocks = [b for b in blocks if b]
            factors_map[col] = blocks or ["Global"]
        # Keep original ordering (nice to have): Global, Soft, Nominal, Labor, COVID
        order = ["Global", "Soft", "Nominal", "Labor", "COVID"]
        # Replace each list with ordered unique list
        for col in list(factors_map.keys()):
            fs = list(dict.fromkeys(factors_map[col]))  # preserve
            fs = sorted(fs, key=lambda x: order.index(x) if x in order else 999)
            factors_map[col] = fs
        return factors_map

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        id_col = get_task_id_column(task)
        ts_col = get_task_timestamp_column(task)

        past_rows = _rows_by_item(past_data, id_col=id_col)
        preds: dict[Any, np.ndarray] = {}

        for item_id in item_order:
            row = past_rows.get(item_id)
            if row is None:
                preds[item_id] = np.full(horizon, np.nan, dtype=float)
                continue

            target_col = get_task_target_column(task)
            y = _to_float_array(row.get(target_col, row.get("target")))
            ts = _to_datetime_index(row.get(ts_col))
            if len(ts) == 0 or y.size == 0:
                preds[item_id] = _safe_forecast_fallback(y, horizon)
                continue

            # Align y and ts lengths (defensive)
            n = min(len(ts), y.size)
            ts = ts[:n]
            y = y[:n]

            cutoff = pd.Timestamp(ts[-1])

            try:
                endog, k_endog_monthly, y_name = self._build_endog(train_ts=ts, y=y, cutoff=cutoff)
                factors_map = self._build_factor_map(list(endog.columns))

                mod = DynamicFactorMQ(
                    endog,
                    k_endog_monthly=k_endog_monthly,
                    factors=factors_map,
                    factor_orders=self.factor_orders,
                    idiosyncratic_ar1=self.idiosyncratic_ar1,
                    standardize=True,
                )
                res = mod.fit(disp=False, maxiter=self.max_em_iter)

                # Forecast out to the quarter ends for the requested horizon
                steps = 3 * horizon
                fc = res.get_forecast(steps=steps)
                pm = fc.predicted_mean

                last_month = endog.index[-1]
                target_months = [last_month + pd.DateOffset(months=3 * k) for k in range(1, horizon + 1)]
                f = []
                for m in target_months:
                    if m in pm.index:
                        f.append(float(pm.loc[m, y_name]))
                    else:
                        # fallback if something odd about the index
                        f.append(float(pm[y_name].iloc[-1]))
                preds[item_id] = np.asarray(f, dtype=float)

            except Exception:
                preds[item_id] = _safe_forecast_fallback(y, horizon)

        return to_prediction_dataset(preds, item_order)


# ---------------------------------------------------------------------
# ECB-style DFM wrapper — configurable blocks / variable selection
# ---------------------------------------------------------------------

class ECBNowcastMQDFMModel(BaseModel):
    """ECB toolbox-style mixed-frequency DFM (generic).

    The ECB toolbox is a MATLAB implementation, but its core DFM is the same family as
    `DynamicFactorMQ` (Banbura & Modugno). This wrapper gives you a configurable block/global
    DFM using the same EM-based estimation.

    To approximate the ECB toolbox behavior, you typically want:
      - a large monthly dataset (surveys, hard data, financials),
      - publication-delay / release-calendar metadata (for real-time nowcast updates),
      - and optional *blocks* of factors (real, nominal, financial, surveys, ...).

    This model does *not* require ECB's Excel template; you can pass factors as int or dict.
    """

    def __init__(
        self,
        md_vintage_panel_path: str,
        *,
        qd_vintage_panel_path: str | None = None,
        quarterly_target_name: str = "TARGET",
        factors: int | list[str] | dict[str, list[str]] = 2,
        factor_orders: int | dict[Any, int] = 1,
        idiosyncratic_ar1: bool = True,
        max_em_iter: int = 200,
        max_monthly_vars: int | None = 80,
        min_non_missing: int = 40,
    ) -> None:
        super().__init__(name="ecb_nowcast_mqdfm")
        self.quarterly_target_name = str(quarterly_target_name)
        self.factors = factors
        self.factor_orders = factor_orders
        self.idiosyncratic_ar1 = bool(idiosyncratic_ar1)
        self.max_em_iter = int(max_em_iter)
        self.max_monthly_vars = None if max_monthly_vars is None else int(max_monthly_vars)
        self.min_non_missing = int(min_non_missing)

        self.md_vintage_panel_path = Path(md_vintage_panel_path).expanduser().resolve()
        self.qd_vintage_panel_path = (
            Path(qd_vintage_panel_path).expanduser().resolve() if qd_vintage_panel_path is not None else None
        )
        self.reload_vintage_panels()

    def reload_vintage_panels(self) -> None:
        self._md = _load_vintage_panel_or_empty(self.md_vintage_panel_path)
        self._qd = _load_vintage_panel_or_empty(self.qd_vintage_panel_path)

    def _select_monthly_variables(self, md_v: pd.DataFrame, idx: pd.DatetimeIndex) -> list[str]:
        cols = [c for c in md_v.columns if c not in ("vintage", "timestamp")]
        if not cols:
            return []
        # Keep columns that have at least `min_non_missing` observations in the training window
        tmp = md_v.reindex(idx)[cols]
        min_obs = min(self.min_non_missing, max(8, int(round(0.35 * len(idx)))))
        non_missing = tmp.notna().sum(axis=0)
        keep = non_missing[non_missing >= min_obs].index.tolist()
        if self.max_monthly_vars is not None and len(keep) > self.max_monthly_vars:
            keep = keep[: self.max_monthly_vars]
        return keep

    def predict(self, past_data: Dataset, future_data: Dataset, task: Any) -> Dataset:
        horizon = get_task_horizon(task)
        item_order = get_item_order(future_data, task)
        id_col = get_task_id_column(task)
        ts_col = get_task_timestamp_column(task)

        past_rows = _rows_by_item(past_data, id_col=id_col)
        preds: dict[Any, np.ndarray] = {}

        for item_id in item_order:
            row = past_rows.get(item_id)
            if row is None:
                preds[item_id] = np.full(horizon, np.nan, dtype=float)
                continue

            target_col = get_task_target_column(task)
            y = _to_float_array(row.get(target_col, row.get("target")))
            ts = _to_datetime_index(row.get(ts_col))
            if len(ts) == 0 or y.size == 0:
                preds[item_id] = _safe_forecast_fallback(y, horizon)
                continue

            n = min(len(ts), y.size)
            ts = ts[:n]
            y = y[:n]
            cutoff = pd.Timestamp(ts[-1])

            try:
                if self._md.empty:
                    raise ValueError("Monthly vintage panel unavailable for ECB nowcast model.")

                start_month = _as_quarter_start(pd.Timestamp(ts[0]))
                end_month = _as_month_start(cutoff)
                idx = pd.date_range(start=start_month, end=end_month, freq="MS")

                v_md = _latest_vintage_at_or_before(self._md, cutoff)
                md_v = self._md.xs(v_md, level="vintage")

                monthly_cols = self._select_monthly_variables(md_v, idx)
                if not monthly_cols:
                    raise ValueError("No eligible monthly inputs for ECB nowcast model.")
                X_m = md_v.reindex(idx)[monthly_cols].copy()

                # Quarterly target series
                y_m = pd.Series(index=idx, dtype=float)
                for q_dt, q_val in zip(ts, y, strict=False):
                    if not np.isfinite(q_val):
                        continue
                    m_dt = _as_month_start(pd.Timestamp(q_dt))
                    if m_dt in y_m.index:
                        y_m.loc[m_dt] = float(q_val)

                endog = X_m.copy()
                endog[self.quarterly_target_name] = y_m

                # Order: monthly first
                endog = endog[monthly_cols + [self.quarterly_target_name]]

                mod = DynamicFactorMQ(
                    endog,
                    k_endog_monthly=len(monthly_cols),
                    factors=self.factors,
                    factor_orders=self.factor_orders,
                    idiosyncratic_ar1=self.idiosyncratic_ar1,
                    standardize=True,
                )
                res = mod.fit(disp=False, maxiter=self.max_em_iter)

                steps = 3 * horizon
                pm = res.get_forecast(steps=steps).predicted_mean

                last_month = endog.index[-1]
                target_months = [last_month + pd.DateOffset(months=3 * k) for k in range(1, horizon + 1)]
                f = []
                for m in target_months:
                    if m in pm.index:
                        f.append(float(pm.loc[m, self.quarterly_target_name]))
                    else:
                        f.append(float(pm[self.quarterly_target_name].iloc[-1]))
                preds[item_id] = np.asarray(f, dtype=float)

            except Exception:
                preds[item_id] = _safe_forecast_fallback(y, horizon)

        return to_prediction_dataset(preds, item_order)
