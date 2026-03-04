from __future__ import annotations

from pathlib import Path

import pandas as pd


def _quarter_end_from_period(q: str | pd.Period) -> pd.Timestamp:
    period = pd.Period(str(q), freq="Q-DEC")
    return period.to_timestamp(how="end").normalize()


def _quarter_end_from_any_date(d: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return pd.Period(ts, freq="Q-DEC").to_timestamp(how="end").normalize()


def _compute_outturn_forecast_horizon(date_qe: pd.Timestamp, vintage_qe: pd.Timestamp) -> int:
    date_period = pd.Period(date_qe, freq="Q-DEC")
    vintage_period = pd.Period(vintage_qe, freq="Q-DEC")
    return int((date_period - vintage_period).n)


def _normalize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    cols = set(out.columns)

    if {"origin_quarter", "target_quarter", "horizon"}.issubset(cols):
        return out

    if "timestamp" not in cols:
        raise ValueError(
            "predictions CSV must include either "
            "{origin_quarter,target_quarter,horizon} or {timestamp,horizon_step} columns."
        )

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["target_quarter"] = pd.PeriodIndex(out["timestamp"], freq="Q-DEC").astype(str)

    if "horizon_step" in out.columns:
        out["horizon"] = pd.to_numeric(out["horizon_step"], errors="coerce").astype("Int64")
    elif "forecast_horizon" in out.columns:
        out["horizon"] = pd.to_numeric(out["forecast_horizon"], errors="coerce").astype("Int64") + 1
    else:
        out["horizon"] = 1

    out = out.loc[out["horizon"].notna()].copy()
    target_period = pd.PeriodIndex(out["target_quarter"], freq="Q-DEC")
    out["origin_quarter"] = (target_period - (out["horizon"].astype(int) - 1)).astype(str)

    if "release_stage" not in out.columns:
        out["release_stage"] = "first"
    if "model" not in out.columns and "model_name" in out.columns:
        out["model"] = out["model_name"].astype(str)

    return out


def _resolve_truth_value_column(pred: pd.DataFrame, truth: str) -> str:
    if truth == "first" and "y_true_level_first" in pred.columns:
        return "y_true_level_first"

    candidates = (
        "y_true_level",
        "y_true",
        "target",
    )
    for col in candidates:
        if col in pred.columns:
            return col
    raise ValueError(
        "Could not infer truth value column from predictions CSV. "
        "Expected one of: y_true_level_first, y_true_level, y_true, target."
    )


def export_to_boe_schema(
    predictions_csv: Path,
    release_table_csv: Path | None,
    out_dir: Path,
    *,
    truth: str,
    variable: str,
    metric: str,
    forecast_value_col: str,
) -> tuple[Path, Path]:
    pred = pd.read_csv(predictions_csv)
    pred = _normalize_predictions(pred)
    value_col = str(forecast_value_col)
    if value_col not in pred.columns:
        if "y_hat_level" in pred.columns:
            value_col = "y_hat_level"
        elif "y_pred" in pred.columns:
            value_col = "y_pred"

    required = {
        "model",
        "origin_quarter",
        "target_quarter",
        "horizon",
        "release_stage",
        value_col,
    }
    missing = sorted(c for c in required if c not in pred.columns)
    if missing:
        raise ValueError(
            "predictions CSV is missing required columns: "
            + ", ".join(missing)
        )

    stage = str(truth).strip().lower()
    fc = pred.loc[pred["release_stage"].astype(str).str.lower() == stage].copy()
    if fc.empty:
        fc = pred.copy()

    fc["date"] = fc["target_quarter"].apply(_quarter_end_from_period)
    fc["vintage_date"] = fc["origin_quarter"].apply(_quarter_end_from_period)
    fc["forecast_horizon"] = pd.to_numeric(fc["horizon"], errors="coerce")
    fc = fc.loc[fc["forecast_horizon"].notna()].copy()
    fc["forecast_horizon"] = fc["forecast_horizon"].astype(int) - 1
    fc = fc.loc[fc["forecast_horizon"] >= 0].copy()

    fc["variable"] = str(variable)
    fc["frequency"] = "Q"
    fc["value"] = pd.to_numeric(fc[value_col], errors="coerce")
    fc["source"] = fc["model"].astype(str)
    if metric:
        fc["metric"] = str(metric)

    forecast_cols = ["date", "vintage_date", "variable", "frequency", "forecast_horizon", "value", "source"]
    if "metric" in fc.columns:
        forecast_cols.append("metric")
    boe_forecasts = fc[forecast_cols].dropna(subset=["value"]).copy()
    boe_forecasts = boe_forecasts.sort_values(
        ["vintage_date", "source", "date", "forecast_horizon"]
    ).reset_index(drop=True)

    truth_col = _resolve_truth_value_column(pred, stage)
    out = pred[["target_quarter", truth_col]].drop_duplicates().copy()
    out = out.rename(columns={truth_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).copy()
    out["date"] = out["target_quarter"].apply(_quarter_end_from_period)

    release_map: dict[str, pd.Timestamp] | None = None
    if release_table_csv is not None:
        rel = pd.read_csv(release_table_csv)
        if "quarter" not in rel.columns:
            if "observation_date" in rel.columns:
                rel["observation_date"] = pd.to_datetime(rel["observation_date"], errors="coerce")
                rel = rel.loc[rel["observation_date"].notna()].copy()
                rel["quarter"] = pd.PeriodIndex(rel["observation_date"], freq="Q-DEC").astype(str)
            else:
                raise ValueError("release table must include either 'quarter' or 'observation_date' column.")
        release_cols = {
            "first": "first_release_date",
            "second": "second_release_date",
            "third": "third_release_date",
        }
        rel_col = release_cols.get(stage, "first_release_date")
        if rel_col not in rel.columns:
            raise ValueError(f"release table missing required column: {rel_col}")
        rel = rel.loc[rel[rel_col].notna(), ["quarter", rel_col]].copy()
        rel["quarter"] = rel["quarter"].astype(str)
        release_map = {
            str(q): pd.Timestamp(d)
            for q, d in zip(rel["quarter"], pd.to_datetime(rel[rel_col], errors="coerce"))
            if pd.notna(d)
        }

    def _outturn_vintage(qstr: str, date_qe: pd.Timestamp) -> pd.Timestamp:
        if release_map is not None and qstr in release_map:
            return _quarter_end_from_any_date(release_map[qstr])
        return (pd.Timestamp(date_qe) + pd.offsets.QuarterEnd(1)).normalize()

    out["vintage_date"] = [_outturn_vintage(str(q), d) for q, d in zip(out["target_quarter"], out["date"])]
    out["forecast_horizon"] = [
        _compute_outturn_forecast_horizon(d, v) for d, v in zip(out["date"], out["vintage_date"])
    ]
    out["variable"] = str(variable)
    out["frequency"] = "Q"
    if metric:
        out["metric"] = str(metric)

    out_cols = ["date", "vintage_date", "variable", "frequency", "forecast_horizon", "value"]
    if "metric" in out.columns:
        out_cols.append("metric")
    boe_outturns = out[out_cols].sort_values(["vintage_date", "date"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts_path = out_dir / "boe_forecasts.csv"
    outturns_path = out_dir / "boe_outturns.csv"
    boe_forecasts.to_csv(forecasts_path, index=False)
    boe_outturns.to_csv(outturns_path, index=False)
    return forecasts_path, outturns_path
