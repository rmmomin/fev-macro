from __future__ import annotations

from pathlib import Path

import pandas as pd


def _import_boe() -> "module":
    try:
        import forecast_evaluation as fe  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing optional dependency 'forecast_evaluation'. Install with:\n"
            "  pip install forecast_evaluation\n"
            "or:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return fe


def _run_rolling_with_fallback(fe: "module", data: object, benchmark_model: str, k: int) -> "object":
    max_window = 40
    main_table = data._main_table  # type: ignore[attr-defined]
    vintage_col = "vintage_date"
    if vintage_col not in main_table.columns:
        if "vintage_date_forecast" in main_table.columns:
            vintage_col = "vintage_date_forecast"
        elif "vintage_date_outturn" in main_table.columns:
            vintage_col = "vintage_date_outturn"
    available = int(pd.to_datetime(main_table[vintage_col], errors="coerce").nunique())
    start = min(max_window, max(1, available - 1))
    for window in range(start, 0, -1):
        try:
            return fe.rolling_analysis(
                data=data,
                window_size=int(window),
                analysis_func=fe.diebold_mariano_table,
                analysis_args={"benchmark_model": str(benchmark_model), "k": int(k)},
            )
        except ValueError:
            continue
    raise ValueError("Could not compute rolling DM: insufficient sample for any window_size >= 1.")


def _run_fluctuation_with_fallback(fe: "module", data: object, benchmark_model: str, k: int) -> "object":
    max_window = 40
    main_table = data._main_table  # type: ignore[attr-defined]
    vintage_col = "vintage_date"
    if vintage_col not in main_table.columns:
        if "vintage_date_forecast" in main_table.columns:
            vintage_col = "vintage_date_forecast"
        elif "vintage_date_outturn" in main_table.columns:
            vintage_col = "vintage_date_outturn"
    available = int(pd.to_datetime(main_table[vintage_col], errors="coerce").nunique())
    start = min(max_window, max(1, available - 1))
    for window in range(start, 0, -1):
        try:
            return fe.fluctuation_tests(
                data=data,
                window_size=int(window),
                test_func=fe.diebold_mariano_table,
                test_args={"benchmark_model": str(benchmark_model), "k": int(k)},
            )
        except ValueError:
            continue
    raise ValueError("Could not compute fluctuation DM: insufficient sample for any window_size >= 1.")


def run_boe_eval(
    forecasts_csv: Path,
    outturns_csv: Path,
    out_dir: Path,
    *,
    k: int,
    benchmark_model: str,
    variable: str | None,
    same_date_range: bool,
    add_boe_random_walk: bool,
    add_boe_ar_p: bool,
) -> None:
    fe = _import_boe()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_csv(forecasts_csv, parse_dates=["date", "vintage_date"])
    outturns = pd.read_csv(outturns_csv, parse_dates=["date", "vintage_date"])
    data = fe.ForecastData(forecasts_data=forecasts, outturns_data=outturns)
    benchmark_metric = "levels"
    if "metric" in forecasts.columns:
        metric_values = sorted({str(v).strip().lower() for v in forecasts["metric"].dropna().tolist()})
        if len(metric_values) == 1 and metric_values[0] in {"levels", "pop", "yoy"}:
            benchmark_metric = metric_values[0]

    if add_boe_random_walk:
        fe.add_random_walk_forecasts(data, variable=variable, metric=benchmark_metric)
    if add_boe_ar_p:
        fe.add_ar_p_forecasts(data, variable=variable, metric=benchmark_metric)

    acc = fe.compute_accuracy_statistics(
        data=data,
        variable=variable,
        k=int(k),
        same_date_range=bool(same_date_range),
    )
    acc.to_df().to_csv(out_dir / "accuracy.csv", index=False)

    try:
        dm = fe.diebold_mariano_table(
            data=data,
            benchmark_model=str(benchmark_model),
            k=int(k),
        )
        dm.to_df().to_csv(out_dir / "diebold_mariano.csv", index=False)
    except Exception as exc:
        pd.DataFrame().to_csv(out_dir / "diebold_mariano.csv", index=False)
        print(f"WARNING: BoE DM table unavailable: {exc}")

    try:
        rolling = _run_rolling_with_fallback(
            fe=fe,
            data=data,
            benchmark_model=str(benchmark_model),
            k=int(k),
        )
        rolling.to_df().to_csv(out_dir / "rolling_dm.csv", index=False)
    except Exception as exc:
        pd.DataFrame().to_csv(out_dir / "rolling_dm.csv", index=False)
        print(f"WARNING: BoE rolling DM unavailable: {exc}")

    try:
        fluctuation = _run_fluctuation_with_fallback(
            fe=fe,
            data=data,
            benchmark_model=str(benchmark_model),
            k=int(k),
        )
        fluctuation.to_df().to_csv(out_dir / "fluctuation_dm.csv", index=False)
    except Exception as exc:
        pd.DataFrame().to_csv(out_dir / "fluctuation_dm.csv", index=False)
        print(f"WARNING: BoE fluctuation DM unavailable: {exc}")

    main_table = getattr(data, "_main_table", None)
    if isinstance(main_table, pd.DataFrame) and not main_table.empty:
        main_table.to_csv(out_dir / "main_table.csv", index=False)
