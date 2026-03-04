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
            "  pip install -r requirements-boe.txt"
        ) from exc
    return fe


def make_plots(
    forecasts_csv: Path,
    outturns_csv: Path,
    out_dir: Path,
    *,
    variable: str,
    source: str,
    metric: str,
    frequency: str,
    k: int,
    horizon: int,
    ma_window: int,
) -> None:
    fe = _import_boe()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_csv(forecasts_csv, parse_dates=["date", "vintage_date"])
    outturns = pd.read_csv(outturns_csv, parse_dates=["date", "vintage_date"])
    data = fe.ForecastData(forecasts_data=forecasts, outturns_data=outturns)

    fig, _ = fe.plot_hedgehog(
        data=data,
        variable=str(variable),
        forecast_source=str(source),
        metric=str(metric),
        frequency=str(frequency),
        k=int(k),
        return_plot=True,
    )
    fig.savefig(out_dir / "hedgehog.png", dpi=200, bbox_inches="tight")

    main_table = getattr(data, "_main_table", pd.DataFrame()).copy()
    highlight_date: str | None = None
    if not main_table.empty:
        scoped = main_table[
            (main_table["variable"] == variable)
            & (main_table["unique_id"] == source)
            & (main_table["metric"] == metric)
            & (main_table["frequency"] == frequency)
            & (main_table["forecast_horizon"] == int(horizon))
        ]
        if not scoped.empty:
            highlight_date = str(pd.to_datetime(scoped["date"]).max().date())

    fig, _ = fe.plot_forecast_error_density(
        data=data,
        variable=str(variable),
        horizon=int(horizon),
        metric=str(metric),
        frequency=str(frequency),
        source=str(source),
        k=int(k),
        highlight_dates=[highlight_date] if highlight_date else None,
        return_plot=True,
    )
    fig.savefig(out_dir / "error_density.png", dpi=200, bbox_inches="tight")

    fig, _ = fe.plot_errors_across_time(
        data=data,
        variable=str(variable),
        metric=str(metric),
        frequency=str(frequency),
        sources=[str(source)],
        k=int(k),
        horizons=[int(horizon)],
        ma_window=int(ma_window),
        error="raw",
        return_plot=True,
    )
    fig.savefig(out_dir / "errors_across_time.png", dpi=200, bbox_inches="tight")
