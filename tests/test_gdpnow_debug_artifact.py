from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import (  # noqa: E402
    GDPNOW_DEBUG_COLUMNS,
    build_eval_arg_parser,
    parse_args_with_provenance,
    run_eval_pipeline,
)
import fev_macro.eval_runner as eval_runner  # noqa: E402


class _DummyTask:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name


def _install_pipeline_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    base_df = pd.DataFrame(
        {
            "item_id": ["LOG_REAL_GDP"] * 6,
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-04-01", "2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01"]
            ),
            "target": [1.0, 1.1, 1.2, 1.15, 1.22, 1.3],
        }
    )

    def fake_build_release_target_scaffold(*, release_csv_path: str, target_series_name: str):
        out = base_df.copy()
        out["item_id"] = target_series_name
        return out, {"release_csv_path": release_csv_path, "rows": int(len(out))}

    def fake_apply_gdpc1_release_truth_target(
        *,
        dataset_df: pd.DataFrame,
        release_csv_path: str,
        release_stage: str,
        release_metric: str,
        target_transform: str,
    ):
        _ = release_csv_path
        _ = release_metric
        _ = target_transform
        out = dataset_df.copy()
        out["item_id"] = f"gdpc1_first_{release_stage}"
        return out, {
            "source": "fake_release_csv",
            "item_id": out["item_id"].iloc[0],
            "release_metric": "level",
            "release_stage": release_stage,
            "release_column": "first_release",
            "target_units": "level",
        }

    def fake_export_local_dataset_parquet(dataset_df: pd.DataFrame, output_path: str | Path, covariate_mode: str = "unprocessed"):
        _ = dataset_df
        _ = covariate_mode
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("stub\n", encoding="utf-8")
        return out

    def fake_make_gdp_tasks(
        *,
        dataset_path: str,
        dataset_config: str | None,
        id_col: str,
        timestamp_col: str,
        target_col: str,
        known_dynamic_columns: list[str],
        past_dynamic_columns: list[str],
        horizons: list[int],
        num_windows: int,
        metric: str,
        task_prefix: str,
    ):
        _ = dataset_path
        _ = dataset_config
        _ = id_col
        _ = timestamp_col
        _ = target_col
        _ = known_dynamic_columns
        _ = past_dynamic_columns
        _ = num_windows
        _ = metric
        return [_DummyTask(task_name=f"{task_prefix}_h{int(h)}") for h in horizons]

    class FakeGDPNowModel:
        def get_selection_debug_rows(self) -> list[dict[str, object]]:
            return [
                {
                    "model_name": "atlantafed_gdpnow_asof_window_cutoff",
                    "task_name": "log_real_gdp_h1",
                    "window_id": 0,
                    "window_cutoff_timestamp": "2020-03-31",
                    "target_quarter": "2020Q2",
                    "selected_forecast_date": "2020-03-20",
                    "bea_first_release_date": "2020-07-31",
                    "info_advantage_days": -11,
                    "is_after_window_cutoff": False,
                    "is_on_or_after_first_release": False,
                }
            ]

    def fake_build_models(model_names: list[str], seed: int = 0):
        _ = seed
        out: dict[str, object] = {}
        for model_name in model_names:
            if model_name == "atlantafed_gdpnow_asof_window_cutoff":
                out[model_name] = FakeGDPNowModel()
            else:
                out[model_name] = object()
        return out

    def fake_run_models_on_tasks_with_records(
        *,
        tasks: list[_DummyTask],
        models: dict[str, object],
        num_windows: int,
        past_data_adapter=None,
    ):
        _ = past_data_adapter
        summaries: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        for task in tasks:
            for model_name in models:
                summaries.append(
                    {
                        "task_name": task.task_name,
                        "model_name": model_name,
                        "test_error": 0.1,
                        "RMSE": 0.1,
                        "eval_metric": "RMSE",
                        "inference_time_s": 0.001,
                        "num_windows": int(num_windows),
                    }
                )
                records.append(
                    {
                        "task_name": task.task_name,
                        "model_name": model_name,
                        "window_idx": 0,
                        "item_id": "gdp",
                        "horizon_step": 1,
                        "timestamp": "2020-04-01",
                        "y_true": 1.0,
                        "y_pred": np.nan if model_name == "atlantafed_gdpnow_asof_window_cutoff" else 1.0,
                        "last_observed_target": 0.9,
                    }
                )
        return summaries, records

    def fake_generate_reports(*, summaries: list[dict[str, object]], results_dir: Path, baseline_model: str, seed: int):
        _ = summaries
        _ = baseline_model
        _ = seed
        leaderboard_df = pd.DataFrame({"model_name": ["naive_last", "atlantafed_gdpnow_asof_window_cutoff"], "RMSE": [0.1, 0.2]})
        pairwise_df = pd.DataFrame({"model_a": ["naive_last"], "model_b": ["atlantafed_gdpnow_asof_window_cutoff"], "delta": [0.1]})
        leaderboard_df.to_csv(results_dir / "leaderboard.csv", index=False)
        pairwise_df.to_csv(results_dir / "pairwise.csv", index=False)
        return leaderboard_df, pairwise_df

    monkeypatch.setattr(eval_runner, "build_release_target_scaffold", fake_build_release_target_scaffold)
    monkeypatch.setattr(eval_runner, "apply_gdpc1_release_truth_target", fake_apply_gdpc1_release_truth_target)
    monkeypatch.setattr(eval_runner, "build_covariate_df", lambda **kwargs: (pd.DataFrame(columns=["timestamp"]), {}))
    monkeypatch.setattr(eval_runner, "export_local_dataset_parquet", fake_export_local_dataset_parquet)
    monkeypatch.setattr(eval_runner, "make_gdp_tasks", fake_make_gdp_tasks)
    monkeypatch.setattr(eval_runner, "build_models", fake_build_models)
    monkeypatch.setattr(eval_runner, "run_models_on_tasks_with_records", fake_run_models_on_tasks_with_records)
    monkeypatch.setattr(eval_runner, "generate_reports", fake_generate_reports)
    monkeypatch.setattr(eval_runner, "infer_metric_column", lambda _: "RMSE")
    monkeypatch.setattr(eval_runner, "validate_y_true_matches_release_table", lambda **kwargs: {})
    monkeypatch.setattr(
        eval_runner,
        "_compute_kpi_tables",
        lambda **kwargs: (
            pd.DataFrame(columns=["model_name", "task_name", "horizon", "sample_count", "rmse_target", "mae_target", "rmse_kpi_saar", "mae_kpi_saar"]),
            pd.DataFrame(columns=["model_name", "task_name", "horizon", "period", "sample_count", "rmse_target", "mae_target", "rmse_kpi_saar", "mae_kpi_saar"]),
        ),
    )


def test_eval_pipeline_writes_gdpnow_selection_debug_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_pipeline_stubs(monkeypatch)

    parser = build_eval_arg_parser(
        description="test",
        default_target_transform="log_level",
        default_results_dir=str(tmp_path / "results"),
        default_model_set="auto",
    )
    results_dir = tmp_path / "results"
    args = parse_args_with_provenance(
        parser,
        [
            "--profile",
            "smoke",
            "--models",
            "atlantafed_gdpnow_asof_window_cutoff",
            "--horizons",
            "1",
            "--num_windows",
            "1",
            "--disable_covariates",
            "--disable_historical_vintages",
            "--allow_snapshot_eval",
            "--eval_release_stages",
            "first",
            "--results_dir",
            str(results_dir),
        ],
    )

    run_eval_pipeline(
        covariate_mode="unprocessed",
        default_target_transform="log_level",
        model_set="auto",
        cli_args=args,
    )

    debug_csv = results_dir / "gdpnow_selection_debug.csv"
    assert debug_csv.exists()
    debug_df = pd.read_csv(debug_csv)
    assert set(GDPNOW_DEBUG_COLUMNS).issubset(set(debug_df.columns))
