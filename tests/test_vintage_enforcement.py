from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.eval_runner import (  # noqa: E402
    _apply_profile_defaults,
    build_eval_arg_parser,
    parse_args_with_provenance,
    run_eval_pipeline,
)
import fev_macro.eval_runner as eval_runner  # noqa: E402


def _build_processed_parser():
    return build_eval_arg_parser(
        description="test",
        default_target_transform="saar_growth",
        default_results_dir="results/test",
        default_model_set="auto",
        default_task_prefix="gdp_saar",
        default_qd_vintage_panel="data/panels/fred_qd_vintage_panel_processed.parquet",
    )


class _DummyTask:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name
        self.id_column = "id"
        self.timestamp_column = "timestamp"
        self.target = "target"
        self.known_dynamic_columns: list[str] = []
        self.past_dynamic_columns: list[str] = []


def _install_lightweight_eval_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
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
        out = dataset_df.copy()
        out["target"] = np.linspace(1.0, 1.5, num=len(out))
        out["item_id"] = f"gdpc1_qoq_saar_{release_stage}_pct"
        return out, {
            "source": "fake_release_csv",
            "item_id": out["item_id"].iloc[0],
            "release_metric": release_metric,
            "release_stage": release_stage,
            "release_column": "qoq_saar_growth_realtime_first_pct",
            "target_units": "pct_qoq_saar",
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

    class FakeVintageProvider:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            self.historical_qd_dir = Path("fake/historical/qd")
            self.vintage_periods = [pd.Period("2020-02", freq="M"), pd.Period("2020-03", freq="M")]
            self._strict = bool(kwargs.get("strict", False))
            self._fallback_to_earliest = bool(kwargs.get("fallback_to_earliest", True))

        @property
        def source_kind(self) -> str:
            return "historical_csv"

        def available_range_str(self) -> str:
            return "2020-02..2020-03"

        def compatible_window_count(self, task: _DummyTask) -> int:
            _ = task
            return 2

        def adapt_past_data(self, past_data: Dataset, task: _DummyTask, return_meta: bool = False):
            _ = task
            meta = {
                "cutoff_timestamp": "2020-03-31T00:00:00",
                "selected_vintage": "2020-03",
                "source_kind": "historical_csv",
                "strict_mode": self._strict,
                "fallback_to_earliest": self._fallback_to_earliest,
            }
            if return_meta:
                return past_data, meta
            return past_data

    def fake_build_models(model_names: list[str], seed: int = 0):
        _ = seed
        return {name: object() for name in model_names}

    def fake_run_models_on_tasks_with_records(
        *,
        tasks: list[_DummyTask],
        models: dict[str, object],
        num_windows: int,
        past_data_adapter=None,
    ):
        summaries: list[dict[str, object]] = []
        prediction_records: list[dict[str, object]] = []
        past = Dataset.from_list(
            [{"id": "gdp", "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")], "target": [1.0, 1.1]}]
        )
        future = Dataset.from_list([{"id": "gdp", "timestamp": [pd.Timestamp("2020-07-01")], "target": [1.2]}])

        for task in tasks:
            for window_idx in range(int(num_windows)):
                if past_data_adapter is not None:
                    _ = past_data_adapter(past, future, task, window_idx)
                for model_name in models.keys():
                    summaries.append(
                        {
                            "task_name": task.task_name,
                            "model_name": model_name,
                            "inference_time_s": 0.001,
                            "test/RMSE": 0.1,
                        }
                    )
                    prediction_records.append(
                        {
                            "task_name": task.task_name,
                            "model_name": model_name,
                            "window_idx": window_idx,
                            "item_id": "gdp",
                            "horizon_step": 1,
                            "timestamp": "2020-07-01",
                            "y_true": 1.2,
                            "y_pred": 1.15,
                            "last_observed_target": 1.1,
                        }
                    )
        return summaries, prediction_records

    def fake_save_summaries(*, summaries: list[dict[str, object]], jsonl_path: str | Path, csv_path: str | Path):
        Path(jsonl_path).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries),
            encoding="utf-8",
        )
        pd.DataFrame(summaries).to_csv(csv_path, index=False)

    def fake_save_timing_report(*, summaries: list[dict[str, object]], output_csv: str | Path):
        df = pd.DataFrame(
            [
                {
                    "model_name": str(row.get("model_name", "")),
                    "inference_time_s": float(row.get("inference_time_s", 0.0) or 0.0),
                }
                for row in summaries
            ]
        )
        df.to_csv(output_csv, index=False)
        return df

    def fake_generate_reports(*, summaries: list[dict[str, object]], results_dir: Path, baseline_model: str, seed: int):
        _ = seed
        models = sorted({str(row.get("model_name", "")) for row in summaries if row.get("model_name")})
        leaderboard_df = pd.DataFrame(
            {
                "model_name": models,
                "RMSE": [0.1] * len(models),
                "win_rate": [0.5] * len(models),
                "skill_vs_baseline": [0.0 if m == baseline_model else 0.1 for m in models],
            }
        )
        pairwise_df = pd.DataFrame(
            {
                "model_a": [baseline_model],
                "model_b": [models[0] if models else baseline_model],
                "metric": ["RMSE"],
                "delta": [0.0],
            }
        )
        leaderboard_df.to_csv(results_dir / "leaderboard.csv", index=False)
        pairwise_df.to_csv(results_dir / "pairwise.csv", index=False)
        return leaderboard_df, pairwise_df

    monkeypatch.setattr(eval_runner, "build_release_target_scaffold", fake_build_release_target_scaffold)
    monkeypatch.setattr(eval_runner, "apply_gdpc1_release_truth_target", fake_apply_gdpc1_release_truth_target)
    monkeypatch.setattr(
        eval_runner,
        "build_covariate_df",
        lambda **kwargs: (pd.DataFrame(columns=["timestamp"]), {}),
    )
    monkeypatch.setattr(eval_runner, "export_local_dataset_parquet", fake_export_local_dataset_parquet)
    monkeypatch.setattr(eval_runner, "make_gdp_tasks", fake_make_gdp_tasks)
    monkeypatch.setattr(eval_runner, "HistoricalQuarterlyVintageProvider", FakeVintageProvider)
    monkeypatch.setattr(eval_runner, "build_models", fake_build_models)
    monkeypatch.setattr(eval_runner, "run_models_on_tasks_with_records", fake_run_models_on_tasks_with_records)
    monkeypatch.setattr(eval_runner, "save_summaries", fake_save_summaries)
    monkeypatch.setattr(eval_runner, "save_timing_report", fake_save_timing_report)
    monkeypatch.setattr(eval_runner, "generate_reports", fake_generate_reports)
    monkeypatch.setattr(eval_runner, "infer_metric_column", lambda _: "RMSE")
    monkeypatch.setattr(eval_runner, "validate_y_true_matches_release_table", lambda **kwargs: {})


def test_standard_profile_defaults_keep_historical_vintages_enabled() -> None:
    parser = _build_processed_parser()

    args_unprocessed = parse_args_with_provenance(parser, ["--profile", "standard"])
    _apply_profile_defaults(args_unprocessed, covariate_mode="unprocessed")
    assert args_unprocessed.disable_historical_vintages is False

    args_processed = parse_args_with_provenance(parser, ["--profile", "standard"])
    _apply_profile_defaults(args_processed, covariate_mode="processed")
    assert args_processed.disable_historical_vintages is False


def test_snapshot_eval_guard_blocks_without_allow() -> None:
    parser = _build_processed_parser()
    args = parse_args_with_provenance(parser, ["--disable_historical_vintages"])
    with pytest.raises(ValueError, match="Snapshot evaluation is blocked by default"):
        run_eval_pipeline(
            covariate_mode="processed",
            default_target_transform="saar_growth",
            model_set="auto",
            cli_args=args,
        )


def test_snapshot_allow_and_window_vintage_provenance_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_lightweight_eval_stubs(monkeypatch)
    parser = _build_processed_parser()

    snapshot_results = tmp_path / "snapshot_eval"
    snapshot_args = parse_args_with_provenance(
        parser,
        [
            "--profile",
            "smoke",
            "--models",
            "naive_last",
            "--horizons",
            "1",
            "--num_windows",
            "2",
            "--disable_covariates",
            "--disable_historical_vintages",
            "--allow_snapshot_eval",
            "--eval_release_stages",
            "first",
            "--results_dir",
            str(snapshot_results),
        ],
    )
    run_eval_pipeline(
        covariate_mode="processed",
        default_target_transform="saar_growth",
        model_set="auto",
        cli_args=snapshot_args,
    )

    snapshot_meta = json.loads((snapshot_results / "run_metadata.json").read_text(encoding="utf-8"))
    assert snapshot_meta["snapshot_eval"] is True

    historical_results = tmp_path / "historical_eval"
    historical_args = parse_args_with_provenance(
        parser,
        [
            "--profile",
            "smoke",
            "--models",
            "naive_last",
            "--horizons",
            "1",
            "--num_windows",
            "2",
            "--disable_covariates",
            "--eval_release_stages",
            "first",
            "--results_dir",
            str(historical_results),
        ],
    )
    run_eval_pipeline(
        covariate_mode="processed",
        default_target_transform="saar_growth",
        model_set="auto",
        cli_args=historical_args,
    )

    window_vintages_csv = historical_results / "window_vintages.csv"
    predictions_csv = historical_results / "predictions_per_window.csv"
    assert window_vintages_csv.exists()
    assert predictions_csv.exists()

    predictions_df = pd.read_csv(predictions_csv)
    assert "train_vintage" in predictions_df.columns
    assert predictions_df["train_vintage"].replace("", pd.NA).notna().any()

    leaderboard_df = pd.read_csv(historical_results / "leaderboard.csv")
    assert {"MAE", "MSE", "RMSE"}.issubset(set(leaderboard_df.columns))
    assert leaderboard_df[["MAE", "MSE", "RMSE"]].notna().all().all()
