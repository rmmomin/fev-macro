from __future__ import annotations

from typing import Sequence

from fev import Task


def make_gdp_tasks(
    dataset_path: str,
    horizons: Sequence[int] = (1, 2, 4),
    num_windows: int = 80,
    metric: str = "RMSE",
    dataset_config: str | None = None,
    id_col: str = "id",
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    known_dynamic_columns: Sequence[str] | None = None,
    past_dynamic_columns: Sequence[str] | None = None,
    task_prefix: str = "gdp_saar",
) -> list[Task]:
    """Build one fev task per horizon for comparable multi-task evaluation."""
    if not horizons:
        raise ValueError("At least one horizon is required")

    tasks: list[Task] = []
    for horizon in horizons:
        task_name = f"{task_prefix}_h{int(horizon)}"

        task = Task(
            dataset_path=dataset_path,
            dataset_config=dataset_config,
            horizon=int(horizon),
            num_windows=int(num_windows),
            window_step_size=1,
            eval_metric=metric,
            seasonality=4,
            id_column=id_col,
            timestamp_column=timestamp_col,
            target=target_col,
            known_dynamic_columns=list(known_dynamic_columns or []),
            past_dynamic_columns=list(past_dynamic_columns or []),
            task_name=task_name,
        )
        tasks.append(task)

    return tasks
