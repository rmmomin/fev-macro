#!/usr/bin/env python3
# DEV ONLY: non-core utility script retained for development workflows.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fev_macro.report import generate_reports, infer_metric_column  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leaderboard/pairwise reports from saved summaries.")
    parser.add_argument("--summaries", type=str, default="results/summaries.jsonl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--baseline_model", type=str, default="naive_last")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_summaries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summaries file not found: {path}")

    if path.suffix == ".csv":
        return pd.read_csv(path)

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    summaries_path = Path(args.summaries)
    summaries_df = load_summaries(summaries_path)

    leaderboard_df, pairwise_df = generate_reports(
        summaries=summaries_df,
        results_dir=args.results_dir,
        baseline_model=args.baseline_model,
        seed=args.seed,
    )

    metric_col = infer_metric_column(summaries_df)

    print(f"Wrote leaderboard: {Path(args.results_dir) / 'leaderboard.csv'}")
    print(f"Wrote pairwise: {Path(args.results_dir) / 'pairwise.csv'}")

    if metric_col in leaderboard_df.columns:
        cols = [c for c in ["model_name", metric_col, "win_rate", "skill_vs_baseline"] if c in leaderboard_df.columns]
        print(leaderboard_df[cols].head(10).to_string(index=False))

    if not pairwise_df.empty:
        print(pairwise_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
