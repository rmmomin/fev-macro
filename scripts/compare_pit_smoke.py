#!/usr/bin/env python3
"""Controlled revision-leakage comparison; deliberately invalid forecasts are labelled.

Reproduces the historical-input effect of the removed current-FRED fallback,
using the last captured (2020-04-30) snapshot for every 2019 origin. The target
quarter itself remains masked so this isolates revised-history contamination.
This is not a rerun of the entire legacy leaderboard.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fev_macro.asof_store import AsofStore
from fev_macro.pit_benchmark import _regression_path, paired_metrics, score_forecasts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True)
    p.add_argument('--results', default='results/pit_backtest')
    args = p.parse_args()
    out = Path(args.results)
    after = pd.read_csv(out / 'forecasts.csv')
    after = after.loc[after.model != 'bridge_ridge'].copy()
    store = AsofStore(args.db)
    try:
        latest = store.snapshot_long(asof_ts='2020-05-01', series_ids=['GDPC1'])
    finally:
        store.close()
    latest.index = pd.PeriodIndex(latest.obs_ts, freq='Q')
    before = after.copy()
    for i, row in before.iterrows():
        y = latest.loc[(latest.index >= pd.Period(row.training_min_quarter)) &
                       (latest.index <= pd.Period(row.training_max_quarter)), 'value'].dropna().to_numpy()
        g = np.diff(np.log(y))
        if row.model == 'naive_last':
            path = [0.]*row.horizon
        elif row.model == 'mean_growth':
            path = [float(g.mean())]*row.horizon
        elif row.model == 'last_growth':
            path = [float(g[-1])]*row.horizon
        else:
            path = _regression_path(g, row.horizon, None)
        before.loc[i, 'g_hat_saar'] = 100*np.expm1(4*path[-1])
        before.loc[i, 'y_hat_level'] = y[-1]*np.exp(sum(path))
    before['pit_validated'] = False
    before['input_policy'] = 'DELIBERATELY LEAKED: 2020-04-30 snapshot at every origin'
    before = before.drop(columns=['audit_id', 'max_vintage_used', 'information_cutoff'])
    truth = pd.read_csv(out / 'truth.csv')
    metrics_before = paired_metrics(score_forecasts(before, truth))
    metrics_after = paired_metrics(score_forecasts(after, truth))
    comparison = metrics_before.merge(metrics_after, on=['model', 'horizon', 'release_stage'], suffixes=('_leaked', '_pit'))
    comparison['rmse_change_pit_minus_leaked'] = comparison.rmse_pit - comparison.rmse_leaked
    before.to_csv(out / 'deliberately_leaked_comparison.csv', index=False)
    comparison.to_csv(out / 'revision_comparison.csv', index=False)
    changes = before[['model', 'origin_date', 'g_hat_saar']].merge(after[['model', 'origin_date', 'g_hat_saar']],
              on=['model', 'origin_date'], suffixes=('_leaked', '_pit'))
    print(comparison.loc[comparison.release_stage=='first', ['model','rmse_leaked','rmse_pit','rmse_change_pit_minus_leaked']].to_string(index=False))
    print('Max forecast change (SAAR percentage points):', float(abs(changes.g_hat_saar_pit-changes.g_hat_saar_leaked).max()))


if __name__ == '__main__':
    main()
