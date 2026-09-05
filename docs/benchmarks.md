# Benchmarks

Use `scripts/run_pit_backtest.py` for the strict benchmark. Its supported models, origin convention and fixtures are described in the [README](../README.md) and [protocol](realtime_protocol.md). The audited four-quarter smoke results are in [AUDIT.md](../AUDIT.md); they do not establish model superiority.

The former `make eval-*-standard` and `make realtime-oos-processed` workflows use archive month labels and broad research models. They now refuse execution under their default strict policy. Direct CLI use with `--no-strict-pit` is an explicit exploratory opt-in.

In particular, leaderboard skill thresholding and selecting top-three/top-five ensemble members from results over the evaluated periods cause ex-post selection bias. Saving a selection JSON does not fix this. Use a separately held-out design period or reconstruct nested selection at each origin, using only losses whose releases were already known. Neither is implemented for the legacy leaderboard path, so it is not certified.

The migrated strict catalog implements nested PIT ensembles in `pit_benchmark.py`. Its earlier predictions use each inner origin's snapshot, and validation outcomes are the GDP vintage known at the outer selection origin. This is distinct from both the legacy leaderboard and first-release scoring. See [model specifications](models.md).
