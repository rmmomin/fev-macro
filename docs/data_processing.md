# Data processing

The strict benchmark operates on raw ALFRED levels after selecting the origin information set. It applies declared FRED transform codes at the **native frequency**, then aggregates released observations to quarters. For example, mean monthly log differences is not a log difference of quarterly means. Unsupported frequencies/undefined transform specifications fail closed; missing observations stay missing through aggregation. The bridge estimates imputation and scaling on training rows only. See [the protocol](realtime_protocol.md).

The archive panel builders in `vintage_panels.py` apply transform codes from each file by default. `latest_file` code selection is exploratory. Contrary to earlier documentation, these builders do **not** implement the full FRED-MD outlier/trimming reference workflow. Separate development/nowcast scripts contain their own IQR processing; they are not the strict benchmark.

Archive `vintage_timestamp` is a month label, not verified publication metadata. Current templates and aliases can encode a survivor universe or changed series definition. These archive panels cannot certify an intramonth historical information set. The legacy quarterly provider now avoids same-month labels and never fills targets/covariates from the later scaffold; those improvements alone do not establish PIT validity.

Latest-data fetch/process scripts under `scripts/` write to `data/latest/` and `data/processed/`. Their output is useful for current-data research, and is never ingested as historical availability by strict mode. Re-running a historical target with latest data is a retrospective scenario, not a PIT forecast.

FRED transform codes 1–7 use levels, differences, log levels/differences and changes in proportional growth. Nonpositive log inputs become missing. The strict benchmark has no COVID dummies, outlier deletion, forward-looking interpolation or backfill. Any future extension needs explicit training-window fit boundaries and adversarial revision tests.
