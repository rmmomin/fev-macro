# Strict model catalog

All 24 entries in the original registry now have explicit strict implementations or admission gates. The strict catalog has 28 names: it also includes `last_growth`, `mean_growth`, `bridge_ridge`, and `ensemble_weighted_top5` (AR4 and other names overlap). This accounting includes randomized controls and deliberately weak baselines; it is not 28 recommended nowcasting models.

Use `scripts/run_pit_backtest.py --models all`. The old `fev` adapters and scripts remain exploratory. Strict implementations live in `pit_models.py` and `pit_benchmark.py`; they do not call the old adapters or independently load MD/QD panels. They are revised model specifications, not numerical replicas of the legacy implementations.

## Targets and specifications

Every output contains a coherent GDP level and last-step q/q SAAR growth. Internally, model forecasts are converted into paths of quarterly log GDP growth. Log-level forecasts are differenced against the final observed GDP level and then consecutive forecast levels. No automatic COVID deletion, intervention dummies, early stopping on evaluation data, or substitute-model fallback is applied.

| Names | Strict specification |
|---|---|
| `naive_last` | Constant GDP level (zero growth). |
| `last_growth`, `mean_growth` | Repeat the last or historical mean quarterly log growth. |
| `mean` | Forecast the historical mean **log level**. This implies reversion to the geometric mean GDP level and can produce extreme negative growth for trending GDP; retained as an intentionally weak control. |
| `drift` | Extrapolate the endpoint log-level slope; numerically equivalent to mean log growth on complete history. |
| `seasonal_naive` | Repeat log GDP from four quarters earlier. Seasonally adjusted GDP need not benefit from this control. |
| `random_normal`, `random_uniform`, `random_permutation` | Seeded draws/permutations of training log-growth distributions. Randomized controls, not substantive forecasts. |
| `ar4`, `bridge_ridge` | Four GDP growth lags; OLS AR or ridge penalty 1 with declared contemporaneous quarterly features, counts and missing indicators. |
| `auto_arima`, `auto_ets`, `theta` | StatsForecast models fitted to log GDP with season length 4. ARIMA uses stepwise AICc without approximation; ETS uses its ZZZ search; Theta uses additive decomposition. Searches use training data at each origin only. Selected order/method details are recorded where exposed. |
| `local_trend_ssm` | Statsmodels local linear trend on log GDP, no cycle or COVID intervention. Refuse an unconverged optimizer. |
| `random_forest`, `xgboost` | Eight GDP growth lags plus declared current-quarter features/counts/missing indicators. RF: 120 trees, depth 12, sqrt features. XGBoost: 200 trees, depth 3, learning rate .05. Fixed settings, one CPU thread, recursive horizons. |
| `bvar_minnesota_8`, `bvar_minnesota_20` | Standardized log GDP plus up to the first 7/19 declared transformed covariates, two lags, Minnesota-style penalized posterior mode with own-first-lag prior 1. Shrinkage 6/7; cross penalty 2; lag decay 1.5. No posterior simulation or uncertainty claim. |
| `bvar_minnesota_growth_8`, `bvar_minnesota_growth_20` | Same construction on GDP log growth, own-first-lag prior 0. Covariate transforms still come from the explicit JSON. Actual dimension is recorded; using one covariate produces a two-variable VAR, not a fictitious 20-variable model. |
| `factor_pca_qd` | Training-only scaling/PCA of declared quarterly features; at most six components, full deterministic SVD, four GDP growth lags and ridge penalty 1. Counts/missing indicators enter separately. |
| `mixed_freq_dfm_md` | Genuine monthly/quarterly DynamicFactorMQ: one shared AR(1) factor, white-noise idiosyncratic errors, GDP quarterly log-growth measurement aggregation. Fit EM on the training period, with fixed initialization, tolerance 1e-6 and maximum 5000 iterations; refuse nonconvergence. Apply fixed parameters/scales to the full origin snapshot and extract conditional GDP growth at quarter-end. |
| `lstm_univariate`, `lstm_multivariate` | From-scratch CPU LSTM, eight-step sequences, 16 hidden units, 80 fixed epochs, Adam .01 and gradient clipping 1. Multivariate sequences add declared contemporaneous features/counts/missingness. No pretrained weights or evaluation-set early stopping. |
| `ensemble_avg_top3`, `ensemble_weighted_top5` | Rank fixed candidates using earlier PIT validation forecasts; combine current GDP **level paths** equally or by inverse validation RMSE. Detailed selection rules below. |
| `chronos2` | Univariate, zero-shot log GDP median from a local immutable checkpoint. Strict admission requires publisher commit publication before origin and matching artifact hashes. Never download latest weights during forecasting. |

The monthly DFM preserves missing months and uses the quarterly growth measurement equation described in [Statsmodels DynamicFactorMQ](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html). It replaces the old model's quarterly PCA proxy. Conditional smoothing uses only observations known at the outer origin, including released monthly information beyond the last GDP quarter. Future months are missing, not filled with observed future values. Parameter estimation and standardization stop at the final released GDP quarter.

## Missingness and training

The bridge/tree/LSTM/PCA paths use the means of released transformed months with counts and missing indicators. Fit all imputation/scaling/PCA parameters on training rows, never target-quarter rows. An entirely unobserved feature uses a zero imputation parameter and remains identifiable through its missing indicator/count.

The VAR treats incomplete monthly quarters as missing, rather than substituting a partial mean for a complete quarterly state. Training missing covariates use training means. Future missing/partial covariates are forecast by the VAR; actually released complete quarterly covariates can condition subsequent recursive steps. This is not a contemporaneously conditioned mixed-frequency VAR. The DFM retains native monthly missingness instead of mean-imputing it.

Training quarters typically have more complete information than current quarters. Counts and masks expose this mismatch but do not eliminate the statistical problem. Tests establish information isolation, not forecasting superiority.

## Ensemble selection

Default candidates, fixed before the outer run: `ar4`, `auto_arima`, `random_forest`, `bridge_ridge`, `mean_growth`, `last_growth`. The last eight GDP quarters with known outcomes at the outer origin are validation targets. Shift the outer origin by whole calendar quarters, preserving its calendar day/time (month-end clipping follows calendar arithmetic), and reconstruct each inner forecast from its own ALFRED snapshot. Do not select on the stored historical leaderboard or on fits rebuilt using the outer vintage.

Score these earlier predictions against same-vintage GDP growth **known at the outer selection origin**. This is not first-release selection truth: revisions available by the selection date are allowed. Every candidate must succeed on every validation target; no model or quarter is silently dropped. Require at least four validation quarters when overriding `--ensemble-windows`. Eight observations still give noisy rankings. Candidate names, inner audits, response evidence, errors, selected members and weights are saved. Ties use model name. Inverse-RMSE weights have an explicit 1e-8 floor. Member level paths are combined before computing consistent growth.

This implements causal, origin-aware selection under a fixed retrospective rule; it does not prove that a researcher chose the rule without hindsight. A validation date on which its target is already released is refused rather than retroactively moved to a favorable date.

## Checkpoints and failures

`chronos2` is unavailable without a pinned evidence manifest. `scripts/prepare_chronos_checkpoint.py` retrieves a named publisher commit, saves config/weights and hashes, and records its dated commit evidence. The loader checks the local bytes and publication cutoff and runs on CPU without network downloads. A checkpoint published in 2025/2026 is inadmissible in a 2019 backtest. Earlier checkpoints can be specified independently for other eligible origins.

Publisher commit evidence establishes that those weights existed before the forecast; it does **not** independently audit the training corpus or prove absence of benchmark overlap before publication. This is an additional external archive assumption, like reliance on ALFRED's historical records. The current model card and publisher repository are [amazon/chronos-2](https://huggingface.co/amazon/chronos-2).

Default failures abort the run. With `--on-model-error record`, every requested model has a row: `status=ok`, `failed`, or `unsupported`, with an explicit reason. Failed rows have missing forecasts and `pit_validated=False`, and also appear in `failures.csv`. Metrics use finite matched forecasts and do not convert failure into a baseline prediction. Runtime dependency versions, warning messages, seeds, parameters and preprocessing evidence are recorded. Seed derivation depends on model/information date/target, never catalog ordering.
