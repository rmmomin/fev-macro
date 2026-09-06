# Strict model catalog

All 24 entries in the original registry now have explicit strict implementations or admission gates. The strict catalog has 32 names: the original migration added `last_growth`, `mean_growth`, `bridge_ridge`, and `ensemble_weighted_top5`; the foundation extension adds `tabpfn_bridge`, `tabpfn_ts`, `timesfm3`, and `chronos2_covariates`. This accounting includes randomized controls and deliberately weak baselines; it is not 32 recommended nowcasting models.

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
| `tabpfn_bridge` | TabPFN-3 default regression checkpoint on quarterly log GDP growth, four GDP-growth lags, origin-known macro summaries, counts and missing indicators. Four fixed ensemble estimators, recursive horizons, training-row preprocessing. Research only. |
| `tabpfn_ts` | Official TabPFN-TS pipeline with the explicit TabPFN-3 time-series checkpoint, GDP log levels, running index/calendar/context-derived seasonal features, four estimators, median prediction, explicit future quarter timestamps. GDP only. Research only. |
| `timesfm3` | Official TimesFM 3 forecaster and immutable checkpoint, GDP log-level median, local CPU inference. A univariate comparator: its multivariate capability is not enabled in this adapter. Non-commercial research, non-production only. |
| `chronos2_covariates` | Chronos-2 median quarterly log growth conditioned on origin-known quarterly macro summaries, release counts and missing indicators. Unavailable feature values stay NaN. Shares the existing Chronos checkpoint; no fine-tuning. |

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

## Foundation adapters and information sets

Install `requirements-pit-foundation.txt` for the optional TabPFN/TimesFM packages; core and earlier catalog installations remain supported. All new adapters validate their checkpoint manifest before importing the optional model library. Checkpoints include exact publisher file hashes and a dated immutable commit. The TabPFN-3 and TimesFM-3 manifests also retain the publisher's license file. Config and license Git-blob hashes and weight LFS SHA-256 hashes are checked. Unrecorded files and alternate checkpoint variants are rejected.

`--model-use production` is the default. The fixed TabPFN-3 and TimesFM-3 specifications require `--model-use research`; editing a manifest's license label cannot bypass this restriction. This flag records the intended use and is not a grant of rights. Consult [TabPFN-3's license](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) and [TimesFM-3's license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE). Research-mode outputs cannot be relabeled as production-authorized outputs.

For `tabpfn_bridge`, each training row predicts its quarter's log GDP growth using four *earlier* growth values and that quarter's macro features from the outer origin's verified snapshot. Labels and preprocessing stop at the last released GDP quarter. Recursive predictions supply later GDP lags. NaN macro features and explicit availability indicators go to TabPFN; no external scaler or imputer is fitted on forecast rows. This remains an origin-vintage regression: it does not claim each training quarter reproduces the ragged edge that existed in that historical quarter.

For `chronos2_covariates`, the API's `future_covariates` argument refers to positions after the last released GDP quarter. Those positions may already have released monthly indicators at the actual origin. We pass only the transformed partial-quarter means, counts, and missing indicators constructed from that origin's verified snapshot. An entirely unobserved quarter has NaN feature values, zero counts and missing indicators; no future macro releases are fetched, interpolated or forecast as if observed. These are quarterly release summaries, not complete-quarter realizations or a native mixed-frequency measurement equation. The adapter predicts log **growth**, whereas `chronos2` predicts log **levels**; differences between them cannot be attributed exclusively to covariates.

`tabpfn_ts` and `timesfm3` receive contiguous GDP history only. TabPFN-TS receives an explicit future quarter calendar with no target values. Its automatic seasonal detection uses the nonmissing context targets only. It receives no past-only macro columns that the wrapper could silently discard. TimesFM-3 receives no covariates and no missing GDP values for its interpolation code to fill. All adapters return quarterly log-growth paths; the benchmark converts each final-quarter growth to SAAR with `100 * expm1(4*g)`.

Inference uses pinned local files on CPU, fixed seeds and single-thread Torch execution. TabPFN telemetry/browser login are disabled and Hugging Face offline mode is enabled for the new adapters. Optional integration tests disable socket connections entirely while running real weights. Audits record the checkpoint, license/use mode, library versions, target definition, context/feature hashes, forecast features including nulls, and the underlying ALFRED input ledger. Input availability follows the same prior-New-York-day rule as every strict model.

See [foundation-model run instructions](foundation_models.md) for preparation, reproduction and limitations.
