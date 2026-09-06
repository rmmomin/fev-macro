# Foundation-model research runs

The additional names are `tabpfn_bridge`, `tabpfn_ts`, `timesfm3`, and `chronos2_covariates`. They are part of `--models all`, with explicit unsupported rows if weights, dependencies, publication dates or use restrictions prevent inference. Their exact definitions and input rules are in [models.md](models.md).

## Install and prepare

```bash
python -m pip install -r requirements-pit-foundation.txt

python scripts/prepare_foundation_checkpoint.py \
  --model tabpfn_bridge --revision 24a16a89d245878b846555110985634aa2e656d7 \
  --origin 2026-09-04 --model-use research --out data/checkpoints/tabpfn3-bridge

python scripts/prepare_foundation_checkpoint.py \
  --model tabpfn_ts --revision 24a16a89d245878b846555110985634aa2e656d7 \
  --origin 2026-09-04 --model-use research --out data/checkpoints/tabpfn3-ts

python scripts/prepare_foundation_checkpoint.py \
  --model timesfm3 --revision 43046b85ec22d584a13f8098c2ed39c889e129c2 \
  --origin 2026-09-04 --model-use research --out data/checkpoints/timesfm3

python scripts/prepare_foundation_checkpoint.py \
  --model chronos2 --revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
  --origin 2026-09-04 --out data/checkpoints/chronos2
```

Existing validated Chronos manifests work for both Chronos variants. Use new output directories; preparation never overwrites existing evidence. Downloads happen only in preparation. If a Hub cache download has the wrong publisher hash, preparation refuses it. Retry with a fresh `--cache-dir`; `HF_HUB_DISABLE_XET=1` selects ordinary HTTP transport when the optional Xet transport fails. Neither option relaxes identity checks.

The pinned TabPFN repository commit is dated July 4, 2026; the TimesFM commit September 2, 2026; the Chronos commit June 5, 2026. These conservative repository-commit dates are used even when unchanged weights existed in earlier commits. These exact manifests are therefore ineligible for a 2019 forecast. To use an earlier eligible artifact, prepare a documented earlier immutable revision, never edit publication dates.

## Run

Create an origin CSV with `origin_date,target_quarter` and row `2026-09-04,2026Q3`. Use a verified ALFRED store covering the declared series. The origin means beginning of September 4 in New York, with a September 3 information cutoff, even when the command is run later.

```bash
python scripts/run_pit_backtest.py \
  --db data/realtime/pit_catalog.duckdb --origins origins.csv --forecast-only \
  --series-specs config/pit/catalog_series.json --covariates-from-specs \
  --models naive_last bridge_ridge tabpfn_bridge tabpfn_ts timesfm3 chronos2 chronos2_covariates \
  --model-use research \
  --tabpfn-bridge-checkpoint data/checkpoints/tabpfn3-bridge/manifest.json \
  --tabpfn-ts-checkpoint data/checkpoints/tabpfn3-ts/manifest.json \
  --timesfm3-checkpoint data/checkpoints/timesfm3/manifest.json \
  --chronos-checkpoint data/checkpoints/chronos2/manifest.json \
  --out results/foundation_nowcast
```

Use `--on-model-error record` to retain explicit failed/unsupported rows instead of aborting. The default production-use mode refuses restricted checkpoints; specifying research use is not permission to use those outputs commercially or in production. Each forecast retains its underlying data ledger, checkpoint evidence and runtime configuration. No Q3 accuracy score is possible before the GDP releases exist.

## Preserve individual months

Add `--foundation-features monthly_slots` to use separate month-1, month-2 and
month-3 values, each with a missing indicator, in `tabpfn_bridge` and
`chronos2_covariates`. Transformations run on the complete native monthly
calendar before training-window selection. January log growth uses December;
a missing February prevents computing March growth from January. Quarterly
covariates stay in a single quarterly channel. Missing observations remain NaN.

For example, a Q3 origin with only July employment available gets July's value
in `PAYEMS__m1`, unknown values in `__m2` and `__m3`, and matching masks. August
employment enters only after its actual vintage date passes the cutoff. GDP
remains quarterly. These are aligned mixed-frequency features, not monthly GDP
interpolation or a native mixed-frequency measurement equation.

The default `quarterly` representation retains the earlier experiment for
controlled comparisons. Other models keep their existing input contracts.
The representation, column order, feature hashes and actual future-quarter
feature values appear in every affected model's audit; the CSV records the
representation as well. See [the expanded data collection](nowcast_data.md).

## Tests and scope

`python -m pytest tests/test_pit_foundation.py -q` runs synthetic checkpoint/recording-backend contract tests without downloading model weights. These test that future revisions and same-day releases cannot enter any adapter, that only training rows fit the TabPFN bridge, that Chronos preserves missing features, and that GDP-only adapters ignore macro inputs. Such tests validate the adapter contract, not pretrained-model accuracy.

For real local inference tests, set `FEV_FOUNDATION_CHECKPOINTS` to a JSON file mapping `chronos2`, `tabpfn_bridge`, `tabpfn_ts`, and `timesfm3` to their local manifest paths, then run:

```bash
FEV_FOUNDATION_CHECKPOINTS=checkpoints.json \
  python -m pytest tests/test_pit_foundation.py -k actual_pinned -q
```

These tests run two-step synthetic-series forecasts with the actual installed packages and checkpoint bytes, with socket connections disabled. Ordinary CI does not download or accept licenses for model weights; it tests core, catalog, and optional foundation package installations separately.

These implementations do not establish superior GDP accuracy. Publisher commit evidence is an external historical-availability assumption, not an independent audit of pretraining or fine-tuning corpora. Retrospective experiments with later weights must remain separate from strict PIT comparisons. Compare models only on common eligible origins, with enough distinct realized target quarters; current-quarter forecasts are demonstrations, not validation rankings.
