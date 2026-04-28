## USGS Streamflow Baseflow Collapse Risk (Next 14 Days)

Portfolio-ready, Kaggle-style prediction task built from **USGS NWIS Daily Values** discharge time series.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_baseflow_collapse_14d` (probability in \([0,1]\))
- **Target**: `target_baseflow_collapse_14d` — the station’s flow drops into a **station-specific low-flow band** within the next 14 days, *conditional on not already being low*

### Data & evaluation highlights
- **Rows**: train **46,101**, test **20,432**
- **Positive rate (train)**: ~**0.395** (this is intentionally not “ultra-rare”)
- **Split**: time-based regime shift with deterministic bridge years (details in `dataset_card.md`)
- **Metric**: composite scorer (LogLoss overall + LogLoss on slice + (1 − AUPRC)); see `instruction.md`
- **Slice**: `slice_low_baseline` focuses on low-baseline conditions where collapse risk is hardest to calibrate

### Repository contents
- `train.csv`, `test.csv`, `solution.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `build_dataset.py`, `build_meta.json`
- `score_submission.py`
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

Baseline tips:
- Use CatBoost/LightGBM with `station_token` as categorical
- Add calibration (LogLoss-heavy metric)

