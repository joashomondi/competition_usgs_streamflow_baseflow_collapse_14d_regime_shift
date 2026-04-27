## USGS Streamflow Baseflow Collapse Risk (Next 14 Days)

Kaggle-style competition package built from **USGS NWIS Daily Values** discharge time series.

- **Task**: binary classification — predict `pred_baseflow_collapse_14d`
- **Target**: risk of entering a low-flow (“baseflow collapse”) episode over the next 14 days
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite LogLoss + slice LogLoss + (1 - AUPRC) (see `instruction.md`)

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

