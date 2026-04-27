## Data checks

- Check base rate by:
  - `event_era` (regime drift)
  - `station_token` (heterogeneous hydrology)
  - `season_bin` (seasonality)
  - `slice_low_baseline` (upweighted in metric)
- Treat `*_bin == -1` as missing (not “low”).

## Validation strategy (avoid leakage)

Because the split is time-based, random CV overstates performance.

Recommended:

- **Blocked time validation** using `event_era`:
  - train on earlier eras, validate on the newest era available in training
- **Group sanity** by `station_token`:
  - ensure performance is not dominated by a handful of gauges

Report metrics on:

- all rows
- slice rows (`slice_low_baseline == 1`)
- per-era breakdown

## Modeling baselines

Good starting points:

- GBDT on binned features
- regularized logistic regression with one-hot for binned features (treat `-1` as its own category)

Useful interactions:

- `season_bin × q_mean_30d_bin × station_token`
- `range_30d_bin × cv_30d_bin × event_era`
- `logq_mean_30d_bin × ddown_30d_bin` (collapse dynamics)

## Calibration

The metric is LogLoss-heavy, and the slice is upweighted:

- Tune for calibration (regularization + early stopping).
- Consider post-hoc calibration using era-aware validation (temperature scaling / Platt).
- Watch for overconfidence during bridge/test-like low-baseline conditions.

## Submission checklist

- `submission.csv` has exactly `row_id` and `pred_baseflow_collapse_14d`
- ids match `test.csv`
- predictions are finite and in [0,1]
- run:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

