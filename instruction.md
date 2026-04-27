## Objective

Predict the probability that a river gauge will experience a **baseflow collapse** within the next **14 days**, using only **past** discharge history and coarse calendar context.

Each row is a gauge-day example. Your job is to output:

- `pred_baseflow_collapse_14d`: a probability in \([0,1]\)

## Target definition (train only)

Let \(q(t)\) be daily discharge (cfs). For each gauge, compute a **training-era** low-flow threshold:

\[
\tau_{low} = Q_{0.10}\left(q(t)\ \text{for years}\le 2020\right)
\]

Define the future minimum over the next 14 days:

\[
q_{\min}(t) = \min\left(q(t+1),\dots,q(t+14)\right)
\]

Then:

- `target_baseflow_collapse_14d = 1` if:
  - \(q_{\min}(t) \le \tau_{low}\) **and**
  - the past 30-day mean (computed from \(q(t-1..t-30)\)) is **not already** at/below \(\tau_{low}\)
- otherwise 0

This focuses the label on **drops into low-flow regimes**, not “already low” days.

## Inputs

- `train.csv`: features + `target_baseflow_collapse_14d`
- `test.csv`: same features, no target

Notes:

- Gauge ids are anonymized via `station_token`.
- Continuous discharge-derived features are provided as **train-fitted bins** (missing encoded as `-1`).

## Submission format

Create `submission.csv` with exactly:

- `row_id`
- `pred_baseflow_collapse_14d`

## Metric (lower is better)

\[
\text{Score} = 0.55\cdot \text{LogLoss}_{all}
             + 0.30\cdot \text{LogLoss}_{\text{low-baseline slice}}
             + 0.15\cdot (1 - \text{AUPRC}_{all})
\]

The low-baseline slice is provided as `slice_low_baseline`.

Deterministic scoring command:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

## Regime shift / leakage notes

- Split is **time-based** under the hood (newer years emphasized in test).
- Bridge years are assigned deterministically, with **low-baseline conditions overrepresented in test**.
- For honest offline evaluation, avoid random CV; validate by `event_era` and gauge groups.

