## Overview

This competition is derived from **USGS National Water Information System (NWIS)** daily discharge (parameter `00060`) time series. The task is a drought-adjacent operational proxy: forecast whether a gauge will experience a **baseflow collapse** (drop into its station-specific low-flow regime) within the next **14 days**.

The task is intentionally challenging because:

- the label is **station-relative** (training-era 10th percentile thresholds),
- the split creates **training–serving skew** (newer years + low-baseline bridge oversampling in test),
- low-flow dynamics are seasonal and non-stationary,
- the metric emphasizes **calibration** (LogLoss) and a deterministic low-baseline slice.

## Source

- USGS NWIS Water Services (Daily Values / DV): `https://waterservices.usgs.gov/nwis/dv/`

The build reads cached DV `.rdb` files already present in this workspace.

## License

Public-Domain-US-Gov

USGS data products are generally in the U.S. public domain. See: `https://www.usgs.gov/data-management/data-licensing`

## Features

Each row is a gauge-day example.

- **Identifiers / grouping**:
  - `row_id`
  - `station_token`: anonymized gauge identifier
- **Coarsened time context**:
  - `event_era` (bucketed year range)
  - `season_bin`, `month`, `dow`, `is_weekend`
- **Discharge history features (binned, past-only)**:
  - lag/rolling mean/variance: `q_*_bin`
  - variability and shape: `cv_30d_bin`, `range_30d_bin`, `ddown_30d_bin`
  - missingness proxies: `q_missing_30d_bin`, `missing_count`
- **Slice**:
  - `slice_low_baseline`: station-specific low baseline condition (threshold learned from training-era history)

## Splitting & Leakage

- **Split policy (deterministic, time-based)**:
  - Train: years \(\le 2020\)
  - Test: years \(\ge 2023\)
  - Bridge years (2021–2022): deterministic hashed assignment, with **low-baseline** days placed in test more often
- **Leakage risks**:
  - random CV can hide regime shift and inflate performance
  - fitting preprocessing/calibration on all data leaks test distribution
- **Mitigations**:
  - validate by `event_era` (blocked time splits)
  - fit any calibration / preprocessing on training folds only

