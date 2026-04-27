from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE_DIR = (
    ROOT / "competition_usgs_streamflow_floodwatch_3day_regime_shift" / "_cache"
)

ID_COLUMN = "row_id"
STATION_COLUMN = "station_token"
TARGET_COLUMN = "target_baseflow_collapse_14d"
PRED_COLUMN = "pred_baseflow_collapse_14d"
SLICE_COLUMN = "slice_low_baseline"

N_BINS = 12
MIN_TRAIN_ROWS = 12_000
EPS = 1e-9


@dataclass(frozen=True)
class Config:
    # Daily values date range in cache files.
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"

    # Label horizon: baseflow collapse risk.
    horizon_days: int = 14
    lowflow_quantile: float = 0.10

    # Past-only feature windows.
    lb_short: int = 7
    lb_long: int = 30
    lb_season: int = 90

    # Deterministic split.
    train_max_year: int = 2020
    test_min_year: int = 2023
    bridge_years: Tuple[int, ...] = (2021, 2022)
    bridge_test_rate_lowbase: int = 75
    bridge_test_rate_other: int = 30

    # Deterministic sampling (stable but keeps files smaller).
    keep_percent_train: int = 70
    keep_percent_test: int = 92

    # Site cap (keeps build runtime bounded even if cache grows).
    max_sites: int = 60


def _hash_percent(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 100


def _station_token(site_no: str) -> str:
    return hashlib.sha256(site_no.encode("utf-8")).hexdigest()[:12]


def _row_id(station_token: str, date: pd.Timestamp) -> int:
    s = f"bf14_{station_token}_{date:%Y-%m-%d}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:15]
    return int(h, 16)


def _read_dv_rdb(path: Path, parameter_cd: str = "00060") -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])

    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[2:]]  # skip types row
    df = pd.DataFrame(rows, columns=header)
    if "datetime" not in df.columns or "site_no" not in df.columns:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])

    matches = [c for c in df.columns if f"_{parameter_cd}_" in c]
    if not matches:
        return pd.DataFrame(columns=["date", "q_cfs", "site_no"])
    value_col = matches[0]

    out = pd.DataFrame(
        {
            "site_no": df["site_no"].astype(str),
            "date": pd.to_datetime(df["datetime"], errors="coerce"),
            "q_cfs": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out[out["date"].notna()].copy()
    out = out.drop_duplicates(subset=["site_no", "date"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.floor("D")
    return out


def _nanquantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([-1.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(x, qs)
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        v = float(edges[0]) if edges.size else 0.0
        return np.array([v - 1.0, v + 1.0], dtype=float)
    return edges


def _bin_with_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], -1, dtype=np.int16)
    mask = np.isfinite(arr)
    if mask.any():
        idx = np.digitize(arr[mask], edges, right=False) - 1
        idx = np.clip(idx, 0, max(0, edges.size - 2))
        out[mask] = idx.astype(np.int16)
    return pd.Series(out, index=series.index, dtype="Int16")


def _engineer_site(site_no: str, dv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if dv.empty:
        return pd.DataFrame()
    df = dv.copy()
    df = df[df["site_no"].astype(str) == str(site_no)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    # Reindex to a complete daily calendar to expose gaps/missingness.
    start = df["date"].min()
    end = df["date"].max()
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.DataFrame()
    full = pd.date_range(start=start, end=end, freq="D")
    df = df.set_index("date").reindex(full)
    df.index.name = "date"
    df = df.reset_index()
    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")

    # Coarsened calendar covariates.
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["event_era"] = pd.cut(
        df["year"],
        bins=[2013, 2016, 2019, 2022, 2025],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)
    df["season_bin"] = pd.cut(
        df["month"],
        bins=[0, 3, 6, 9, 12],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    ).astype(str)

    # Past-only rolling features.
    s = df["q_cfs"].shift(1)
    df["q_1d"] = s
    df["q_mean_7d"] = s.rolling(cfg.lb_short).mean()
    df["q_mean_30d"] = s.rolling(cfg.lb_long).mean()
    df["q_std_30d"] = s.rolling(cfg.lb_long).std()
    df["q_min_30d"] = s.rolling(cfg.lb_long).min()
    df["q_max_30d"] = s.rolling(cfg.lb_long).max()
    df["q_mean_90d"] = s.rolling(cfg.lb_season).mean()
    df["q_missing_30d"] = s.rolling(cfg.lb_long).apply(lambda x: float(np.isnan(x).sum()), raw=True)

    df["logq_1d"] = np.log(np.maximum(df["q_1d"].to_numpy(dtype=float), 0.0) + 1.0)
    df["logq_mean_30d"] = np.log(np.maximum(df["q_mean_30d"].to_numpy(dtype=float), 0.0) + 1.0)
    df["cv_30d"] = df["q_std_30d"] / (df["q_mean_30d"].abs() + EPS)
    df["range_30d"] = (df["q_max_30d"] - df["q_min_30d"]) / (df["q_mean_30d"].abs() + EPS)
    df["ddown_30d"] = (df["q_1d"] - df["q_min_30d"]) / (df["q_mean_30d"].abs() + EPS)

    # Station token (anonymized, stable).
    token = _station_token(str(site_no))
    df[STATION_COLUMN] = token

    # Train-era low-flow threshold per station (<= train_max_year).
    train_mask = df["year"].astype(int) <= cfg.train_max_year
    train_vals = df.loc[train_mask, "q_cfs"].to_numpy(dtype=float)
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size < 365:
        return pd.DataFrame()
    thr_low = float(np.nanquantile(train_vals, cfg.lowflow_quantile))
    if not np.isfinite(thr_low):
        return pd.DataFrame()

    # Future minimum over horizon (next H days).
    h = int(cfg.horizon_days)
    future_min = df["q_cfs"].shift(-1).rolling(h).min().shift(-(h - 1))
    df["future_min_h"] = future_min

    # Baseflow collapse: future hits station low-flow threshold,
    # but the past-long average isn't already at/below that level.
    baseline_ok = df["q_mean_30d"].to_numpy(dtype=float) >= thr_low
    df[TARGET_COLUMN] = (
        (baseline_ok)
        & (df["future_min_h"].to_numpy(dtype=float) <= thr_low)
    ).astype(int)

    # Slice: station-specific low baseline based on train-era q_mean_30d.
    base_vals = df.loc[train_mask, "q_mean_30d"].to_numpy(dtype=float)
    base_vals = base_vals[np.isfinite(base_vals)]
    slice_thr = float(np.quantile(base_vals, 0.25)) if base_vals.size else float("-inf")
    df[SLICE_COLUMN] = (df["q_mean_30d"].to_numpy(dtype=float) <= slice_thr).astype(int)

    # Keep only rows with required features + future window.
    req = [
        "q_1d",
        "q_mean_7d",
        "q_mean_30d",
        "q_std_30d",
        "q_min_30d",
        "q_max_30d",
        "q_mean_90d",
        "q_missing_30d",
        "logq_1d",
        "logq_mean_30d",
        "cv_30d",
        "range_30d",
        "ddown_30d",
        "future_min_h",
    ]
    df = df[np.isfinite(df["future_min_h"].to_numpy(dtype=float))].copy()
    df = df.dropna(subset=["q_mean_30d", "q_1d"]).copy()
    df["missing_count"] = df[req].isna().sum(axis=1).astype(int)

    # Drop rows with too much missingness in the lookback window.
    df = df[df["q_missing_30d"].fillna(9999) <= 5].copy()
    if df.empty:
        return pd.DataFrame()

    df[ID_COLUMN] = df["date"].map(lambda d: _row_id(token, pd.Timestamp(d))).astype("int64")
    return df


def _assign_split(data: pd.DataFrame, cfg: Config) -> pd.Series:
    years = data["date"].dt.year.astype(int)
    is_test = years >= cfg.test_min_year

    # Bridge years: push low-baseline days into test more often.
    key = data[STATION_COLUMN].astype(str) + "_" + data["dayofyear"].astype(int).astype(str)
    key_pct = key.map(_hash_percent).astype(int)
    is_low = data[SLICE_COLUMN].astype(int) > 0

    is_test |= years.isin(cfg.bridge_years) & is_low & (key_pct < cfg.bridge_test_rate_lowbase)
    is_test |= years.isin(cfg.bridge_years) & (~is_low) & (key_pct < cfg.bridge_test_rate_other)
    is_test &= years > cfg.train_max_year
    return is_test


def main() -> None:
    cfg = Config()

    if not UPSTREAM_CACHE_DIR.exists():
        raise FileNotFoundError(f"Missing upstream cache dir: {UPSTREAM_CACHE_DIR}")

    rdb_files = sorted(UPSTREAM_CACHE_DIR.glob("dv_*_00060_*.rdb"))
    if not rdb_files:
        raise RuntimeError("No cached USGS DV .rdb files found.")

    # Deterministic site selection by filename hash.
    keyed = []
    for p in rdb_files:
        keyed.append((int(hashlib.md5(p.name.encode("utf-8")).hexdigest()[:8], 16), p))
    keyed.sort(key=lambda t: t[0])
    rdb_files = [p for _, p in keyed[: int(cfg.max_sites)]]

    site_frames = []
    for p in rdb_files:
        dv = _read_dv_rdb(p, parameter_cd="00060")
        if dv.empty:
            continue
        site_no = str(dv["site_no"].iloc[0])
        ex = _engineer_site(site_no, dv, cfg)
        if not ex.empty:
            site_frames.append(ex)

    if not site_frames:
        raise RuntimeError("No examples engineered from cache files.")

    data = pd.concat(site_frames, axis=0, ignore_index=True)
    data = data.sort_values([STATION_COLUMN, "date"]).reset_index(drop=True)

    is_test = _assign_split(data, cfg)

    # Deterministic sampling.
    keep_key = data[STATION_COLUMN].astype(str) + "|" + data["date"].dt.strftime("%Y-%m-%d")
    keep_pct = keep_key.map(_hash_percent).astype(int)
    keep = (~is_test & (keep_pct < cfg.keep_percent_train)) | (is_test & (keep_pct < cfg.keep_percent_test))
    data = data[keep].reset_index(drop=True)
    is_test = is_test[keep].reset_index(drop=True)

    if data.empty:
        raise RuntimeError("No rows after sampling.")

    train_df = data[~is_test].copy()
    test_df = data[is_test].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Build produced empty train or test split.")

    # Train-only bin edges.
    to_bin = [
        "q_1d",
        "q_mean_7d",
        "q_mean_30d",
        "q_std_30d",
        "q_min_30d",
        "q_max_30d",
        "q_mean_90d",
        "q_missing_30d",
        "logq_1d",
        "logq_mean_30d",
        "cv_30d",
        "range_30d",
        "ddown_30d",
    ]
    edges: Dict[str, list] = {}
    for c in to_bin:
        e = _nanquantile_edges(train_df[c].to_numpy(dtype=float), N_BINS)
        edges[c] = e.tolist()
        data[f"{c}_bin"] = _bin_with_edges(data[c], e)

    data["missing_count"] = data[to_bin].isna().sum(axis=1).astype(int)

    # Permute to avoid joinable ordering.
    perm_key = (
        data[STATION_COLUMN].astype(str)
        + "|"
        + data["date"].dt.strftime("%Y-%m-%d")
    ).map(lambda s: int(hashlib.md5(("perm:" + s).encode("utf-8")).hexdigest()[:8], 16))
    order = np.argsort(perm_key.to_numpy(dtype=np.int64), kind="mergesort")
    data = data.iloc[order].reset_index(drop=True)
    is_test = is_test.iloc[order].reset_index(drop=True)

    out_cols = [
        ID_COLUMN,
        STATION_COLUMN,
        "event_era",
        "season_bin",
        "month",
        "dow",
        "is_weekend",
        SLICE_COLUMN,
        "missing_count",
    ] + [f"{c}_bin" for c in to_bin]

    out = data[out_cols + [TARGET_COLUMN]].copy()
    train_out = out[~is_test].copy()
    test_out = out[is_test].copy()

    if len(train_out) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train_out)} rows). Need >= {MIN_TRAIN_ROWS}.")

    # Write outputs.
    train_out.to_csv(COMP_DIR / "train.csv", index=False)
    test_out[out_cols].to_csv(COMP_DIR / "test.csv", index=False)
    test_out[[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].to_csv(COMP_DIR / "solution.csv", index=False)

    sample = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: 0.5})
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)

    y = test_out[TARGET_COLUMN].astype(int).to_numpy()
    perf_p = np.where(y == 1, 0.999, 0.001)
    perfect = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: perf_p})
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "source_cache_dir": str(UPSTREAM_CACHE_DIR),
        "parameter_cd": "00060",
        "horizon_days": int(cfg.horizon_days),
        "lowflow_quantile": float(cfg.lowflow_quantile),
        "n_bins": int(N_BINS),
        "bin_edges": edges,
        "split": {
            "train_max_year": int(cfg.train_max_year),
            "bridge_years": list(cfg.bridge_years),
            "test_min_year": int(cfg.test_min_year),
            "bridge_test_rate_lowbaseline": int(cfg.bridge_test_rate_lowbase),
            "bridge_test_rate_other": int(cfg.bridge_test_rate_other),
            "keep_percent_train": int(cfg.keep_percent_train),
            "keep_percent_test": int(cfg.keep_percent_test),
        },
        "row_counts": {"train": int(len(train_out)), "test": int(len(test_out))},
        "positive_rate": {
            "train": float(train_out[TARGET_COLUMN].mean()),
            "test": float(test_out[TARGET_COLUMN].mean()),
        },
        "sites_used": int(len(set(data[STATION_COLUMN].astype(str).tolist()))),
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", int(len(train_out)), "test rows:", int(len(test_out)))
    print("train positive rate:", float(train_out[TARGET_COLUMN].mean()))
    print("test positive rate:", float(test_out[TARGET_COLUMN].mean()))


if __name__ == "__main__":
    main()

