"""
Feature engineering module for SCANIA Predictive Maintenance.

Contains functions to:
- Merge operational, TTE, and specification datasets
- Compute counter-based features per vehicle
- Compute histogram-based features (bin stats + totals + centroids)
- Encode specification columns
- Build final per-vehicle feature matrix
- Align validation/test features to match training feature columns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------------------
# Utility: Linear trend (slope) and R²
# --------------------------------------------------------------------------------------

def linear_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a linear regression y = a*x + b.
    Return slope, intercept, and R².
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan

    x = x[mask]
    y = y[mask]

    x_mean = x.mean()
    x_centered = x - x_mean

    slope, intercept = np.polyfit(x_centered, y, 1)
    intercept = intercept - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, r2


# --------------------------------------------------------------------------------------
# Counter-based features
# --------------------------------------------------------------------------------------

def compute_counter_features(group: pd.DataFrame, counter_cols: List[str], time_col: str) -> pd.Series:
    """
    Compute counter-based aggregation features per vehicle.
    """
    out = {}
    time = group[time_col].values.astype(float)

    for col in counter_cols:
        vals = group[col].values.astype(float)
        mask = np.isfinite(vals)

        if mask.sum() == 0:
            out[f"{col}_first"] = np.nan
            out[f"{col}_last"] = np.nan
            out[f"{col}_delta"] = np.nan
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
            out[f"{col}_slope"] = np.nan
            out[f"{col}_r2"] = np.nan
            continue

        vals_valid = vals[mask]
        time_valid = time[mask]

        out[f"{col}_first"] = vals_valid[0]
        out[f"{col}_last"] = vals_valid[-1]
        out[f"{col}_delta"] = vals_valid[-1] - vals_valid[0]
        out[f"{col}_mean"] = vals_valid.mean()
        out[f"{col}_std"] = vals_valid.std()

        slope, intercept, r2 = linear_trend(time_valid, vals_valid)
        out[f"{col}_slope"] = slope
        out[f"{col}_r2"] = r2

    return pd.Series(out)


# --------------------------------------------------------------------------------------
# Histogram-derived features (per row)
# --------------------------------------------------------------------------------------

def add_histogram_derived_columns(df: pd.DataFrame, histogram_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Add histogram-derived columns per row: total mass and centroid.
    """
    df = df.copy()

    for prefix, cols in histogram_groups.items():
        bin_idx = np.arange(len(cols))
        values = df[cols].values.astype(float)

        total = np.nansum(values, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            centroid = np.nansum(values * bin_idx, axis=1) / total

        df[f"{prefix}_total"] = total
        df[f"{prefix}_centroid"] = centroid

    return df


# --------------------------------------------------------------------------------------
# Specification encoding
# --------------------------------------------------------------------------------------

def encode_specifications(df_spec: pd.DataFrame, spec_feature_cols: List[str] = None
                         ) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode specification columns.

    If spec_feature_cols is provided, enforce these columns
    (for validation/test alignment).

    Returns encoded df and the list of one-hot columns (excluding vehicle_id).
    """
    vehicle_col = "vehicle_id"
    spec_cols = [c for c in df_spec.columns if c != vehicle_col]

    df_encoded = pd.get_dummies(df_spec, columns=spec_cols, drop_first=True)

    # Determine or enforce feature columns
    if spec_feature_cols is None:
        # Training: derive from df_encoded
        spec_feature_cols = [c for c in df_encoded.columns if c != vehicle_col]
    else:
        # Validation/test: enforce same columns as training
        for col in spec_feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # category missing in this split

        # Remove any extra unexpected columns
        df_encoded = df_encoded[[vehicle_col] + spec_feature_cols]

    return df_encoded, spec_feature_cols


# --------------------------------------------------------------------------------------
# Build full per-vehicle feature table (TRAIN)
# --------------------------------------------------------------------------------------

def build_train_features(
    df_oper: pd.DataFrame,
    df_tte: pd.DataFrame,
    df_spec: pd.DataFrame,
    counter_cols: List[str],
    histogram_groups: Dict[str, List[str]],
    time_col: str = "time_step",
    vehicle_col: str = "vehicle_id",
    target_col: str = "in_study_repair",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build per-vehicle feature matrix for TRAIN split.
    Returns:
        df_features     : final per-vehicle feature table
        spec_feature_cols : one-hot encoded specification columns
        feature_columns : list of feature columns to enforce on val/test
    """

    # ----- Merge operational + TTE -----
    df_merged = pd.merge(
        df_oper,
        df_tte[[vehicle_col, target_col, "length_of_study_time_step"]],
        on=vehicle_col,
        how="left",
        validate="many_to_one",
    )

    # ----- Merge specs -----
    df_full = pd.merge(
        df_merged,
        df_spec,
        on=vehicle_col,
        how="left",
        validate="many_to_one",
    )

    # ----- Histogram-derived features -----
    df_full = add_histogram_derived_columns(df_full, histogram_groups)
    hist_derived_cols = [c for c in df_full.columns if c.endswith("_total") or c.endswith("_centroid")]

    # ----- Counter features -----
    agg_counters = (
        df_full.groupby(vehicle_col)
        .apply(lambda g: compute_counter_features(g, counter_cols, time_col))
        .reset_index()
    )

    # ----- Histogram bin aggregations -----
    hist_bin_cols = [c for cols in histogram_groups.values() for c in cols]
    agg_hist_bins = df_full.groupby(vehicle_col)[hist_bin_cols].agg(["mean", "std", "min", "max"])
    agg_hist_bins.columns = [f"{c}_{stat}" for (c, stat) in agg_hist_bins.columns.to_flat_index()]
    agg_hist_bins = agg_hist_bins.reset_index()

    # ----- Histogram-derived aggregations -----
    agg_hist_derived = df_full.groupby(vehicle_col)[hist_derived_cols].agg(["mean", "std", "min", "max"])
    agg_hist_derived.columns = [f"{c}_{stat}" for (c, stat) in agg_hist_derived.columns.to_flat_index()]
    agg_hist_derived = agg_hist_derived.reset_index()

    # ----- Study length -----
    study_length = df_tte[[vehicle_col, "length_of_study_time_step"]].copy()

    # ----- Specifications encoding -----
    df_spec_encoded, spec_feature_cols = encode_specifications(df_spec)

    # ----- Merge all blocks -----
    df_features = agg_counters.copy()
    df_features = df_features.merge(agg_hist_bins, on=vehicle_col, how="left")
    df_features = df_features.merge(agg_hist_derived, on=vehicle_col, how="left")
    df_features = df_features.merge(study_length, on=vehicle_col, how="left")
    df_features = df_features.merge(df_spec_encoded, on=vehicle_col, how="left")

    # ----- Add target -----
    df_features = df_features.merge(df_tte[[vehicle_col, target_col]], on=vehicle_col, how="left")

    # ----- Save feature column list (excluding id + target) -----
    feature_columns = [c for c in df_features.columns if c not in [vehicle_col, target_col]]

    return df_features, spec_feature_cols, feature_columns


# --------------------------------------------------------------------------------------
# Build per-vehicle feature table (VALIDATION/TEST)
# --------------------------------------------------------------------------------------

def build_eval_features(
    df_oper: pd.DataFrame,
    df_tte: pd.DataFrame,
    df_spec: pd.DataFrame,
    counter_cols: List[str],
    histogram_groups: Dict[str, List[str]],
    spec_feature_cols: List[str],
    feature_columns: List[str],
    time_col: str = "time_step",
    vehicle_col: str = "vehicle_id",
    target_col: str = "in_study_repair",
) -> pd.DataFrame:
    """
    Build per-vehicle feature matrix for validation/test splits.
    Ensures feature columns match the training set.
    """

    # Repeat same merge logic
    df_merged = pd.merge(
        df_oper,
        df_tte[[vehicle_col, target_col, "length_of_study_time_step"]],
        on=vehicle_col,
        how="left",
        validate="many_to_one",
    )

    df_full = pd.merge(
        df_merged,
        df_spec,
        on=vehicle_col,
        how="left",
        validate="many_to_one",
    )

    df_full = add_histogram_derived_columns(df_full, histogram_groups)
    hist_derived_cols = [c for c in df_full.columns if c.endswith("_total") or c.endswith("_centroid")]

    agg_counters = (
        df_full.groupby(vehicle_col)
        .apply(lambda g: compute_counter_features(g, counter_cols, time_col))
        .reset_index()
    )

    hist_bin_cols = [c for cols in histogram_groups.values() for c in cols]
    agg_hist_bins = df_full.groupby(vehicle_col)[hist_bin_cols].agg(["mean", "std", "min", "max"])
    agg_hist_bins.columns = [f"{c}_{stat}" for (c, stat) in agg_hist_bins.columns.to_flat_index()]
    agg_hist_bins = agg_hist_bins.reset_index()

    agg_hist_derived = df_full.groupby(vehicle_col)[hist_derived_cols].agg(["mean", "std", "min", "max"])
    agg_hist_derived.columns = [f"{c}_{stat}" for (c, stat) in agg_hist_derived.columns.to_flat_index()]
    agg_hist_derived = agg_hist_derived.reset_index()

    study_length = df_tte[[vehicle_col, "length_of_study_time_step"]].copy()

    df_spec_encoded, _ = encode_specifications(df_spec, spec_feature_cols)

    df_features = agg_counters.copy()
    df_features = df_features.merge(agg_hist_bins, on=vehicle_col, how="left")
    df_features = df_features.merge(agg_hist_derived, on=vehicle_col, how="left")
    df_features = df_features.merge(study_length, on=vehicle_col, how="left")
    df_features = df_features.merge(df_spec_encoded, on=vehicle_col, how="left")

    df_features = df_features.merge(df_tte[[vehicle_col, target_col]], on=vehicle_col, how="left")

    # Enforce same feature structure as training
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0  # feature missing => fill with default

    df_features = df_features[[vehicle_col] + feature_columns + [target_col]]

    return df_features
