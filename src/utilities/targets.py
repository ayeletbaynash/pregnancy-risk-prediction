"""Outcome derivation and two-head target routing.

Preserves the scientific target definitions from the final research pipeline.
"""
import numpy as np
import pandas as pd
from .constants import BASE_RISK_COLS, GDM_OUTCOME_CANDIDATES

def valid_binary_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def first_valid_outcome(df, candidate_cols):
    outcome = pd.Series(np.nan, index=df.index, dtype="float64")
    used_cols = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = valid_binary_series(df[col])
        fill_mask = outcome.isna() & values.notna()
        outcome.loc[fill_mask] = values.loc[fill_mask]
        if values.notna().any():
            used_cols.append(col)
    return outcome, used_cols


def compute_or_target(df, outcome_cols):
    target = pd.Series(pd.NA, index=df.index, dtype="object")
    existing_cols = [col for col in outcome_cols if col in df.columns]
    if not existing_cols:
        return target
    outcomes = df[existing_cols].apply(pd.to_numeric, errors="coerce")
    target.loc[(outcomes == 1).any(axis=1)] = 1
    clean_zero = outcomes.notna().all(axis=1) & target.isna()
    target.loc[clean_zero] = 0
    return target


def add_unified_risk_targets(work, risk_cols=None):
    if risk_cols is None:
        risk_cols = BASE_RISK_COLS

    for col in risk_cols:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = valid_binary_series(work[col])

    gdm_outcome, used_gdm_cols = first_valid_outcome(work, GDM_OUTCOME_CANDIDATES)
    work["gdm_outcome"] = gdm_outcome
    work["has_valid_gdm_outcome"] = work["gdm_outcome"].notna()
    work["gdm_outcome_source_cols"] = ",".join(used_gdm_cols)

    work["unified_risk_3"] = compute_or_target(work, risk_cols)
    work["unified_risk_4"] = compute_or_target(work, list(risk_cols) + ["gdm_outcome"])

    work["final_unified_risk"] = pd.NA
    head_4_mask = work["has_valid_gdm_outcome"]
    work.loc[head_4_mask, "final_unified_risk"] = work.loc[head_4_mask, "unified_risk_4"]
    work.loc[~head_4_mask, "final_unified_risk"] = work.loc[~head_4_mask, "unified_risk_3"]

    work["head_3_target"] = pd.NA
    work["head_4_target"] = pd.NA
    route_head_3 = ~head_4_mask & work["unified_risk_3"].notna()
    route_head_4 = head_4_mask & work["unified_risk_4"].notna()
    work.loc[route_head_3, "head_3_target"] = work.loc[route_head_3, "unified_risk_3"]
    work.loc[route_head_4, "head_4_target"] = work.loc[route_head_4, "unified_risk_4"]
    work["ae_head_route"] = pd.NA
    work.loc[route_head_3, "ae_head_route"] = 3
    work.loc[route_head_4, "ae_head_route"] = 4

    # Keep downstream plotting/evaluation code unchanged: it now means available-outcomes risk.
    work["unified_risk"] = work["final_unified_risk"]
    return work


def add_zscore_and_risk(df, risk_cols=None):
    from scipy.stats import norm

    mu_offset = 199
    mu_int, mu_lin, mu_quad, mu_cub = (3.0893, 0.008350, -0.00002965, -0.00000006062)
    sig_delta_int = 0.02464
    sig_delta_lin = 0.00005640
    sig_e = 0.03363
    z_cutoff_3rd = norm.ppf(0.03)

    work = df.copy()
    work["out_ga"] = pd.to_numeric(work.get("out_ga"), errors="coerce")
    work["bw"] = pd.to_numeric(work.get("bw"), errors="coerce")

    ga_days = work["out_ga"].astype(float) * 7.0
    bw = work["bw"].astype(float)

    ga_c = ga_days - mu_offset
    mu = mu_int + mu_lin * ga_c + mu_quad * (ga_c ** 2) + mu_cub * (ga_c ** 3)
    sigma_delta = sig_delta_int + sig_delta_lin * ga_days
    sd = np.sqrt(sigma_delta ** 2 + sig_e ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        work["z.bw"] = (np.log10(bw) - mu) / sd

    work["sga_st3"] = pd.Series(np.nan, index=work.index, dtype="float64")
    valid_bw_z = work["z.bw"].notna()
    work.loc[valid_bw_z, "sga_st3"] = (work.loc[valid_bw_z, "z.bw"] < z_cutoff_3rd).astype(float)
    work["preterm"] = pd.Series(np.nan, index=work.index, dtype="float64")
    valid_out_ga = work["out_ga"].notna()
    work.loc[valid_out_ga, "preterm"] = (work.loc[valid_out_ga, "out_ga"] < 34).astype(float)

    for col in work.columns:
        if work[col].dtype == "bool":
            work[col] = work[col].astype(float)
        if work[col].dtype == "object" and col != "source":
            work[col] = work[col].astype(str).str.strip().str.upper()
            work[col] = work[col].replace({"TRUE": 1, "FALSE": 0, "NAN": pd.NA, "NONE": pd.NA})
            work[col] = pd.to_numeric(work[col], errors="coerce")

    add_unified_risk_targets(work, risk_cols=risk_cols)

    return work
