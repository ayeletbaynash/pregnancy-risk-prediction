"""Preprocessing for the original GDM, PE and SGA datasets.

This module contains no file-system side effects; callers provide DataFrames and save outputs explicitly.
"""
from __future__ import annotations
import pandas as pd

COLS_TO_REMOVE = ['id sort', 'Scan day', 'Scan month', 'scan year', 'Prediction', 'chr_No', 'smoking_No', 'dm_No', 'sle_No', 'fh_No', 'Outcome Live birth', 'outcome_Live_Birth', 'Outcome Live birth NND', 'Outcome Stillbirth', 'outcome_Stillbirth-antenatal', 'outcome_Stillbirth-intrapartum', 'SGA_st10_', 'SGA_st10_SGA', 'Race ']
TO_COMBINE = {'dm_type1': ['dm_Type_1_DM', 'dm_Type_1_DM_'], 'chronic_htn': ['chr_Chronic_hypertension', 'chr_Chronic_hypertension_'], 'pe': ['PE_a_PE', 'PE_a_PIH', 'PE/PH a PE', 'PE/PH b PIH']}
RENAME_MAP = {'Age': 'age', 'age': 'age', 'Height': 'height', 'ht': 'height', 'height': 'height', 'Weight': 'weight12', 'wt': 'weight12', 'weight_22': 'weight22', 'BMI 12w': 'bmi', 'BMI': 'bmi', 'Race Black': 'race_black', 'race_Black': 'race_black', 'Race East Asian': 'race_east_asian', 'race_East Asian': 'race_east_asian', 'race_East_Asian': 'race_east_asian', 'Race Mixed': 'race_mixed', 'race_Mixed': 'race_mixed', 'Race South Asian': 'race_south_asian', 'race_South Asian': 'race_south_asian', 'race_South_Asian': 'race_south_asian', 'Race White': 'race_white', 'race_White': 'race_white', 'Conception IVF': 'conception_ivf', 'conception_IVF': 'conception_ivf', 'conception__IVF': 'conception_ivf', 'Conception Ovulation drugs': 'conception_ovulation_drugs', 'conception_Ovulation drugs': 'conception_ovulation_drugs', 'conception__Ovulation_drugs': 'conception_ovulation_drugs', 'Conception Spontaneous': 'conception_spontaneous', 'conception_Spontaneous': 'conception_spontaneous', 'conception__Spontaneous': 'conception_spontaneous', 'no vs yes': 'smoking', 'smoking_Smoker': 'smoking', 'Chronic hypertension vs no': 'chronic_htn', 'Chronic hypertension_No': 'chronic_htn', 'chronic_htn': 'chronic_htn', 'FH DM 1st degree': 'FH_dm_type1', 'Diabetes_Type 1 DM': 'dm_type1', 'dm_type1': 'dm_type1', 'FH DM 2nd degree': 'FH_dm_type2', 'Diabetes_Type 2 DM': 'dm_type2', 'dm_Type_2_DM': 'dm_type2', 'SLE_SLE/APS': 'sle', 'sle_SLE/APS': 'sle', 'out ga': 'out_ga', 'out.ga': 'out_ga', 'BW': 'bw', 'bw': 'bw', 'last.ga': 'last_out_ga', 'last.bwzscore': 'last_bw_z', 'last.bwcent': 'last_bw_cent', 'CRL': 'crl12', 'crl12': 'crl12', 'crl22': 'crl22', 'PAPP-A': 'pappa12', 'pappa': 'pappa12', 'pappa12': 'pappa12', ' vs Delfia': 'pappa_Delfia', 'Ut PI': 'utpi12', 'utpi': 'utpi12', 'u12': 'utpi12', 'u22': 'utpi22', 'ga': 'ga12', 'ga12': 'ga12', 'ga22': 'ga22', 'GDM GA': 'ga_discover_of_gdm', 'PE/PH a PE': 'pe', 'PE/PH b PIH': 'pih', 'PE/PH no': 'no_pe', 'PE_': 'no_pe', 'PE_a_PE': 'pe', 'PE_a_PIH': 'pih', 'pe_vs_noPe': 'pe', 'pePreterm_vs_noPe': 'pe_preterm', 'pePreterm_vs_noPe_and_fullTerm': 'pe_preterm_fullterm', 'fh_FH-PE_mother': 'fh_pe_mother', 'fh_FH-PE_sister': 'fh_pe_sister', 'Previous GDM Multip - GDM': 'prev_gdm', 'Previous GDM Multip - no GDM': 'prev_no_gdm', 'Previous GDM Nullip': 'nulliparity', 'Previous FGR Multip - FGR': 'prev_fgr', 'Previous FGR Multip - no FGR': 'prev_no_fgr', 'Previous FGR Nullip': 'nulliparity', 'Previous LGA Multip - LGA': 'prev_lga', 'Previous LGA Multip - no LGA': 'prev_no_lga', 'Previous LGA Nullip': 'nulliparity', 'Previous_IUD_Multip-IUD': 'prev_iud', 'Previous_IUD_Multip-no_IUD': 'prev_no_iud', 'Previous_IUD_Nullip': 'nulliparity', 'Previous PE_Multip-PE': 'prev_pe', 'prev.pe_Multip-PE': 'prev_pe', 'Previous PE_Multip-no PE': 'prev_pe_no', 'prev.pe_Multip-no_PE': 'prev_pe_no', 'Previous PE_Nullip': 'prev_pe_nullip', 'prev.pe_Nullip': 'prev_pe_nullip'}

def merge_duplicate_binary_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dup_cols = df.loc[:, df.columns == col]
    if dup_cols.shape[1] <= 1:
        return df
    dup_cols = dup_cols.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    conflicts = dup_cols.nunique(axis=1, dropna=True) > 1
    if conflicts.any():
        print(f"warning: {int(conflicts.sum())} rows have conflicting values in {col}")
    merged = dup_cols.max(axis=1, skipna=True)
    df = df.loc[:, df.columns != col].copy()
    df[col] = merged
    return df

def _normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        elif df[col].dtype == "object":
            cleaned = df[col].astype(str).str.strip().str.upper()
            cleaned = cleaned.replace({"TRUE": 1, "FALSE": 0, "NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df

def process_original_datasets(gdm: pd.DataFrame, pe: pd.DataFrame, sga: pd.DataFrame):
    """Harmonize the three original source tables into the project's shared schema."""
    df1, df2, df3 = gdm.copy(), pe.copy(), sga.copy()
    if {"Race ", "Outcome Live birth NND", "Outcome Stillbirth"}.issubset(df1.columns):
        df1 = df1[(df1["Race "] != 1) & (df1["Outcome Live birth NND"] != 1) & (df1["Outcome Stillbirth"] != 1)]
    if {"outcome_Stillbirth-antenatal", "outcome_Stillbirth-intrapartum"}.issubset(df3.columns):
        df3 = df3[(df3["outcome_Stillbirth-antenatal"] != 1) & (df3["outcome_Stillbirth-intrapartum"] != 1)]
    for df in [df1, df2, df3]:
        df.drop(columns=COLS_TO_REMOVE, errors="ignore", inplace=True)
        for target_name, source_cols in TO_COMBINE.items():
            existing = [c for c in source_cols if c in df.columns]
            if existing:
                df[target_name] = df[existing].apply(pd.to_numeric, errors="coerce").max(axis=1)
                df.drop(columns=existing, inplace=True)
    if "ga" in df2.columns:
        df2["ga"] = pd.to_numeric(df2["ga"], errors="coerce") / 7.0
    if "z.bw" in df3.columns:
        df3["sga_st3"] = (pd.to_numeric(df3["z.bw"], errors="coerce") < -1.88).astype(int)
    df1, df2, df3 = map(_normalize_values, [df1, df2, df3])
    if "no vs yes" in df1.columns:
        df1["no vs yes"] = df1["no vs yes"].replace({0:1,1:0})
    if "Chronic hypertension_No" in df2.columns:
        df2["Chronic hypertension_No"] = df2["Chronic hypertension_No"].replace({1:0,0:1})
    df1 = df1.rename(columns=RENAME_MAP)
    df2 = df2.rename(columns=RENAME_MAP)
    df3 = df3.rename(columns=RENAME_MAP)
    df1 = merge_duplicate_binary_column(df1, "nulliparity")
    return df1.reset_index(drop=True), df2.reset_index(drop=True), df3.reset_index(drop=True)
