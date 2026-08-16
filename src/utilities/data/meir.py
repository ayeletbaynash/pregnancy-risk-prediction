"""Preprocessing for the Meir dataset and its pregnancy join table."""
from __future__ import annotations
import re
import pandas as pd

RENAME_MAP = {'age': 'age', 'maternal height': 'height', 'maternal weight before pregnancy': 'weight12', 'BMI': 'bmi', 'DM1 yes no': 'dm_type1', 'DM2 yes no': 'dm_type2', 'CHTN yes no': 'chronic_htn', 'IVF yes no': 'conception_ivf', 'smoking': 'smoking', 'LUPUS yes no': 'sle', 'First_Trimester_HCG': 'hCG', 'First_Trimester_Pappa': 'pappa12', 'multigestational pregnancy yes no': 'multigestational_pregnancy', 'nulliparity': 'nulliparity', 'Had_Previous_PTB': 'prev_ptb', 'Gestational DM': 'gdm', 'Preeclampsia/GHTN': 'pe', 'APLA SYNDROME': 'apla_syndrome', 'Anemia': 'anemia', 'Thalassemia': 'thalassemia', 'hypothyroidism yes no': 'hypothyroidism', 'Hyperthyroidism yes no': 'hyperthyroidism', 'Asthma yes no': 'asthma', 'Depression yes no': 'depression', 'Anxiety yes no': 'anxiety', 'bipolar yes no': 'bipolar', 'epilepsy yes no': 'epilepsy', 'Other rheumatoid disease yes no': 'other_rheumatoid_disease', 'UC or crohn yes no': 'uc_or_crohn', 'G6PD yes no': 'g6pd', '1/First_Trimester_Ds': 'first_trimester_ds', 'URINE GBS AT first trimester': 'urine_gbs_12', 'Vaginal GBS AT FIRST TRIMESTER': 'vaginal_gbs_12', 'SP_CS': 'sp_cs', 'GHTN/preeclampsia': 'ghtn_preeclampsia', 'FAMILIAL MEDITERRANEAN FEVER': 'familial_mediterranean_fever', 'RHEUMATOID ARTHRITIS': 'rheumatoid_arthritis', 'MITRAL VALVE': 'mitral_valve', 'pregnancy_age': 'out_ga', 'pregnancy_weight': 'bw'}
IDENTIFIER_COLUMNS = ["id", "identify"]

def normalize_column_name(col: str) -> str:
    col = col.strip(); col = re.sub(r"[^0-9A-Za-z]+", "_", col); return re.sub(r"_+", "_", col).strip("_").lower()

def unique_column_names(columns):
    counts={}; unique=[]
    for col in columns:
        count=counts.get(col,0); unique.append(col if count==0 else f"{col}_{count+1}"); counts[col]=count+1
    return unique

def join_pregnancy_data(df: pd.DataFrame, pregnancy_df: pd.DataFrame) -> pd.DataFrame:
    if "identify" not in df.columns: return df
    pregnancy_df = pregnancy_df.copy()
    pregnancy_df["ID"] = pregnancy_df["ID"].astype(str).str.strip().str.lstrip("0")
    pregnancy_df["date"] = pregnancy_df["date"].astype(str).str.strip()
    parts=df["identify"].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
    df=df.copy(); df["_join_id"]=parts[0].fillna("").str.strip().str.lstrip("0"); df["_join_date"]=parts[1].fillna("").str.strip()
    df=df.merge(pregnancy_df[["ID","date","Pregnancy_Age","Weight"]], left_on=["_join_id","_join_date"], right_on=["ID","date"], how="left", suffixes=("","_preg"))
    return df.rename(columns={"Pregnancy_Age":"pregnancy_age","Weight":"pregnancy_weight"}).drop(columns=["_join_id","_join_date","ID","date"], errors="ignore")

def process_meir(df: pd.DataFrame, pregnancy_df: pd.DataFrame) -> pd.DataFrame:
    """Join and harmonize Meir into the common project schema."""
    df=join_pregnancy_data(df.copy(), pregnancy_df)
    df.columns=unique_column_names([RENAME_MAP.get(c, normalize_column_name(c)) for c in df.columns])
    for col in df.columns:
        if col in IDENTIFIER_COLUMNS: continue
        if df[col].dtype=="bool": df[col]=df[col].astype(int)
        elif df[col].dtype=="object":
            cleaned=df[col].astype(str).str.strip().str.upper().replace({"TRUE":1,"FALSE":0,"NAN":pd.NA,"NONE":pd.NA,"":pd.NA})
            df[col]=pd.to_numeric(cleaned, errors="coerce")
    pe_cols=[c for c in ["pe","ghtn_preeclampsia"] if c in df.columns]
    if pe_cols:
        df["pe"]=(df[pe_cols].fillna(0).eq(1).any(axis=1)).astype(int); df=df.drop(columns=["ghtn_preeclampsia"], errors="ignore")
    for col in ["migraine","proteinuria"]:
        if col in df.columns: df[col]=df[col].replace(2,1)
    if "height" in df.columns:
        df["height"]=pd.to_numeric(df["height"], errors="coerce"); df.loc[df["height"]<=0,"height"]=pd.NA; df.loc[df["height"]<3,"height"]*=100
    for col in ["weight_before_pregnancy","bmi"]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col], errors="coerce"); df.loc[df[col]<=0,col]=pd.NA
    df=df.dropna(subset=[c for c in ["out_ga","bw"] if c in df.columns])
    if "multigestational_pregnancy" in df.columns: df=df[df["multigestational_pregnancy"] != 1]
    if "lc" in df.columns: df=df[pd.to_numeric(df["lc"], errors="coerce") <= 14]
    if "vbac" in df.columns: df=df[pd.to_numeric(df["vbac"], errors="coerce") <= 10]
    df=df.drop(columns=["prev_ptb_amount","id","anemia_thalassemia","identify","apla_2014_2020","polycystic_ovariesmultiple_sclerosis"], errors="ignore")
    return df.reset_index(drop=True)
