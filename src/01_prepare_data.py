"""Prepare and harmonize the four source datasets.

Inputs: private raw CSVs from data/raw (or $PREGNANCY_DATA_DIR).
Outputs: data/processed/{gdm,pe,sga,meir}.csv.
No train/test split occurs in this stage.
"""
import argparse, os, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.data.original_sources import process_original_datasets
from utilities.data.meir import process_meir

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--sample',action='store_true'); args=ap.parse_args()
    cfg=load_config(ROOT/args.config)
    names=cfg['files']
    raw = resolve_path(cfg, cfg['data']['sample'] if args.sample else os.getenv('PREGNANCY_DATA_DIR', cfg['data']['raw']))
    required = [names['gdm'], names['pe'], names['sga'], names['meir'], names['meir_pregnancy']]
    missing = [name for name in required if not (raw / name).exists()]
    if missing:
        sample_dir = resolve_path(cfg, cfg['data']['sample'])
        if args.sample:
            raw = sample_dir
        else:
            print(f"The raw data directory is empty or missing required files: {raw}")
            print("Please fill data/raw/ with the private CSV files, or copy the sample files from data/sample/ into data/raw/.")
            print("To run the synthetic demo instead, use: python run_pipeline.py --sample")
            return 0
    out=resolve_path(cfg,cfg['data']['processed']); out.mkdir(parents=True,exist_ok=True)
    gdm=pd.read_csv(raw/names['gdm']); pe=pd.read_csv(raw/names['pe']); sga=pd.read_csv(raw/names['sga'])
    gdm,pe,sga=process_original_datasets(gdm,pe,sga)
    meir=pd.read_csv(raw/names['meir']); preg=pd.read_csv(raw/names['meir_pregnancy'],dtype=str); meir=process_meir(meir,preg)
    for name,df in [('gdm',gdm),('pe',pe),('sga',sga),('meir',meir)]: df.to_csv(out/f'{name}.csv',index=False); print(f'Saved {out/name}.csv: {len(df)} rows')
    return 0
if __name__=='__main__': raise SystemExit(main())
