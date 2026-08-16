"""Convenience runner for the complete public pipeline."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--sample',action='store_true'); p.add_argument('--device',default=None); args=p.parse_args()
steps=['01_prepare_data.py','02_create_targets.py','03_train_embedding.py','04_encode_datasets.py','05_train_risk_models.py','06_evaluate.py']
required_raw = {'GDM_OH.csv','PE.csv','SGA.csv','Meir.csv','pregnancy_merged.csv'}
raw_dir = ROOT / 'data' / 'raw'
if not args.sample and not raw_dir.exists():
    print(f"The raw data directory does not exist: {raw_dir}")
    print("Please fill data/raw/ with the private CSV files, or copy the sample files from data/sample/ into data/raw/.")
    print("To run the synthetic demo instead, use: python run_pipeline.py --sample")
    raise SystemExit(0)
if not args.sample and not ({p.name for p in raw_dir.iterdir() if p.is_file()} & required_raw):
    print(f"The raw folder is empty or missing required files: {raw_dir}")
    print("Please fill data/raw/ with the private CSV files, or copy the sample files from data/sample/ into data/raw/.")
    print("To run the synthetic demo instead, use: python run_pipeline.py --sample")
    raise SystemExit(0)
for i,step in enumerate(steps):
    cmd=[sys.executable,str(ROOT/'src'/step)]
    if i==0 and args.sample: cmd.append('--sample')
    if args.device and step in {'03_train_embedding.py','04_encode_datasets.py','05_train_risk_models.py'}: cmd += ['--device',args.device]
    print('\n>>>',' '.join(cmd)); subprocess.run(cmd,cwd=ROOT,check=True)
