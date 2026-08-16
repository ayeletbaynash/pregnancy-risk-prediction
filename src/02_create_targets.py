"""Create outcome targets and perform the single 80/20 source-wise train/test split.

Inputs: data/processed/*.csv.
Outputs: data/splits/*_train.csv, *_test.csv, and target_summary.csv.
"""
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.targets import add_zscore_and_risk
from utilities.data.splitting import split_dataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); args=ap.parse_args(); cfg=load_config(ROOT/args.config)
    inp=resolve_path(cfg,cfg['data']['processed']); out=resolve_path(cfg,cfg['data']['splits']); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for source in ['sga','gdm','pe','meir']:
        df=pd.read_csv(inp/f'{source}.csv'); df['source']=source.upper(); df=add_zscore_and_risk(df)
        train,test=split_dataset(df,cfg['split']['test_size'],cfg['split']['random_state'])
        train.to_csv(out/f'{source}_train.csv',index=False); test.to_csv(out/f'{source}_test.csv',index=False)
        for split_name,part in [('train',train),('test',test)]:
            y=pd.to_numeric(part['final_unified_risk'],errors='coerce'); rows.append({'source':source.upper(),'split':split_name,'n':len(part),'labeled':int(y.notna().sum()),'positive':int((y==1).sum()),'negative':int((y==0).sum())})
    pd.DataFrame(rows).to_csv(out/'target_summary.csv',index=False); print(f'Saved splits to {out}')
if __name__=='__main__': main()
