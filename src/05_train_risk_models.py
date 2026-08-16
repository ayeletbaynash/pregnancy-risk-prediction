"""Train leave-one-dataset-out MLP risk models on latent embeddings.

Inputs: data/embeddings/*.csv.
Outputs: saved MLP models, held-out predictions, and per-experiment metrics.
"""
import argparse, sys, json, pickle
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.evaluation.metrics import bottom_percentile_risk,bottom_percentile_positive_capture_rate
from utilities.helpers import save_json
from utilities.models.classifier import train_embedding_mlp
EXPERIMENTS={'sga_gdm_pe_to_meir':(['sga','gdm','pe'],'meir'),'sga_gdm_meir_to_pe':(['sga','gdm','meir'],'pe'),'sga_pe_meir_to_gdm':(['sga','pe','meir'],'gdm'),'gdm_pe_meir_to_sga':(['gdm','pe','meir'],'sga')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--device',default=None); args=ap.parse_args(); cfg=load_config(ROOT/args.config); edir=resolve_path(cfg,cfg['data']['embeddings'])
    pred=ROOT/'results/predictions'; mdir=ROOT/'models/mlp'; pred.mkdir(parents=True,exist_ok=True); mdir.mkdir(parents=True,exist_ok=True); metrics={}; latent_cols=[c for c in pd.read_csv(edir/'sga_train_embedding.csv',nrows=1).columns if c.startswith('sae_latent_')]
    for name,(train_sources,test_source) in EXPERIMENTS.items():
        train=pd.concat([pd.read_csv(edir/f'{s}_train_embedding.csv') for s in train_sources],ignore_index=True); test=pd.read_csv(edir/f'{test_source}_test_embedding.csv')
        model,out,m=train_embedding_mlp(train,test,latent_cols,cfg['split']['random_state'],cfg['mlp'],args.device); m['bottom_40_risk']=bottom_percentile_risk(out,40); m['bottom_40_positive_capture_rate']=bottom_percentile_positive_capture_rate(out,40); metrics[name]=m; out.to_csv(pred/f'{name}.csv',index=False)
        with open(mdir/f'{name}.pkl','wb') as f: pickle.dump(model,f)
    save_json(metrics,ROOT/'results/metrics/experiment_metrics.json'); print('Saved downstream models and predictions.')
if __name__=='__main__': main()
