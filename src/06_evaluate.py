"""Generate final metrics and figures from saved predictions/embeddings.

Inputs
------
results/predictions/*.csv
data/embeddings/*_test_embedding.csv
models/autoencoder/preprocessor.pkl + features.json (for pre-alignment UMAP)

Outputs
-------
ROC and precision-recall curves, risk-by-percentile curves, experiment comparison,
and three UMAP views: pre-alignment by source, post-Sinkhorn by source, and
post-Sinkhorn by observed unified risk.

This stage performs no training.
"""
import argparse, sys, json, pickle
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.data.features import transform_features
from utilities.evaluation.plots import plot_risk_by_percentile,plot_roc_pr,plot_experiment_comparison,plot_umap
from utilities.helpers import load_source_splits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); args=ap.parse_args(); cfg=load_config(ROOT/args.config)
    pred=ROOT/'results/predictions'; fig=ROOT/'results/figures'; metric_dir=ROOT/'results/metrics'; fig.mkdir(parents=True,exist_ok=True); metric_dir.mkdir(parents=True,exist_ok=True)
    files={p.stem:p for p in pred.glob('*.csv')}
    for name,p in files.items():
        df=pd.read_csv(p)
        plot_risk_by_percentile(df,fig/f'{name}_risk_by_percentile.png',metric_dir/f'{name}_risk_by_percentile.csv')
        plot_roc_pr(df,str(fig/name))
    if files:
        plot_experiment_comparison(files,fig/'experiment_comparison.png')

    # Post-alignment UMAPs from the learned Sinkhorn-aligned latent space.
    edir=resolve_path(cfg,cfg['data']['embeddings'])
    frames=[pd.read_csv(p) for p in sorted(edir.glob('*_test_embedding.csv'))]
    if frames:
        all_test=pd.concat(frames,ignore_index=True)
        latent=[c for c in all_test.columns if c.startswith('sae_latent_')]
        plot_umap(all_test,latent,'source',fig/'umap_after_sinkhorn_by_source.png',cfg['split']['random_state'])
        plot_umap(all_test,latent,'unified_risk',fig/'umap_after_sinkhorn_by_risk.png',cfg['split']['random_state'])

    # Pre-alignment reference UMAP from the standardized input feature space.
    mdir=ROOT/'models/autoencoder'
    if (mdir/'features.json').exists() and (mdir/'preprocessor.pkl').exists():
        features=json.load(open(mdir/'features.json'))
        prep=pickle.load(open(mdir/'preprocessor.pkl','rb'))
        splits=load_source_splits(resolve_path(cfg,cfg['data']['splits']))
        before=[]
        for source in ['sga','gdm','pe','meir']:
            df=splits[f'{source}_test'].copy()
            X=transform_features(df,features,prep['imputer'],prep['scaler'])
            temp=pd.DataFrame(X,columns=[f'input_{i+1}' for i in range(X.shape[1])])
            temp['source']=source.upper(); before.append(temp)
        before=pd.concat(before,ignore_index=True)
        input_cols=[c for c in before.columns if c.startswith('input_')]
        plot_umap(before,input_cols,'source',fig/'umap_before_sinkhorn_by_source.png',cfg['split']['random_state'])
    print(f'Figures written to {fig}')
if __name__=='__main__': main()
