"""Train the two-head Sinkhorn-aligned supervised autoencoder.

Inputs: data/splits/*_train.csv only.
Outputs: autoencoder checkpoint, fitted imputer/scaler, feature list, training history and metadata.
"""
import argparse, sys, json, pickle
from pathlib import Path
import pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.data.features import get_embedding_features, fit_preprocessor
from utilities.helpers import load_source_splits, save_json
from utilities.models.autoencoder import train_supervised_autoencoder
from utilities.evaluation.plots import plot_ae_history

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--device',default=None); args=ap.parse_args(); cfg=load_config(ROOT/args.config)
    splits=load_source_splits(resolve_path(cfg,cfg['data']['splits'])); trains=[splits[f'{s}_train'] for s in ['sga','gdm','pe','meir']]; combined=pd.concat(trains,ignore_index=True)
    features=get_embedding_features(*trains); imputer,scaler,X=fit_preprocessor(combined,features); a=cfg['autoencoder']
    model,history,meta=train_supervised_autoencoder(X,combined['head_3_target'],combined['head_4_target'],combined['source'],random_state=cfg['split']['random_state'],device_name=args.device,**a)
    out=ROOT/'models/autoencoder'; out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/'autoencoder.pt')
    with open(out/'preprocessor.pkl','wb') as f: pickle.dump({'imputer':imputer,'scaler':scaler},f)
    save_json(features,out/'features.json'); save_json({**meta,'input_dim':len(features)},out/'metadata.json')
    hist=pd.DataFrame(history); metrics=ROOT/'results/metrics'; figures=ROOT/'results/figures'; metrics.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True); hist.to_csv(metrics/'ae_training_history.csv',index=False); plot_ae_history(hist,figures)
    print(f'Saved trained embedding model to {out}')
if __name__=='__main__': main()
