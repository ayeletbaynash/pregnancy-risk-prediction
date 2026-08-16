"""Encode all source train/test splits into the shared latent representation.

Inputs: split CSVs + saved autoencoder/preprocessor.
Outputs: data/embeddings/*_embedding.csv.
"""
import argparse, sys, json, pickle
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from utilities.config import load_config, resolve_path
from utilities.data.features import transform_features
from utilities.helpers import load_source_splits
from utilities.models.autoencoder import SupervisedAutoencoder, encode_latent, add_latent_columns

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--device',default=None); args=ap.parse_args(); cfg=load_config(ROOT/args.config)
    mdir=ROOT/'models/autoencoder'; features=json.load(open(mdir/'features.json')); meta=json.load(open(mdir/'metadata.json')); prep=pickle.load(open(mdir/'preprocessor.pkl','rb'))
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu')); model=SupervisedAutoencoder(meta['input_dim'],meta['hidden_layer_sizes'],meta['latent_dim']).to(device); model.load_state_dict(torch.load(mdir/'autoencoder.pt',map_location=device)); model.eval()
    splits=load_source_splits(resolve_path(cfg,cfg['data']['splits'])); out=resolve_path(cfg,cfg['data']['embeddings']); out.mkdir(parents=True,exist_ok=True); latent_cols=[f'sae_latent_{i+1}' for i in range(meta['latent_dim'])]
    for name,df in splits.items():
        X=transform_features(df,features,prep['imputer'],prep['scaler']); emb=add_latent_columns(df,encode_latent(model,X,latent_cols)); emb.to_csv(out/f'{name}_embedding.csv',index=False)
    print(f'Saved embeddings to {out}')
if __name__=='__main__': main()
