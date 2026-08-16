# Predicting Pregnancy-Related Risks Using Machine Learning

A cross-dataset machine-learning pipeline for identifying pregnancies with a low predicted risk of adverse outcomes. The project combines heterogeneous pregnancy datasets through a shared supervised latent representation and evaluates whether a model trained on several data sources can generalize to a held-out source.

> **Data privacy:** The original datasets contain sensitive medical information and are not distributed in this repository. Full datasets remain in the laboratory's private storage. The files under `data/sample/` are entirely synthetic and exist only to demonstrate the expected schema and pipeline flow.

## Project goal

The project focuses on reliable identification of a low-risk subgroup rather than only maximizing conventional discrimination metrics. The final target combines preeclampsia (PE), severe small-for-gestational-age (SGA), preterm birth, and gestational diabetes mellitus (GDM) when GDM outcome information is available.

Because GDM labels are not available in every source, the representation model uses two supervised heads:

- **Head 3:** PE OR severe SGA OR preterm birth.
- **Head 4:** PE OR severe SGA OR preterm birth OR GDM.

Each training sample is routed only to the head for which its labels are valid.

## Model architecture

```text
Private raw datasets
       │
       ▼
01  Data harmonization
       │
       ▼
02  Target creation + single 80/20 split
       │
       ├────────────── held-out test sets
       ▼
03  Two-head supervised autoencoder
       │
       ├─ reconstruction loss
       ├─ supervised classification loss
       └─ Sinkhorn source-alignment loss
       │
       ▼
04  Shared latent embeddings
       │
       ▼
05  Leave-one-dataset-out MLP prediction
       │
       ▼
06  Evaluation and figures
    ROC / PR / risk-by-percentile / UMAP
```

## Repository structure

```text
config/                 Runtime configuration and final hyperparameters
data/
  raw/                  Private data location (Git-ignored)
  sample/               Synthetic demonstration inputs
  processed/            Harmonized datasets (Git-ignored)
  splits/               Train/test checkpoints (Git-ignored)
  embeddings/           Latent checkpoints (Git-ignored)
models/                  Trained artifacts (Git-ignored by default)
results/
  predictions/           Patient-level predictions (Git-ignored)
  metrics/               Aggregate metrics / curves
  figures/               Evaluation figures
src/
  01_prepare_data.py
  02_create_targets.py
  03_train_embedding.py
  04_encode_datasets.py
  05_train_risk_models.py
  06_evaluate.py
  utilities/             Reusable implementation modules
```

## Pipeline stages

### 1. Data preparation

`src/01_prepare_data.py` harmonizes the GDM, PE, SGA and Meir source tables into a common schema. Dataset-specific renaming and cleaning are isolated under `src/utilities/data/`.

**Inputs:** raw source CSV files.  
**Outputs:** `data/processed/{gdm,pe,sga,meir}.csv`.

### 2. Target creation and splitting

`src/02_create_targets.py` derives severe SGA, preterm birth and unified risk targets, performs two-head target routing, and performs the project's only outer train/test split.

**Outputs:** source-specific 80% train and 20% held-out test files under `data/splits/`.

### 3. Shared representation training

`src/03_train_embedding.py` fits missing-value imputation and scaling on combined training data only, then trains the two-head supervised autoencoder with Sinkhorn alignment between source latent distributions.

**Outputs:** autoencoder weights, fitted preprocessor, selected features, training metadata and loss history.

### 4. Dataset encoding

`src/04_encode_datasets.py` loads the saved encoder and transforms every train/test source split into the shared latent space.

**Outputs:** reusable embedding CSVs under `data/embeddings/`. This is a deliberate checkpoint: downstream classifiers can be replaced without retraining the representation model.

### 5. Risk prediction

`src/05_train_risk_models.py` runs four leave-one-dataset-out experiments:

- SGA + GDM + PE → Meir
- SGA + GDM + Meir → PE
- SGA + PE + Meir → GDM
- GDM + PE + Meir → SGA

A downstream MLP is fitted on training-source embeddings and evaluated only on the held-out source test split.

### 6. Evaluation

`src/06_evaluate.py` performs no training. It creates ROC and precision-recall curves, cumulative observed risk by population percentile, experiment-comparison curves, and UMAP visualizations by source and observed risk.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running with private laboratory data

Place the original files locally under `data/raw/` using the names specified in `config/default.yaml`, or keep them entirely outside the repository:

```bash
export PREGNANCY_DATA_DIR=/path/to/private/lab/data
```

Then run stages independently:

```bash
python src/01_prepare_data.py
python src/02_create_targets.py
python src/03_train_embedding.py
python src/04_encode_datasets.py
python src/05_train_risk_models.py
python src/06_evaluate.py
```

This stage-by-stage design allows work to resume from any major checkpoint without rerunning earlier expensive stages.

## Synthetic demo

No access to medical data is needed to inspect the code path:

```bash
python run_pipeline.py --sample
```

The synthetic demo is intentionally tiny and should **not** be interpreted as reproducing the research results.

## Configuration

All final-model parameters are centralized in `config/default.yaml`. Optuna is intentionally excluded from the main workflow: hyperparameter optimization was part of model development, while this repository exposes the **selected final architecture** directly.

The published configuration corresponds to **Optuna trial 43**, selected by minimizing the mean observed complication rate in the bottom 40% of model-ranked patients across valid validation seeds. The best objective value was **0.07734**. Model selection used seeds **11, 22 and 33**; the public stage-by-stage pipeline defaults to seed **11** for a single reproducible run.

### Selected final hyperparameters

| Component | Parameter | Value |
|---|---|---:|
| Autoencoder | Hidden layers | 64, 32 |
| Autoencoder | Latent dimension | 8 |
| Autoencoder | Reconstruction weight | 0.460569 |
| Autoencoder | Classification weight | 2.886344 |
| Sinkhorn | Weight | 0.025919 |
| Sinkhorn | Epsilon | 0.006867 |
| Sinkhorn | Iterations | 10 |
| Autoencoder | Learning rate | 0.000201369 |
| Autoencoder | Batch size | 512 |
| Autoencoder | Epochs / patience | 150 / 37 |
| MLP | Backend | PyTorch |
| MLP | Hidden layers | 64, 32 |
| MLP | Activation | GELU |
| MLP | Dropout | 0.255531 |
| MLP | Weight decay (`alpha`) | 0.005498 |
| MLP | Learning rate | 0.000100493 |
| MLP | Batch size | 1024 |
| MLP | Epochs / patience | 800 / 48 |

## Evaluation emphasis

In addition to ROC-AUC and average precision, the project evaluates the observed complication rate among patients ranked in the lowest predicted-risk percentiles. In particular, the pipeline reports cumulative observed risk and positive-event capture in the bottom 40% of predicted scores.

## Privacy and responsible use

- No original patient-level dataset is included.
- Synthetic examples are generated independently and contain no patient records.
- Raw, processed, split, embedding and patient-level prediction files are Git-ignored.
- Trained model artifacts are also ignored by default because learned parameters can potentially disclose properties of training data.
- This repository is a research prototype and is **not a clinical decision-support system**.

## Reproducibility notes

The outer source-wise 80/20 split occurs once, before representation training. The autoencoder may create an additional internal validation subset from the training portion solely for early stopping; the held-out test data remain untouched during model fitting.

## Authors

Final-year Computational Biology @ Bar Ilan University / Bioinformatics project.
Made by Gavriel Schwarz and Ayelet Baynash under the supervision of Yoram Luzon.