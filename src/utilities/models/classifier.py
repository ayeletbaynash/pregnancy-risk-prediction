"""Downstream MLP classifiers operating on the learned latent space."""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TorchLatentMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


class TorchMLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        activation="relu",
        dropout=0.0,
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=300,
        batch_size=128,
        patience=20,
        random_state=42,
        device_name=None,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.dropout = float(dropout)
        self.learning_rate_init = float(learning_rate_init)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model = None
        if device_name is not None and device_name != "cpu" and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device_name))
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        indices = np.arange(len(y_array))
        y_int = y_array.astype(int)
        stratify = y_int if len(np.unique(y_int)) == 2 and min(np.bincount(y_int)) >= 2 else None
        if len(indices) >= 20 and stratify is not None:
            idx_train, idx_valid = train_test_split(
                indices,
                test_size=0.15,
                random_state=self.random_state,
                shuffle=True,
                stratify=stratify,
            )
        else:
            idx_train, idx_valid = indices, indices

        train_dataset = TensorDataset(
            torch.tensor(X_scaled[idx_train], dtype=torch.float32),
            torch.tensor(y_array[idx_train], dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
        )
        X_valid = torch.tensor(X_scaled[idx_valid], dtype=torch.float32).to(self.device)
        y_valid = torch.tensor(y_array[idx_valid], dtype=torch.float32).to(self.device)

        self.model = TorchLatentMLP(
            input_dim=X_scaled.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        best_state = None
        best_valid_loss = np.inf
        epochs_without_improvement = 0

        for _ in range(1, self.max_iter + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(self.model(X_valid), y_valid).item())
            if valid_loss < best_valid_loss - 1e-5:
                best_valid_loss = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted.")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])


def build_mlp_classifier(random_state=42, mlp_params=None, device_name=None):
    params = {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
        "early_stopping": True,
        "random_state": random_state,
    }
    if mlp_params:
        params.update(mlp_params)
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    params["random_state"] = random_state
    if params.pop("backend", "sklearn") == "torch":
        params.pop("early_stopping", None)
        params["device_name"] = device_name
        return TorchMLPClassifier(**params)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**params)),
        ]
    )


def train_embedding_mlp(train_df, test_df, latent_cols, random_state=42, mlp_params=None, device_name=None):
    train_labeled = train_df[train_df["unified_risk"].notna()].copy()
    test_labeled = test_df[test_df["unified_risk"].notna()].copy()
    if train_labeled.empty:
        raise ValueError("No labeled rows in train set.")

    X_train = train_labeled[latent_cols]
    y_train = train_labeled["unified_risk"].astype(int)
    model = build_mlp_classifier(random_state=random_state, mlp_params=mlp_params, device_name=device_name)
    model.fit(X_train, y_train)

    result = test_df.copy()
    result["predicted_prob"] = model.predict_proba(result[latent_cols])[:, 1]
    result["predicted_label"] = (result["predicted_prob"] > 0.5).astype(int)

    metrics = {}
    if not test_labeled.empty:
        y_true = test_labeled["unified_risk"].astype(int)
        y_prob = result.loc[test_labeled.index, "predicted_prob"]
        y_pred = result.loc[test_labeled.index, "predicted_label"]
        print(classification_report(y_true, y_pred))
        if y_true.nunique() == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
            print(f"AUC Score: {metrics['auc']:.4f}")
            print(f"Average Precision: {metrics['average_precision']:.4f}")
        else:
            print("Skipping AUC/AP - test labels contain one class only")
    else:
        print("No labeled rows in test set to evaluate.")

    return model, result, metrics
