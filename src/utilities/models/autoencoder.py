"""Two-head supervised autoencoder with Sinkhorn source alignment."""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utilities.constants import SOURCE_TO_ID

def normalize_ae_hidden_layer_sizes(hidden_layer_sizes=None, hidden_dim=None):
    if hidden_layer_sizes is None:
        hidden_layer_sizes = (hidden_dim if hidden_dim is not None else 64,)
    elif isinstance(hidden_layer_sizes, (int, np.integer)):
        hidden_layer_sizes = (int(hidden_layer_sizes),)
    else:
        hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes)
    if not hidden_layer_sizes or any(size <= 0 for size in hidden_layer_sizes):
        raise ValueError("AE hidden_layer_sizes must contain one or more positive integers.")
    return hidden_layer_sizes


class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=None, latent_dim=16, hidden_dim=None):
        super().__init__()
        hidden_layer_sizes = normalize_ae_hidden_layer_sizes(hidden_layer_sizes, hidden_dim)

        encoder_layers = []
        prev_dim = input_dim
        for hidden_size in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(prev_dim, hidden_size))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_size
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_size in reversed(hidden_layer_sizes):
            decoder_layers.append(nn.Linear(prev_dim, hidden_size))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_size
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.head_3 = nn.Linear(latent_dim, 1)
        self.head_4 = nn.Linear(latent_dim, 1)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        logits_3 = self.head_3(latent).squeeze(1)
        logits_4 = self.head_4(latent).squeeze(1)
        return reconstructed, logits_3, logits_4, latent


def make_tensor_loader(X, y_head_3, y_head_4, source=None, batch_size=256, shuffle=False):
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y3_array = pd.to_numeric(y_head_3, errors="coerce").to_numpy(dtype=np.float32)
    y4_array = pd.to_numeric(y_head_4, errors="coerce").to_numpy(dtype=np.float32)
    y3_tensor = torch.tensor(y3_array, dtype=torch.float32)
    y4_tensor = torch.tensor(y4_array, dtype=torch.float32)
    if source is None:
        source_tensor = torch.full((len(X),), -1, dtype=torch.long)
    else:
        source_array = source.map(SOURCE_TO_ID).to_numpy(dtype=np.int64)
        source_tensor = torch.tensor(source_array, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y3_tensor, y4_tensor, source_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def source_pairwise_loss(latent, source_ids, pair_loss_fn):
    valid_mask = source_ids >= 0
    latent = latent[valid_mask]
    source_ids = source_ids[valid_mask]
    source_values = torch.unique(source_ids)
    if len(source_values) < 2:
        return torch.tensor(0.0, device=latent.device)

    losses = []
    for i, source_a in enumerate(source_values):
        for source_b in source_values[i + 1 :]:
            z_a = latent[source_ids == source_a]
            z_b = latent[source_ids == source_b]
            if z_a.shape[0] >= 2 and z_b.shape[0] >= 2:
                losses.append(pair_loss_fn(z_a, z_b))
    if not losses:
        return torch.tensor(0.0, device=latent.device)
    return torch.stack(losses).mean()


def sinkhorn_transport_cost(x, y, cost_scale, epsilon=0.1, n_iters=30):
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    cost = torch.cdist(x, y, p=2).pow(2)
    cost = cost / cost_scale.clamp_min(1e-6)
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-12)
    a = torch.full((x.shape[0],), 1.0 / x.shape[0], dtype=x.dtype, device=x.device)
    b = torch.full((y.shape[0],), 1.0 / y.shape[0], dtype=y.dtype, device=y.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.T @ u).clamp_min(1e-12)
    transport = u[:, None] * kernel * v[None, :]
    return torch.sum(transport * cost)


def sinkhorn_loss_two_sample(x, y, epsilon=0.1, n_iters=30):
    with torch.no_grad():
        combined = torch.cat([x, y], dim=0)
        dists = torch.cdist(combined, combined, p=2).pow(2)
        nonzero_dists = dists[dists > 0]
        if nonzero_dists.numel() == 0:
            cost_scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        else:
            cost_scale = nonzero_dists.mean()
    xy = sinkhorn_transport_cost(x, y, cost_scale, epsilon=epsilon, n_iters=n_iters)
    xx = sinkhorn_transport_cost(x, x, cost_scale, epsilon=epsilon, n_iters=n_iters)
    yy = sinkhorn_transport_cost(y, y, cost_scale, epsilon=epsilon, n_iters=n_iters)
    return xy - 0.5 * xx - 0.5 * yy


def source_sinkhorn_loss(latent, source_ids, epsilon=0.1, n_iters=30):
    return source_pairwise_loss(
        latent,
        source_ids,
        lambda z_a, z_b: sinkhorn_loss_two_sample(z_a, z_b, epsilon=epsilon, n_iters=n_iters),
    )


def two_head_loss_components(
    model,
    xb,
    y3b,
    y4b,
    source_ids,
    device,
    mse,
    bce_none,
    reconstruction_weight,
    classification_weight,
    sinkhorn_weight,
    sinkhorn_epsilon,
    sinkhorn_iters,
):
    reconstructed, logits_3, logits_4, latent = model(xb)
    recon_loss = mse(reconstructed, xb)

    mask_3 = ~torch.isnan(y3b)
    mask_4 = ~torch.isnan(y4b)
    n3 = int(mask_3.sum().item())
    n4 = int(mask_4.sum().item())
    cls_sum = torch.tensor(0.0, device=device)
    head_3_loss = torch.tensor(0.0, device=device)
    head_4_loss = torch.tensor(0.0, device=device)

    if n3:
        head_3_losses = bce_none(logits_3[mask_3], y3b[mask_3])
        head_3_loss = head_3_losses.mean()
        cls_sum = cls_sum + head_3_losses.sum()
    if n4:
        head_4_losses = bce_none(logits_4[mask_4], y4b[mask_4])
        head_4_loss = head_4_losses.mean()
        cls_sum = cls_sum + head_4_losses.sum()

    n_cls = n3 + n4
    if n_cls:
        cls_loss = cls_sum / n_cls
    else:
        cls_loss = torch.tensor(0.0, device=device)

    sinkhorn = source_sinkhorn_loss(
        latent, source_ids, epsilon=sinkhorn_epsilon, n_iters=sinkhorn_iters
    )
    total_loss = (
        reconstruction_weight * recon_loss
        + classification_weight * cls_loss
        + sinkhorn_weight * sinkhorn
    )
    return {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "classification_loss": cls_loss,
        "head_3_loss": head_3_loss,
        "head_4_loss": head_4_loss,
        "sinkhorn_loss": sinkhorn,
        "weighted_reconstruction_loss": reconstruction_weight * recon_loss,
        "weighted_classification_loss": classification_weight * cls_loss,
        "weighted_sinkhorn_loss": sinkhorn_weight * sinkhorn,
        "n_samples": xb.shape[0],
        "n_head_3": n3,
        "n_head_4": n4,
        "n_classified": n_cls,
    }


def empty_epoch_totals():
    return {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "classification_loss": 0.0,
        "head_3_loss": 0.0,
        "head_4_loss": 0.0,
        "sinkhorn_loss": 0.0,
        "weighted_reconstruction_loss": 0.0,
        "weighted_classification_loss": 0.0,
        "weighted_sinkhorn_loss": 0.0,
        "n_samples": 0,
        "n_head_3": 0,
        "n_head_4": 0,
        "n_classified": 0,
    }


def update_epoch_totals(totals, components):
    n_samples = components["n_samples"]
    n_classified = components["n_classified"]
    n_head_3 = components["n_head_3"]
    n_head_4 = components["n_head_4"]
    sample_weighted = [
        "total_loss",
        "reconstruction_loss",
        "sinkhorn_loss",
        "weighted_reconstruction_loss",
        "weighted_sinkhorn_loss",
    ]
    for key in sample_weighted:
        totals[key] += float(components[key].detach().item()) * n_samples
    totals["classification_loss"] += float(components["classification_loss"].detach().item()) * n_classified
    totals["weighted_classification_loss"] += (
        float(components["weighted_classification_loss"].detach().item()) * n_classified
    )
    totals["head_3_loss"] += float(components["head_3_loss"].detach().item()) * n_head_3
    totals["head_4_loss"] += float(components["head_4_loss"].detach().item()) * n_head_4
    totals["n_samples"] += n_samples
    totals["n_classified"] += n_classified
    totals["n_head_3"] += n_head_3
    totals["n_head_4"] += n_head_4


def finalize_epoch_metrics(totals, prefix):
    n_samples = totals["n_samples"]
    n_classified = totals["n_classified"]
    n_head_3 = totals["n_head_3"]
    n_head_4 = totals["n_head_4"]
    metrics = {}
    for key in [
        "total_loss",
        "reconstruction_loss",
        "sinkhorn_loss",
        "weighted_reconstruction_loss",
        "weighted_sinkhorn_loss",
    ]:
        metrics[f"{prefix}_{key}"] = totals[key] / n_samples if n_samples else np.nan
    metrics[f"{prefix}_classification_loss"] = (
        totals["classification_loss"] / n_classified if n_classified else 0.0
    )
    metrics[f"{prefix}_weighted_classification_loss"] = (
        totals["weighted_classification_loss"] / n_classified if n_classified else 0.0
    )
    metrics[f"{prefix}_head_3_loss"] = totals["head_3_loss"] / n_head_3 if n_head_3 else 0.0
    metrics[f"{prefix}_head_4_loss"] = totals["head_4_loss"] / n_head_4 if n_head_4 else 0.0
    return metrics


def compute_epoch_metrics(
    model,
    loader,
    device,
    reconstruction_weight,
    classification_weight,
    sinkhorn_weight,
    sinkhorn_epsilon,
    sinkhorn_iters,
    prefix,
):
    model.eval()
    mse = nn.MSELoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    totals = empty_epoch_totals()
    with torch.no_grad():
        for xb, y3b, y4b, source_ids in loader:
            xb = xb.to(device)
            y3b = y3b.to(device)
            y4b = y4b.to(device)
            source_ids = source_ids.to(device)
            components = two_head_loss_components(
                model,
                xb,
                y3b,
                y4b,
                source_ids,
                device,
                mse,
                bce_none,
                reconstruction_weight,
                classification_weight,
                sinkhorn_weight,
                sinkhorn_epsilon,
                sinkhorn_iters,
            )
            update_epoch_totals(totals, components)
    return finalize_epoch_metrics(totals, prefix)


def train_supervised_autoencoder(
    X_train,
    y_head_3,
    y_head_4,
    source_train,
    latent_dim=16,
    hidden_layer_sizes=None,
    hidden_dim=64,
    reconstruction_weight=1.0,
    classification_weight=1.0,
    sinkhorn_weight=0.1,
    sinkhorn_epsilon=0.1,
    sinkhorn_iters=30,
    epochs=200,
    patience=20,
    batch_size=256,
    learning_rate=1e-3,
    random_state=42,
    device_name=None,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    hidden_layer_sizes = normalize_ae_hidden_layer_sizes(hidden_layer_sizes, hidden_dim)

    idx_train, idx_valid = train_test_split(
        np.arange(len(X_train)), test_size=0.1, random_state=random_state, shuffle=True
    )
    train_loader = make_tensor_loader(
        X_train[idx_train],
        y_head_3.iloc[idx_train].reset_index(drop=True),
        y_head_4.iloc[idx_train].reset_index(drop=True),
        source_train.iloc[idx_train].reset_index(drop=True),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = make_tensor_loader(
        X_train[idx_valid],
        y_head_3.iloc[idx_valid].reset_index(drop=True),
        y_head_4.iloc[idx_valid].reset_index(drop=True),
        source_train.iloc[idx_valid].reset_index(drop=True),
        batch_size=batch_size,
        shuffle=False,
    )

    if device_name is not None and device_name != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device_name))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SupervisedAutoencoder(
        input_dim=X_train.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    best_state = None
    best_valid_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = empty_epoch_totals()
        for xb, y3b, y4b, source_ids in train_loader:
            xb = xb.to(device)
            y3b = y3b.to(device)
            y4b = y4b.to(device)
            source_ids = source_ids.to(device)

            optimizer.zero_grad()
            components = two_head_loss_components(
                model,
                xb,
                y3b,
                y4b,
                source_ids,
                device,
                mse,
                bce_none,
                reconstruction_weight,
                classification_weight,
                sinkhorn_weight,
                sinkhorn_epsilon,
                sinkhorn_iters,
            )
            loss = components["total_loss"]
            loss.backward()
            optimizer.step()
            update_epoch_totals(train_totals, components)

        valid_metrics = compute_epoch_metrics(
            model,
            valid_loader,
            device,
            reconstruction_weight,
            classification_weight,
            sinkhorn_weight,
            sinkhorn_epsilon,
            sinkhorn_iters,
            prefix="valid",
        )
        train_metrics = finalize_epoch_metrics(train_totals, "train")
        epoch_metrics = {"epoch": epoch, **train_metrics, **valid_metrics}
        history.append(epoch_metrics)
        valid_loss = epoch_metrics["valid_total_loss"]
        train_loss = epoch_metrics["train_total_loss"]

        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: "
                f"train_total={train_loss:.5f}, valid_total={valid_loss:.5f}, "
                f"train_head_3={epoch_metrics['train_head_3_loss']:.5f}, "
                f"train_head_4={epoch_metrics['train_head_4_loss']:.5f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "latent_dim": latent_dim,
        "hidden_layer_sizes": hidden_layer_sizes,
        "reconstruction_weight": reconstruction_weight,
        "classification_weight": classification_weight,
        "sinkhorn_weight": sinkhorn_weight,
        "sinkhorn_epsilon": sinkhorn_epsilon,
        "sinkhorn_iters": sinkhorn_iters,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "device": str(device),
    }
    return model, history, meta


def encode_latent(model, X, latent_cols):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        latent = model.encoder(x_tensor).cpu().numpy()
    return pd.DataFrame(latent, columns=latent_cols)


def add_latent_columns(df, latent_df):
    return pd.concat([df.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)
