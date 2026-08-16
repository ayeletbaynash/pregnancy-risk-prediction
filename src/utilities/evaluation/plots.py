"""Final publication/portfolio plots for model evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

def plot_risk_by_percentile(test_df, out_path_png, out_path_csv):
    labeled = test_df[test_df["unified_risk"].notna()].copy()
    if labeled.empty:
        print("Skipping percentile risk plot - no labeled rows in test set")
        return

    labeled = labeled.sort_values("predicted_prob", ascending=True)
    y_true = labeled["unified_risk"].astype(float).to_numpy()
    cum_pos = np.cumsum(y_true)
    n = len(y_true)
    percentiles = np.arange(1, 101)
    k_vals = np.ceil(n * percentiles / 100.0).astype(int)
    k_vals = np.clip(k_vals, 1, n)
    risk_cumulative = cum_pos[k_vals - 1] / k_vals

    out_df = pd.DataFrame({"percentile": percentiles, "risk": risk_cumulative})
    out_df.to_csv(out_path_csv, index=False)

    y_max = min(1.0, max(0.1, float(np.max(risk_cumulative)) * 1.05))
    plt.figure(figsize=(8, 6))
    plt.plot(percentiles, risk_cumulative, color="slateblue", lw=2)
    plt.xlim([1, 100])
    plt.ylim([0.0, y_max])
    plt.xlabel("population percentile (bottom x% scores)")
    plt.ylabel("cumulative observed risk")
    plt.title("Cumulative Risk as a Function of Population Percentile")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=200)
    plt.close()
    print(f"Saved: {out_path_png}")


def plot_roc_pr(test_df, out_prefix):
    labeled = test_df[test_df["unified_risk"].notna()].copy()
    if labeled.empty or labeled["unified_risk"].nunique() < 2:
        print("Skipping ROC/PR plots - no labeled two-class test set")
        return

    y_true = labeled["unified_risk"].astype(int)
    y_prob = labeled["predicted_prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="teal", lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr_curve.png", dpi=200)
    plt.close()


def plot_ae_history(history_df, out_dir):
    if history_df.empty:
        return

    def plot_columns(columns, out_name, title, ylabel="loss"):
        available = [col for col in columns if col in history_df.columns]
        if not available:
            return
        plt.figure(figsize=(9, 6))
        for col in available:
            plt.plot(history_df["epoch"], history_df[col], lw=2, label=col)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        out_path = out_dir / out_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")

    plot_columns(
        ["train_total_loss", "valid_total_loss"],
        "ae_total_loss.png",
        "Supervised AE Total Loss",
    )
    plot_columns(
        ["valid_reconstruction_loss", "valid_classification_loss", "valid_sinkhorn_loss"],
        "ae_loss_components.png",
        "Supervised AE Validation Loss Components",
    )
    plot_columns(
        [
            "valid_weighted_reconstruction_loss",
            "valid_weighted_classification_loss",
            "valid_weighted_sinkhorn_loss",
        ],
        "ae_weighted_loss_components.png",
        "Supervised AE Weighted Validation Loss Components",
    )
    plot_columns(
        ["train_head_3_loss", "valid_head_3_loss", "train_head_4_loss", "valid_head_4_loss"],
        "ae_two_heads_loss.png",
        "Supervised AE Two Heads Loss",
    )
    plot_columns(
        ["train_classification_loss", "valid_classification_loss"],
        "ae_classification_loss.png",
        "Supervised AE Classification Loss",
    )

def plot_umap(df: pd.DataFrame, latent_cols: list[str], color_by: str, out_path: str | Path, random_state: int = 42):
    """Create a 2D UMAP of latent features, colored by source or risk."""
    try:
        import umap
    except ImportError:
        print(f"Skipping UMAP {out_path}: install umap-learn to enable UMAP plots.")
        return False
    work = df.dropna(subset=latent_cols).copy()
    if work.empty:
        return
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    xy = reducer.fit_transform(work[latent_cols].to_numpy())
    plt.figure(figsize=(8, 6))
    if color_by == "source":
        labels = work[color_by].astype(str)
        for label in sorted(labels.unique()):
            mask = labels == label
            plt.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.7, label=label)
        plt.legend(loc="best", fontsize=9)
    else:
        values = pd.to_numeric(work[color_by], errors="coerce")
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=values, s=14, alpha=0.7)
        plt.colorbar(sc, label=color_by)
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.title(f"Latent-space UMAP by {color_by}")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return True


def plot_experiment_comparison(prediction_files: dict[str, Path], out_path: str | Path):
    """Compare cumulative observed risk across leave-one-source-out experiments."""
    plt.figure(figsize=(10, 7))
    for name, path in prediction_files.items():
        df = pd.read_csv(path)
        labeled = df[df["unified_risk"].notna()].sort_values("predicted_prob")
        if labeled.empty: continue
        y = labeled["unified_risk"].astype(float).to_numpy(); n=len(y); p=np.arange(1,101)
        k=np.clip(np.ceil(n*p/100).astype(int),1,n); risk=np.cumsum(y)[k-1]/k
        plt.plot(p, risk, lw=2, label=name)
    plt.xlabel("Population percentile (bottom x% scores)"); plt.ylabel("Cumulative observed risk")
    plt.title("Leave-one-dataset-out comparison"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out_path,dpi=200); plt.close()
