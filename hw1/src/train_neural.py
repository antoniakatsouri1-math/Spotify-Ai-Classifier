import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

class SpotifyNet(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_neural_network(
    X_train: np.ndarray, y_train,
    X_val:   np.ndarray, y_val,
    plots_dir: str = "plots",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    random_state: int = 42
) -> nn.Module:

    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Neural Net] Using device: {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1).to(device)

    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SpotifyNet(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    no_improve    = 0

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            val_auc = roc_auc_score(
                y_val.values, val_pred.cpu().numpy()
            )
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={epoch_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs = model(X_v).cpu().numpy()
    val_auc = roc_auc_score(y_val.values, val_probs)
    print(f"[Neural Net] Best epoch: {best_epoch} | Val ROC-AUC: {val_auc:.4f}")

    # ── Loss curve plot ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train Loss", color="#2196F3")
    ax.plot(val_losses,   label="Val Loss",   color="#F44336")
    ax.axvline(best_epoch - 1, color="gray", linestyle="--", linewidth=0.8,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Neural Network — Training & Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "nn_loss_curve.png"), dpi=150)
    plt.close()

    return model

def save_neural_network(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"Neural network saved → {path}")


def load_neural_network(path: str, input_dim: int) -> nn.Module:
    model = SpotifyNet(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def experiment_activation_functions(
    X_train: np.ndarray, y_train,
    X_val:   np.ndarray, y_val,
    plots_dir: str = "plots",
    random_state: int = 42
) -> dict:

    import copy

    activations = {
        "ReLU":       nn.ReLU(),
        "LeakyReLU":  nn.LeakyReLU(negative_slope=0.1),
        "ELU":        nn.ELU(alpha=1.0),
        "Tanh":       nn.Tanh(),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1).to(device)

    results = {}
    print("\n[Activation Function Experiment]")
    print(f"{'Activation':<12} {'Val AUC':>10} {'Best Epoch':>12}")
    print("-" * 36)

    for act_name, act_fn in activations.items():
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),               nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),                act_fn,
            nn.Linear(32, 1),                 nn.Sigmoid()
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCELoss()
        best_auc, best_ep, no_imp = 0.0, 0, 0
        patience = 8

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

        for epoch in range(1, 60):
            model.train()
            for xb, yb in train_dl:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                probs = model(X_v).cpu().numpy()
            auc = roc_auc_score(y_val.values, probs)
            if auc > best_auc + 1e-4:
                best_auc, best_ep, no_imp = auc, epoch, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        results[act_name] = {"val_auc": round(best_auc, 4), "best_epoch": best_ep}
        print(f"{act_name:<12} {best_auc:>10.4f} {best_ep:>12}")

    os.makedirs(plots_dir, exist_ok=True)
    names = list(results.keys())
    aucs  = [results[n]["val_auc"] for n in names]
    colors = ["#90CAF9", "#EF9A9A", "#A5D6A7", "#FFE082"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, aucs, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.01)
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Activation Function Comparison (3rd Hidden Layer)")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{auc:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "activation_comparison.png"), dpi=150)
    plt.close()
    print(f"  → Plot saved: {plots_dir}/activation_comparison.png\n")

    best_act = max(results, key=lambda k: results[k]["val_auc"])
    print(f"  → Best activation: {best_act} (AUC={results[best_act]['val_auc']})")
    return results


def predict_proba_nn(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32).to(device)
        probs = model(t).cpu().numpy().ravel()
    return probs
