"""
train_neural.py
---------------
PyTorch feedforward neural network for multi-class classification.
Architecture:  Input → Dense(128, ReLU) → Dropout(0.3)
                     → Dense(64,  ReLU) → Dropout(0.2)
                     → Dense(32,  ELU)
                     → Dense(n_classes, Softmax)

Training:
  - Adam optimiser
  - CrossEntropyLoss
  - Early stopping with patience=15 (restores best weights)
  - Plots train/val loss curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════
# Model definition
# ══════════════════════════════════════════════════════════════════════════

class StudentBehaviorNet(nn.Module):
    """
    Feedforward network with:
      - 3 hidden layers (≥ 2 as required)
      - ReLU on first two hidden layers
      - ELU on third hidden layer (alternative activation, tested per assignment)
      - Dropout(0.3) and Dropout(0.2) for regularisation
      - Linear output → CrossEntropyLoss handles softmax internally
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ELU(),           # alternative activation (ELU)

            nn.Linear(32, n_classes),
            # No softmax here — nn.CrossEntropyLoss applies log-softmax internally
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════

def make_loaders(X_train, y_train, X_val, y_val, batch_size: int = 512):
    def to_tensor(X, y):
        X_t = torch.tensor(X.values if hasattr(X, "values") else X,
                           dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def train_neural(X_train, y_train, X_val, y_val,
                 n_classes: int,
                 models_dir: str = "models",
                 max_epochs: int = 150,
                 patience: int = 15,
                 batch_size: int = 512,
                 lr: float = 1e-3):
    """
    Full training loop with early stopping.
    Saves best weights to models/neural_network.pt.
    Plots loss curves to models/nn_loss_curve.png.
    Returns trained model.
    """
    os.makedirs(models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[neural]  Training on {device}")

    input_dim = X_train.shape[1]
    model = StudentBehaviorNet(input_dim, n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=7, factor=0.5, verbose=False
    )

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size)

    best_val_loss  = float("inf")
    best_weights   = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        tr_loss  = train_epoch(model, train_loader, criterion, optimiser, device)
        val_loss = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[neural]  Early stopping at epoch {epoch}  "
                  f"(best val_loss={best_val_loss:.4f} at epoch {epoch - patience})")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"[neural]  Best val_loss: {best_val_loss:.4f}")

    # Save model
    pt_path = os.path.join(models_dir, "neural_network.pt")
    torch.save({"model_state_dict": best_weights,
                "input_dim": input_dim,
                "n_classes": n_classes}, pt_path)
    print(f"[neural]  Saved → {pt_path}")

    # Plot loss curves
    _plot_loss_curves(train_losses, val_losses, models_dir)

    return model, device


def _plot_loss_curves(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train loss", color="#4C72B0")
    ax.plot(val_losses,   label="Val loss",   color="#DD8452")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.set_title("Neural Network — Train / Validation Loss Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "nn_loss_curve.png"), dpi=150)
    plt.close(fig)
    print("[neural]  Loss curve saved → models/nn_loss_curve.png")


# ══════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════

def predict_proba_nn(model, X, device, batch_size: int = 512):
    """Return softmax probabilities for a numpy / DataFrame input."""
    model.eval()
    X_t = torch.tensor(
        X.values if hasattr(X, "values") else X, dtype=torch.float32
    )
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def load_neural_network(models_dir: str = "models"):
    """Load a saved neural network from models/neural_network.pt."""
    pt_path = os.path.join(models_dir, "neural_network.pt")
    ckpt = torch.load(pt_path, map_location="cpu")
    model = StudentBehaviorNet(ckpt["input_dim"], ckpt["n_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
