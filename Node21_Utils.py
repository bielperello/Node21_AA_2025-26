import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def compute_pos_weight(train_ids, labels_dict, device=None):
    """
    pos_weight = #neg / #pos per BCEWithLogitsLoss.
    """
    neg = sum(labels_dict[i] == 0 for i in train_ids)
    pos = sum(labels_dict[i] == 1 for i in train_ids)
    if pos == 0:
        raise ValueError("No hi ha cap mostra positiva al train; pos_weight no és definible.")
    w = torch.tensor([neg / pos], dtype=torch.float32)
    if device is not None:
        w = w.to(device)
    return w


def make_loss(pos_weight):
    import torch.nn as nn
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_optimizer(model, lr=1e-3, weight_decay=1e-4):
    import torch.optim as optim
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def _batch_accuracy_from_logits(logits, y_true):
    """
    logits: [B,1], y_true: [B,1] float or int
    """
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob >= 0.5).long().view(-1).cpu().numpy()
    y_true_np = y_true.long().view(-1).cpu().numpy()
    return accuracy_score(y_true_np, y_pred)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        total_acc += _batch_accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.detach().item()
        total_acc += _batch_accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }

    for t in tqdm(range(epochs), desc="Èpoques"):

        tr_loss, tr_acc = train_one_epoch( model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"[Època {t+1}] Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"Val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

    return history


def plot_training_history(history, figsize=(15, 5)):
    """
    Dibuixa les corbes de loss i accuracy a partir del diccionari `history`
    retornat per la funció fit().

    history ha de contenir:
      - train_loss
      - val_loss
      - train_acc
      - val_acc
    """
    import matplotlib.pyplot as plt

    required_keys = {"train_loss", "val_loss", "train_acc", "val_acc"}
    missing = required_keys - history.keys()
    if missing:
        raise ValueError(f"Falten claus al history: {missing}")

    plt.figure(figsize=figsize)

    # LOSS
    plt.subplot(1, 2, 1)
    plt.title("Loss function")
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val. loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ACCURACY
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(history["train_acc"], label="Train accuracy")
    plt.plot(history["val_acc"], label="Val. accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_one_epoch_det(model, loader, optimizer):
    """
    Entrenament per models torchvision detection (SSD/FasterRCNN).
    Retorna: avg_total_loss, avg_loss_dict
    """
    model.train()
    total_loss = 0.0
    loss_sums = {}
    n_batches = 0

    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        loss.backward()

        optimizer.step()

        total_loss += loss.detach().item()
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.detach().item()

        n_batches += 1

    avg_total = total_loss / max(n_batches, 1)
    avg_dict = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    return avg_total, avg_dict


@torch.no_grad()
def eval_one_epoch_det(model, loader):
    """
        AVALUACIÓ de loss en validació. Per obtenir loss_dict, els models torchvision detection han d'estar en train().
        Retorna: avg_total_loss, avg_loss_dict
    """
    model.train()
    total_loss = 0.0
    loss_sums = {}
    n_batches = 0

    for images, targets in loader:
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        total_loss += loss.detach().item()
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.detach().item()

        n_batches += 1

    avg_total = total_loss / max(n_batches, 1)
    avg_dict = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    return avg_total, avg_dict

def fit_det(model, train_loader, val_loader, optimizer, epochs, device):
    """
    Fit simple per detecció:
      - train loss total + components
      - val loss total + components
    """
    history = {"train_loss": [], "val_loss": [], "train_loss_components": [], "val_loss_components": [],}

    for t in tqdm(range(epochs), desc="Èpoques"):
        tr_loss, tr_comp = train_one_epoch_det(model, train_loader, optimizer)
        va_loss, va_comp = eval_one_epoch_det(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_loss_components"].append(tr_comp)
        history["val_loss_components"].append(va_comp)

        # Print clar per veure què domina
        tr_comp_str = " | ".join([f"{k}:{v:.4f}" for k, v in tr_comp.items()])
        va_comp_str = " | ".join([f"{k}:{v:.4f}" for k, v in va_comp.items()])

        print(
            f"[Època {t + 1}] "
            f"Train loss {tr_loss:.4f} ({tr_comp_str})  ||  "
            f"Val loss {va_loss:.4f} ({va_comp_str})"
        )

    return history


