import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------------
# CLASSIFICACIÓ
# -----------------------------------------

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
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_optimizer(model, lr=1e-3, weight_decay=1e-4):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_optimizer_denseNet(model, lr=1e-5, weight_decay=1e-2):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def _batch_accuracy_from_logits(logits, y_true):
    """
    logits: [B,1], y_true: [B,1] float or int
    """
    y_prob = torch.sigmoid(logits)
    y_pred = (y_prob >= 0.5).long().view(-1).cpu().numpy()
    y_true_np = y_true.long().view(-1).cpu().numpy()

    #print(f"Pred: {y_pred.tolist()}", flush=True)
    #print(f"True: {y_true_np.tolist()}", flush=True)
    #print("-" * 30)  # Separador visual opcional
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


def plot_confusion_matrix(model, loader, device):
    """
    Genera una matriu de confusió i un informe de classificació.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            # Apliquem sigmoid perquè el model treu probabilitats per a classificació binària
            # Si el valor és > 0.5, prediu nòdul (1), altrament sa (0)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculem la matriu de confusió
    cm = confusion_matrix(all_labels, all_preds)

    # Dibuixem la matriu amb Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sa (0)', 'Nòdul (1)'],
                yticklabels=['Sa (0)', 'Nòdul (1)'])
    plt.xlabel('Predicció del Model')
    plt.ylabel('Realitat (Ground Truth)')
    plt.title('Matriu de Confusió - Conjunt de Validació')
    plt.show()

    # Imprimim les mètriques detallades (Precision, Recall, F1-Score)
    print("\n--- Informe de Classificació ---")
    print(classification_report(all_labels, all_preds, target_names=['Sa', 'Nòdul']))

# -----------------------------------------
# DETECCIÓ
# -----------------------------------------
def make_optimizer_detection(model, lr=1e-3, weight_decay=5e-4, momentum=0.9):
    # Filtrem paràmetres que no s'entrenen
    params = [p for p in model.parameters() if p.requires_grad]

    return optim.SGD(params,lr=lr,momentum=momentum,weight_decay=weight_decay)


def train_one_epoch_det(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i_batch, (images, targets) in enumerate(loader):
        # Adaptació per a llistes de tensors (necessari per detecció)
        images = list(image.to(device, non_blocking=True) for image in images)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # El model retorna un diccionari amb les pèrdues calculades internament
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        n_batches += 1

    return total_loss / n_batches

def validate_one_epoch_det(model, loader, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = list(image.to(device, non_blocking=True) for image in images)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())

        total_loss += loss.detach().item()
        n_batches += 1

    return total_loss / n_batches


def fit_det(model, train_loader, val_loader, optimizer, epochs, device):
    history = {"train_loss": [], "val_loss": []}

    # Només mantenim la barra de les èpoques com en el teu exemple
    for t in tqdm(range(epochs), desc="Èpoques"):
        tr_loss = train_one_epoch_det(model, train_loader, optimizer, device)
        va_loss = validate_one_epoch_det(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        print(
            f"[Època {t + 1}] Train loss {tr_loss:.4f} | "
            f"Val loss {va_loss:.4f}"
        )

    return history

def plot_detection_history(history, figsize=(10, 5)):
    """
    Dibuixa les corbes de loss per a la Tasca 2 (Detecció).

    history ha de contenir:
        - train_loss
        - val_loss
    """
    import matplotlib.pyplot as plt

    # Validem que tenim les claus de loss
    if "train_loss" not in history or "val_loss" not in history:
        raise ValueError("L'historial ha de contenir 'train_loss' i 'val_loss'")

    plt.figure(figsize=figsize)

    # Dibuixem la Loss
    plt.title("Evolució de la Loss (RetinaNet)")
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val. Loss", marker='o')

    plt.xlabel("Època")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
