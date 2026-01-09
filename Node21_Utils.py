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


def train_one_epoch_det(model, loader, optimizer, device): # <--- AFEGIR device AQUÍ
    model.train()
    total_loss = 0.0
    loss_sums = {}
    n_batches = 0

    for images, targets in loader:
        # MOURE AL DEVICE (Si no ho fas, el model a la CPU anirà lent,
        # però si algun dia uses GPU, fallaria sense això)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
def eval_one_epoch_det(model, loader, device): # <--- ARA TÉ 3 ARGUMENTS
    model.train()
    total_loss = 0.0
    loss_sums = {}
    n_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())

        total_loss += loss.detach().item()
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.detach().item()

        n_batches += 1

    avg_total = total_loss / max(n_batches, 1)
    avg_dict = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    return avg_total, avg_dict

def fit_det(model, train_loader, val_loader, optimizer, epochs, device):
    history = {"train_loss": [], "val_loss": [], "train_loss_components": [], "val_loss_components": [],}

    for t in tqdm(range(epochs), desc="Èpoques"):
        # ERROR CORREGIT: Ara passem 4 arguments a train i 3 a eval
        tr_loss, tr_comp = train_one_epoch_det(model, train_loader, optimizer, device)
        va_loss, va_comp = eval_one_epoch_det(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_loss_components"].append(tr_comp)
        history["val_loss_components"].append(va_comp)

        tr_comp_str = " | ".join([f"{k}:{v:.4f}" for k, v in tr_comp.items()])
        va_comp_str = " | ".join([f"{k}:{v:.4f}" for k, v in va_comp.items()])

        print(
            f"[Època {t + 1}] "
            f"Train loss {tr_loss:.4f} ({tr_comp_str})  ||  "
            f"Val loss {va_loss:.4f} ({va_comp_str})"
        )

    return history


def plot_combined_history(h1, h2, figsize=(15, 5)):
    import matplotlib.pyplot as plt

    # Fusionem les dades
    combined = {}
    for key in h1.keys():
        combined[key] = h1[key] + h2[key]

    epochs = range(1, len(combined['train_loss']) + 1)
    split_point = len(h1['train_loss'])  # On acaba la Fase 1

    plt.figure(figsize=figsize)

    # --- ESQUERRA: LOSS TOTAL ---
    plt.subplot(1, 2, 1)
    plt.title("Total Loss (Fase 1 + Fase 2)")
    plt.plot(epochs, combined["train_loss"], label="Train loss")
    plt.plot(epochs, combined["val_loss"], label="Val. loss")
    # Línia vertical per indicar el canvi de fase
    plt.axvline(x=split_point, color='gray', linestyle='--', label='Inici Fase 2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # --- DRETA: COMPONENTS (Detecció) ---
    plt.subplot(1, 2, 2)
    plt.title("Loss Components")
    train_comps = combined['train_loss_components']
    cls_loss = [c['classification'] for c in train_comps]
    reg_loss = [c['bbox_regression'] for c in train_comps]

    plt.plot(epochs, cls_loss, label="Class loss")
    plt.plot(epochs, reg_loss, label="Reg. loss")
    plt.axvline(x=split_point, color='gray', linestyle='--', label='Inici Fase 2')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

import matplotlib.patches as patches


def visualize_positive_predictions(model, dataset, device, num_images=3, threshold=0.2):
    model.eval()
    model.to(device)

    # Cercquem els índexs de les imatges que tenen nòduls (label == 1)
    # Això ho fem mirant el diccionari del dataset que ja tenies
    positive_indices = [i for i, img_id in enumerate(dataset.img_ids)
                        if int(dataset.labels_dict[img_id]) == 1]

    if len(positive_indices) == 0:
        print("No s'han trobat imatges amb nòduls al dataset proporcionat.")
        return

    # Seleccionem uns quants índexs positius aleatoris
    selected_indices = np.random.choice(positive_indices, min(num_images, len(positive_indices)), replace=False)

    fig, axs = plt.subplots(1, len(selected_indices), figsize=(18, 6))
    if len(selected_indices) == 1: axs = [axs]

    for i, idx in enumerate(selected_indices):
        image, target = dataset[idx]

        with torch.no_grad():
            # El model espera una llista de tensors [C, H, W]
            prediction = model([image.to(device)])[0]

        # Convertim imatge per visualitzar (de [3, H, W] a [H, W, 3])
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Normalitzem si cal per mostrar-la bé amb imshow
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        axs[i].imshow(img_np, cmap='gray')
        axs[i].set_title(f"Imatge ID: {dataset.img_ids[idx]}")
        axs[i].axis('off')

        # 1. Dibuixar caixes REALS (Verd) - Les que sabem que hi són
        for box in target['boxes']:
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='lime', facecolor='none',
                                     label='Real')
            axs[i].add_patch(rect)

        # 2. Dibuixar prediccions del MODEL (Vermell)
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Ordenem per score i agafem les top 5 per veure què sospita el model
        # fins i tot si estan per sota del threshold
        found_any = False
        for box, score in zip(boxes, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(x1, y1 - 10, f"{score:.2f}", color='red', fontsize=12, fontweight='bold',
                            backgroundcolor='none')
                found_any = True

        if not found_any:
            axs[i].text(10, 30, "Cap pred. > threshold", color='yellow', backgroundcolor='black')

    plt.legend(handles=[
        patches.Patch(color='g', label='Real (Nòdul)'),
        patches.Patch(color='r', label='Predicció')
    ], loc='upper right')
    plt.tight_layout()
    plt.show()