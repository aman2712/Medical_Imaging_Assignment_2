import copy
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def dice_bce_loss(logits, targets, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return bce + (1 - dice)


@torch.no_grad()
def evaluate_patches(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp = fp = fn = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += dice_bce_loss(logits, masks).item()
        n_batches += 1
        preds = (torch.sigmoid(logits) > 0.5).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    return total_loss / n_batches, dice


def train_model(model, train_dataset, val_dataset, device, epochs, batch_size, lr, checkpoint_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    history, best_dice, best_state = [], -1.0, None

    for epoch in range(epochs):
        start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            loss = dice_bce_loss(model(images), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss, val_dice = evaluate_patches(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice})
        if val_dice > best_dice:
            best_dice, best_state = val_dice, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch + 1:2d}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val Dice {val_dice:.4f} | {time.time() - start:.0f}s")

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    if checkpoint_path is not None:
        torch.save({"model_state": best_state, "history": history, "best_val_dice": best_dice}, checkpoint_path)
    return model, history
