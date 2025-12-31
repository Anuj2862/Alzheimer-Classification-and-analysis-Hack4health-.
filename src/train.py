import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

from src import config, utils
from src.dataset import AlzheimerBinaryDataset, AlzheimerSeverityDataset
from src.model import DenseNetModel

def get_binary_loaders(df):
    binary_labels = (df["label"] != 0).astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    train_idx, val_idx = next(sss.split(df, binary_labels))

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    train_ds = AlzheimerBinaryDataset(train_df)
    val_ds   = AlzheimerBinaryDataset(val_df)

    class_counts = np.bincount((train_df["label"] != 0).astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[(train_df["label"] != 0).astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def get_severity_loaders(df):
    # Filter for severity (labels 1, 2, 3)
    df_severity = df[df["label"].isin([1, 2, 3])].copy()
    df_severity["label"] = df_severity["label"].astype(int)
    
    severity_labels = (df_severity["label"] - 1).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    train_idx, val_idx = next(sss.split(df_severity, severity_labels))

    train_df = df_severity.iloc[train_idx]
    val_df   = df_severity.iloc[val_idx]

    train_ds = AlzheimerSeverityDataset(train_df)
    val_ds   = AlzheimerSeverityDataset(val_df)

    # Weights
    train_labels = (train_df["label"] - 1).values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    preds_all, labels_all = [], []
    
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return acc

def validate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return acc

def main(mode="binary"):
    utils.seed_everything(config.SEED)
    
    if not os.path.exists(config.TRAIN_PARQUET):
        print(f"Error: Data file not found at {config.TRAIN_PARQUET}")
        return

    print("Loading data...")
    df = pd.read_parquet(config.TRAIN_PARQUET)
    
    if mode == "binary":
        print("Preparing Binary Classification...")
        train_loader, val_loader = get_binary_loaders(df)
        num_classes = 2
        save_name = "best_binary_model.pth"
    else:
        print("Preparing Severity Classification...")
        train_loader, val_loader = get_severity_loaders(df)
        num_classes = 3
        save_name = "best_severity_model.pth"

    print(f"Initializing Model for {num_classes} classes...")
    model = DenseNetModel(num_classes=num_classes).to(config.DEVICE)
    
    criterion = utils.FocalLoss(gamma=1.0 if mode=="binary" else 1.5)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_acc = 0.0

    print("Starting Training...")
    for epoch in range(config.EPOCHS):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_acc = validate(model, val_loader, config.DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"Saved Best Model ({val_acc:.4f})")

if __name__ == "__main__":
    # Change mode to "severity" for multi-class training
    main(mode="binary")
