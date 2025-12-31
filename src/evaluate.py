import torch
from sklearn.metrics import f1_score

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.numpy())

    return f1_score(targets, preds, average="weighted")
