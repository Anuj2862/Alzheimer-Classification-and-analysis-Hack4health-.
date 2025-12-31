import torch
from torch.utils.data import DataLoader

from src.model import PatientSeverityNet
from src.dataset import AlzheimerPatientDataset
from src.train import train_one_epoch
from src.evaluate import evaluate

# NOTE:
# You must load preprocessed numpy arrays here:
# train_images, train_labels
# val_images, val_labels

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = AlzheimerPatientDataset(train_images, train_labels)
val_ds   = AlzheimerPatientDataset(val_images, val_labels)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

model = PatientSeverityNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

best_f1 = 0.0

for epoch in range(20):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_f1 = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | Val F1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "models/best_severity_model.pth")
        print("✅ Best model saved")

print("🏁 Training complete")
