import torch
from src.model import PatientSeverityNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PatientSeverityNet().to(device)
model.load_state_dict(torch.load("models/best_severity_model.pth", map_location=device))
model.eval()

def predict_patient(patient_tensor):
    """
    patient_tensor: [1, S, 3, 224, 224]
    """
    with torch.no_grad():
        outputs = model(patient_tensor.to(device))
        pred = outputs.argmax(dim=1).item()
    return pred
