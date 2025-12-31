import io
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import config

def preprocess_mri(raw):
    """
    Preprocesses raw image bytes:
    1. Converts to Grayscale
    2. Crops to remove black background
    3. Resizes to (160, 160)
    4. Applies CLAHE
    5. Normalizes to [0, 1]
    6. Stacks to 3 channels
    """
    if isinstance(raw, dict):
        raw = raw["bytes"]

    img = Image.open(io.BytesIO(raw)).convert("L")
    img = np.array(img)

    coords = np.where(img > 10)
    if len(coords[0]) > 0:
        y0, y1 = coords[0].min(), coords[0].max()
        x0, x1 = coords[1].min(), coords[1].max()
        img = img[y0:y1, x0:x1]

    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

    clahe = cv2.createCLAHE(2.0, (8,8))
    img = clahe.apply(img)

    img = img / 255.0
    img = np.stack([img]*3, axis=0)

    return torch.tensor(img, dtype=torch.float32)

class AlzheimerBinaryDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = preprocess_mri(self.df.loc[idx, "image"])

        # Binary labels: 0=Healthy, 1=Demented
        original_label = int(self.df.loc[idx, "label"])
        binary_label = 0 if original_label == 0 else 1

        return img, binary_label

class AlzheimerSeverityDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use iloc for severity dataset as seen in notebook
        row = self.df.iloc[idx]
        img = preprocess_mri(row["image"])
        
        # Labels 1,2,3 -> 0,1,2
        label = int(row["label"]) - 1 

        return img, label
