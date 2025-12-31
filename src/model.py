import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.densenet121(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class PatientSeverityNet(nn.Module):
    """
    Patient-level ResNet50 model (Legacy/Alternative)
    """
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: [B, S, 3, 224, 224]
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        features = self.backbone(x)       # [B*S, 2048]
        features = features.view(B, S, -1)
        features = features.mean(dim=1)   # patient aggregation

        return self.classifier(features)
