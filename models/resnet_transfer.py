import torch.nn as nn
import torchvision.models as models

class ResNet18TransferCNN(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 100)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
