import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4

            nn.Flatten(),

            nn.Linear(128*4*4, 256),
            nn.ReLU(),

            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.net(x)
