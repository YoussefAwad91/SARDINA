import torch.nn as nn
from .config import NUM_CLASSES

# Input: 21 landmarks × 2 (x, y) = 42 features
LANDMARK_INPUT_SIZE = 42

class FingerNet(nn.Module):
    def __init__(self):
        super(FingerNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(LANDMARK_INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(x)
