"""Skribl's compact convolutional character classifier."""
import string
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
LABELS = {"digits": list(string.digits), "letters": list(string.ascii_uppercase + string.ascii_lowercase)}
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

class CharacterCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, 128),
                                        nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def load_model(mode):
    checkpoint = torch.load(MODELS / f"{mode}.pt", map_location="cpu", weights_only=True)
    if checkpoint["labels"] != LABELS[mode]:
        raise ValueError("Checkpoint labels do not match this application.")
    model = CharacterCNN(len(LABELS[mode]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint

def atomic_save(payload, destination):
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
