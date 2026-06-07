import os
import glob
import torch
import torch.nn.functional as F
from .model import FingerNet
from .config import *
from .crop_hand import get_hand_landmarks
from pathlib import Path

def get_latest_model(model_path):
    model_dir=str(Path(__file__).parent) + '/' +os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]  # e.g. finger_model
    pattern = os.path.join(model_dir, f"{base}_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No model files found matching {pattern}")
    return max(candidates)  # lexicographic max works because timestamp is YYYYMMDD_HHMMSS


class FingerPredictor:

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = FingerNet()
        latest = get_latest_model(MODEL_PATH)
        print(f"Loading model: {latest}")
        self.model.load_state_dict(torch.load(latest, map_location=self.device))
        self.model.eval()

    def preprocess(self, frame):
        landmarks = get_hand_landmarks(frame)
        if landmarks is None:
            return None

        tensor = torch.tensor(landmarks, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # (1, 42)
        return tensor

    def predict(self, frame):
        tensor = self.preprocess(frame)
        if tensor is None:
            return -1, 0.0

        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        count = int(pred.item())
        confidence = float(confidence.item())

        if confidence < CONFIDENCE_THRESHOLD:
            return -1, confidence

        return count, confidence
