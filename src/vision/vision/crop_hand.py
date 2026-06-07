import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from pathlib import Path

model_asset_path=str(Path(__file__).parent /"hand_landmarker.task")

# Automatically download model
base_options = python.BaseOptions(model_asset_path=model_asset_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


def get_hand_landmarks(image):
    """
    Runs MediaPipe on a BGR image and returns a normalised (42,) numpy array
    of [x0, y0, x1, y1, ..., x20, y20] landmark coordinates, or None if no
    hand is detected.

    Coordinates are normalised relative to the hand bounding box so the
    output is invariant to hand position and scale in the frame.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_result = hand_landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    )
    if not detection_result.hand_landmarks:
        return None

    landmarks = detection_result.hand_landmarks[0]  # first hand

    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])

    # Normalise to [0, 1] within the hand bounding box so position/scale
    # in the frame doesn't affect the features
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    xs = (xs - x_min) / x_range
    ys = (ys - y_min) / y_range

    # Interleave x, y → [x0, y0, x1, y1, ...]
    coords = np.empty(42, dtype=np.float32)
    coords[0::2] = xs
    coords[1::2] = ys

    return coords


# ── kept for backward compatibility (no longer used in main pipeline) ────────

def get_hand_crop(image, padding=10):
    h, w, _ = image.shape

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_result = hand_landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    )
    if not detection_result.hand_landmarks:
        return None

    landmarks = detection_result.hand_landmarks[0]

    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    box_w = x_max - x_min
    box_h = y_max - y_min
    size = max(box_w, box_h)
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    size = size + size // padding

    new_x_min = int(max(0, center_x - size // 2))
    new_y_min = int(max(0, center_y - size // 2))
    new_x_max = int(min(w, center_x + size // 2))
    new_y_max = int(min(h, center_y + size // 2))

    return image[new_y_min:new_y_max, new_x_min:new_x_max]
