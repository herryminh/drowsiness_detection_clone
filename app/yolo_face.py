from ultralytics import YOLO
from .config import FACE_WEIGHTS
_model = None
def get_face_model():
    global _model
    if _model is None:
        print("Loaded face model from:", FACE_WEIGHTS)
        _model = YOLO(FACE_WEIGHTS)
    return _model