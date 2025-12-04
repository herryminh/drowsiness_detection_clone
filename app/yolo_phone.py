from ultralytics import YOLO
from .config import PHONE_DETECT
_model = None
def get_phone_model():
    global _model
    if _model is None:
        print("Loaded face model from:", PHONE_DETECT)
        _model = YOLO(PHONE_DETECT)
    return _model