from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_DIR = ROOT / "dataset"
YAWN_CROPS_DIR  = ROOT / "data_crops" / "mouth"

EYE_DATASET_DIR  = ROOT / "dataset_eye"
YAWN_DATASET_DIR = YAWN_CROPS_DIR

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EYE_CKPT = CHECKPOINT_DIR / "eye_best.pt"
YAWN_CKPT = CHECKPOINT_DIR / "yawn_best.pt"

FACE_WEIGHTS = str(ROOT / "models" / "face_yolov8n.pt")
PHONE_DETECT = str((ROOT / "models" / "phone_yolov8n.pt"))

EYE_CLOSED_SEC = 2.5
YAWN_SEC = 2.5
WINDOW_ACCUM_SEC = 20.0
YAWN_ACCUM_TOTAL_SEC = 10.0
HEAD_AWAY_SEC = 2.0

PHONE_HOLD_SEC = 5.0   

EYE_IMG_SIZE = 128
MOUTH_IMG_SIZE = 128

ALARM_FILES = {
    "sleepy": ROOT / "buon_ngu.mp3",
    "phone": ROOT / "su_dung_dt.mp3",
    "not_focus": ROOT / "khong_tap_trung.mp3",
}
