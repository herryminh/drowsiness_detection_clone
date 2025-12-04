# core.py (refactored)
import time
import collections

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

# Local imports (config / models / audio / landmarks)
from .yolo_phone import get_phone_model
from .yolo_face import get_face_model
from .config import (
    EYE_CKPT, YAWN_CKPT,
    EYE_IMG_SIZE, MOUTH_IMG_SIZE,
    EYE_CLOSED_SEC, YAWN_SEC,
    YAWN_ACCUM_TOTAL_SEC, WINDOW_ACCUM_SEC,
    HEAD_AWAY_SEC, PHONE_HOLD_SEC
)
from .audio import AlarmPlayer
from .landmarks import (
    FaceLandmarker, MOUTH_OUTER, MOUTH_INNER,
    LEFT_EYE, RIGHT_EYE, bbox_from_points,
    get_head_pose
)

# Device (GPU của macbook nếu có MPS)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def _load_model(ckpt_path, out_classes=2):
    """
    Load một ResNet18 đã được fine-tune.
    Trả về model đã set device và eval().
    """
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, out_classes)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    return model.to(DEVICE).eval()


class DrowsinessSystem:
    """
    Hệ thống chính:
    - detect phone + face bằng YOLO
    - detect landmarks + headpose
    - predict eye (open/closed) và yawn bằng 2 CNN
    - lưu history và quyết định alarm (sleepy/not_focus/phone)
    """

    def __init__(self):
        # --- models ---
        self.phone_model = get_phone_model()
        self.face_model = get_face_model()
        self.lmk = FaceLandmarker(static=False, max_faces=1)
        self.eye_model = _load_model(EYE_CKPT)
        self.yawn_model = _load_model(YAWN_CKPT)
        
        # --- preprocessing transforms ---
        self.tf_eye = transforms.Compose([
            transforms.Resize((EYE_IMG_SIZE, EYE_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.tf_mouth = transforms.Compose([
            transforms.Resize((MOUTH_IMG_SIZE, MOUTH_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # --- state & thresholds (lấy từ config) ---
        self.history = collections.deque(maxlen=600)
        self.last_alarm_ts = 0.0
        self.phone_start_ts = None

        self.eye_closed_sec = EYE_CLOSED_SEC
        self.yawn_sec = YAWN_SEC
        self.window_accum_sec = WINDOW_ACCUM_SEC
        self.yawn_accum_total_sec = YAWN_ACCUM_TOTAL_SEC
        self.head_away_sec = HEAD_AWAY_SEC
        self.phone_away_sec = PHONE_HOLD_SEC

        # Alarm player (không chồng âm, phát đến hết)
        self.alarm_player = AlarmPlayer(volume=1.0)

    # -------------------------
    # ROI helpers
    # -------------------------
    def _heuristic_eye_mouth(self, img, xyxy):
        """Fallback ROI khi landmarks fail: crop tỉ lệ trong face bbox."""
        x1, y1, x2, y2 = map(int, xyxy)
        face = img[y1:y2, x1:x2]
        h, w = face.shape[:2]
        if h <= 0 or w <= 0:
            return None, None
        eye = face[0:int(0.45 * h), :]
        mouth = face[int(0.55 * h):, int(0.2 * w):int(0.8 * w)]
        return eye, mouth

    def _landmark_eye_mouth(self, img):
        """
        Lấy ROI bằng landmarks. Trả về:
        (eyes_box), (mouth_box), eye_patch, mouth_patch
        """
        pts = self.lmk.detect(img)
        if pts is None:
            return None, None, None, None

        # Mouth bbox (mở rộng bằng scale)
        mouth_pts = np.vstack([pts[MOUTH_OUTER], pts[MOUTH_INNER]])
        mx1, my1, mx2, my2 = bbox_from_points(mouth_pts, scale=1.3, img_shape=img.shape)

        # Eyes (gộp 2 mắt)
        eyes_pts = np.vstack([pts[LEFT_EYE], pts[RIGHT_EYE]])
        ex1, ey1, ex2, ey2 = bbox_from_points(eyes_pts, scale=1.4, img_shape=img.shape)

        eye_patch = img[ey1:ey2, ex1:ex2].copy() if ey2 > ey1 and ex2 > ex1 else None
        mouth_patch = img[my1:my2, mx1:mx2].copy() if my2 > my1 and mx2 > mx1 else None
        return (ex1, ey1, ex2, ey2), (mx1, my1, mx2, my2), eye_patch, mouth_patch

    # -------------------------
    # Model predict helper
    # -------------------------
    def _pred(self, model, patch, tf):
        """Predict single patch; trả về (pred, prob_array, confidence) hoặc None."""
        if patch is None or patch.size == 0:
            return None
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = transforms.ToPILImage()(patch)
        t = tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(t)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(prob.argmax())
        conf = float(prob.max())
        return pred, prob, conf

    # -------------------------
    # Main step (frame processing)
    # -------------------------
    def step(self, frame):
        """
        Xử lý 1 frame:
        - phát hiện phone/face
        - lấy ROI eyes/mouth (landmark hoặc heuristic)
        - dự đoán eye/yawn
        - cập nhật history
        - trả về dữ liệu cần vẽ cho GUI
        """
        # --- YOLO PHONE ---
        p = self.phone_model.predict(source=frame, verbose=False, conf=0.45)[0]
        phone_detected = len(p.boxes) > 0
        phone_xyxy = None
        if phone_detected:
            areas = (p.boxes.xyxy[:, 2] - p.boxes.xyxy[:, 0]) * \
                    (p.boxes.xyxy[:, 3] - p.boxes.xyxy[:, 1])
            i = int(areas.argmax())  # chọn box lớn nhất
            phone_xyxy = p.boxes.xyxy[i].cpu().numpy()

        # --- YOLO FACE ---
        r = self.face_model.predict(source=frame, verbose=False, conf=0.2)[0]
        face_xyxy = None
        if len(r.boxes):
            areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * \
                    (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
            i = int(areas.argmax())
            face_xyxy = r.boxes.xyxy[i].cpu().numpy()

        # --- LANDMARK ROI ---
        eyes_box, mouth_box, eye_roi, mouth_roi = self._landmark_eye_mouth(frame)

        # --- HEAD POSE ---
        yaw, pitch, roll = None, None, None
        pts = self.lmk.detect(frame)
        if pts is not None:
            pose_result = get_head_pose(pts, frame.shape)
            if pose_result:
                yaw, pitch, roll = pose_result

        # --- FACE THUMBNAIL ---
        face_roi = None
        if face_xyxy is not None:
            fx1, fy1, fx2, fy2 = map(int, face_xyxy)
            if fy2 > fy1 and fx2 > fx1:
                face_roi = frame[fy1:fy2, fx1:fx2].copy()

        # --- FALLBACK ROI nếu landmark thất bại ---
        if (eye_roi is None or mouth_roi is None) and face_xyxy is not None:
            e2, m2 = self._heuristic_eye_mouth(frame, face_xyxy)
            eye_roi = eye_roi if eye_roi is not None else e2
            mouth_roi = mouth_roi if mouth_roi is not None else m2

        # --- CNN predictions ---
        eyes_closed = False
        yawn_open = False
        eye_conf = None
        yawn_conf = None

        ep = self._pred(self.eye_model, eye_roi, self.tf_eye) if eye_roi is not None else None
        mp = self._pred(self.yawn_model, mouth_roi, self.tf_mouth) if mouth_roi is not None else None

        if ep is not None:
            eyes_closed = (ep[0] == 0)
            eye_conf = ep[2]
        if mp is not None:
            yawn_open = (mp[0] == 1)
            yawn_conf = mp[2]

        # --- UPDATE HISTORY ---
        current_yaw = yaw if yaw is not None else 0.0
        now = time.time()
        # Lưu tuple: (eyes_closed, yawn_open, yaw, phone_detected, timestamp)
        self.history.append((eyes_closed, yawn_open, current_yaw, phone_detected, now))

        # --- OUTPUT FOR GUI ---
        draw_box = eyes_box if eyes_box is not None else face_xyxy

        return (
            draw_box, eyes_closed, yawn_open,
            face_roi, eye_roi, mouth_roi,
            eyes_box, mouth_box, face_xyxy,
            eye_conf, yawn_conf,
            (yaw, pitch, roll),
            pts,
            phone_detected,
            phone_xyxy,
        )

    # -------------------------
    # History analysis helpers
    # -------------------------
    def _get_reversed_history(self):
        """Trả lịch sử theo thứ tự mới nhất -> cũ nhất (list)."""
        return list(self.history)[::-1]  # mới nhất đầu tiên

    def _calc_continuous(self, hist, idx):
        """
        Tính thời gian liên tục cho flag nằm ở vị trí idx trong record.
        hist: list các bản ghi theo thứ tự mới nhất -> cũ nhất
        idx: index của flag (0:eyes_closed, 1:yawn_open, 2:yaw, 3:phone, 4:ts)
        """
        if not hist:
            return 0.0
        total = 0.0
        last_ts = hist[0][4]
        for rec in hist:
            flag = rec[idx]
            ts = rec[4]
            dt = max(last_ts - ts, 0.0)
            if not flag:
                break
            total += dt
            last_ts = ts
        return total

    def _calc_head_away(self, hist, yaw_threshold_deg):
        """Tính thời gian liên tục head-away dựa trên yaw angle."""
        if not hist:
            return 0.0
        total = 0.0
        last_ts = hist[0][4]
        for closed, yawn, yaw_angle, phone, ts in hist:
            dt = max(last_ts - ts, 0.0)
            if abs(yaw_angle) > yaw_threshold_deg:
                total += dt
            else:
                break
            last_ts = ts
        return total

    def _calc_yawn_accum(self, window_start_ts):
        """
        Tính tổng thời gian trong cửa sổ mà trạng thái 'yawn' = True (tích luỹ).
        Sử dụng self.history (cũ -> mới).
        """
        yawn_accum = 0.0
        # Lọc các record nằm trong cửa sổ thời gian
        window_records = [(rec[1], rec[4]) for rec in self.history if rec[4] >= window_start_ts]
        if not window_records:
            return 0.0
        prev_ts = window_records[0][1]
        for yawn_flag, ts in window_records:
            if yawn_flag:
                yawn_accum += ts - prev_ts
            prev_ts = ts
        return yawn_accum
    

    # -------------------------
    # Decision + Alarm
    # -------------------------
    def should_alarm(self):
        """
        Phân tích history, quyết định loại alarm:
        trả (alarm_bool, info_dict)
        info_dict chứa durations và alarm_type.
        """
        now = time.time()
        if len(self.history) < 2:
            return False, {}

        # Chuẩn bị
        yaw_threshold_deg = 25.0
        hist_rev = self._get_reversed_history()  # mới nhất -> cũ nhất

        # 1) thời gian liên tục: eyes closed và yawn
        eyes_dur = self._calc_continuous(hist_rev, idx=0)
        yawn_dur = self._calc_continuous(hist_rev, idx=1)

        # 2) phone duration (liên tục): track bằng phone_start_ts để tránh phụ thuộc sampling
        phone_flag = self.history[-1][3]
        if phone_flag:
            if self.phone_start_ts is None:
                self.phone_start_ts = now
            phone_dur = now - self.phone_start_ts
        else:
            self.phone_start_ts = None
            phone_dur = 0.0

        # 3) head-away duration (liên tục)
        head_away_dur = self._calc_head_away(hist_rev, yaw_threshold_deg)

        # 4) yawn_accum trong cửa sổ
        window_start = now - self.window_accum_sec
        yawn_accum = self._calc_yawn_accum(window_start)

        # 5) quyết định loại alarm (ưu tiên sleepy > not_focus > phone)
        alarm_type = None
        if (eyes_dur >= self.eye_closed_sec) or (yawn_dur >= self.yawn_sec) or (yawn_accum >= self.yawn_accum_total_sec):
            alarm_type = "sleepy"
        elif head_away_dur >= self.head_away_sec:
            alarm_type = "not_focus"
        elif phone_dur >= self.phone_away_sec:
            alarm_type = "phone"

        # 6) phát âm thanh (cooldown tránh spam)
        if alarm_type is not None:
            # throttle phát để khỏi spam (ví dụ 1s)
            if (now - self.last_alarm_ts) > 1.0:
                try:
                    self.alarm_player.play(alarm_type)
                except Exception:
                    # giữ an toàn: không làm crash GUI nếu audio lỗi
                    pass
                self.last_alarm_ts = now

        # 7) trả kết quả (info cho GUI hiển thị debug)
        info = {
            "eyes_dur": eyes_dur,
            "yawn_dur": yawn_dur,
            "yawn_accum": yawn_accum,
            "head_away_dur": head_away_dur,
            "phone_dur": phone_dur,
            "alarm_type": alarm_type
        }
        return alarm_type is not None, info
