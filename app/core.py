# core.py (fixed, calibrated pitch: straight=0, down=positive)
import time
import collections
import math

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

    Mỗi record trong self.history có 6 phần tử:
      (0) eyes_closed (bool)
      (1) yawn_open  (bool)
      (2) yaw_angle  (float, deg, continuous)
      (3) pitch_angle(float, deg, continuous, **calibrated: straight=0, down=+**)
      (4) phone_detected (bool)
      (5) timestamp (float)
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
        self.history = collections.deque(maxlen=600)  # lưu các record (mới nhất cuối cùng)
        self.last_alarm_ts = 0.0
        #phone
        self.phone_start_ts = None
        self.phone_last_seen_ts = None   # ← THÊM
        self.phone_reset_grace = 0.5     # ← 0.5 giây



        self.eye_closed_sec = EYE_CLOSED_SEC
        self.yawn_sec = YAWN_SEC
        self.window_accum_sec = WINDOW_ACCUM_SEC
        self.yawn_accum_total_sec = YAWN_ACCUM_TOTAL_SEC
        self.head_away_sec = HEAD_AWAY_SEC
        self.phone_away_sec = PHONE_HOLD_SEC

        # Alarm player (không chồng âm, phát đến hết)
        self.alarm_player = AlarmPlayer(volume=1.0)

        # ---- for pose continuity & smoothing ----
        self.prev_pose = None             # previous adjusted raw angles (yaw,pitch,roll)
        self.prev_pose_smoothed = None    # previous smoothed angles
        self.pose_smooth_alpha = 0.6      # EMA alpha (0..1). 0.6 is a reasonable default
        self.pose_baseline = None         # baseline for calibration (yaw,pitch,roll) on first valid frame

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
    def _mouth_open_ratio(self, pts):
        """
        Tính Mouth Aspect Ratio (MAR)
        Chỉ dùng landmarks – rất ổn định để phân biệt nói vs ngáp
        """
        if pts is None:
            return 0.0

        # dùng inner mouth cho chính xác
        inner = pts[MOUTH_INNER]

        # vertical distances
        v1 = np.linalg.norm(inner[2] - inner[6])
        v2 = np.linalg.norm(inner[3] - inner[5])

        # horizontal distance
        h = np.linalg.norm(inner[0] - inner[4])

        if h < 1e-6:
            return 0.0

        return (v1 + v2) / (2.0 * h)

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
    # Pose helpers: unwrap + smooth + baseline
    # -------------------------
    def _normalize_angle(self, a):
        """Normalize angle to [-180, 180]."""
        if a is None:
            return None
        a = float(a)
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def _adjust_angles_continuity_and_smooth(self, angles):
        """
        angles: tuple (yaw, pitch, roll) in degrees (each may be None),
        where each angle is in some canonical [-180,180] range.

        Returns: smoothed_adjusted tuple (yaw_s, pitch_s, roll_s)
        - unwraps to ensure temporal continuity
        - applies EMA smoothing
        - on first valid frame sets pose_baseline so that "straight" becomes 0
        - maps pitch so that: straight=0, down=positive (pitch = -raw_after_normalize - baseline_adjust)
        """
        if angles is None:
            return None

        # normalize raw inputs into [-180,180]
        yaw_r = self._normalize_angle(angles[0])
        pitch_r = self._normalize_angle(angles[1])
        roll_r = self._normalize_angle(angles[2])

        raw = (yaw_r, pitch_r, roll_r)

        # if no previous raw adjusted pose, initialize prevs and baseline
        if self.prev_pose is None:
            # initial adjust = raw (no unwrap)
            self.prev_pose = raw
            # initialize smoothed same as raw
            self.prev_pose_smoothed = raw
            # set baseline to current raw smoothed (so first frame defines "straight")
            # baseline used to subtract so that "straight" -> 0
            self.pose_baseline = raw
            # create returned smoothed and calibrated values (after baseline and pitch flip)
            yaw_adj = yaw_r if yaw_r is not None else 0.0
            # map pitch so that down=positive: flip sign and subtract baseline
            pitch_adj = None if pitch_r is None else (-(pitch_r - self.pose_baseline[1]))
            roll_adj = roll_r if roll_r is not None else 0.0
            smoothed = (yaw_adj, pitch_adj, roll_adj)
            self.prev_pose_smoothed = (yaw_adj, pitch_adj, roll_adj)
            return smoothed

        # unwrap: for each angle, try k in (-1,0,1) to minimize difference to prev
        adj_raw = []
        for new, prev in zip(raw, self.prev_pose):
            if new is None or prev is None:
                adj_raw.append(new)
                continue
            best = None
            best_diff = None
            for k in (-1, 0, 1):
                cand = new + 360.0 * k
                diff = abs(cand - prev)
                if best is None or diff < best_diff:
                    best = cand
                    best_diff = diff
            adj_raw.append(best)
        adj_raw = tuple(adj_raw)

        # smoothing (EMA) on adjusted raw values
        alpha = float(self.pose_smooth_alpha)
        if self.prev_pose_smoothed is None:
            smoothed_raw = adj_raw
        else:
            smoothed_raw_list = []
            for adj_val, prev_sm in zip(adj_raw, self.prev_pose_smoothed):
                if adj_val is None or prev_sm is None:
                    smoothed_raw_list.append(adj_val)
                else:
                    smoothed_raw_list.append(alpha * adj_val + (1.0 - alpha) * prev_sm)
            smoothed_raw = tuple(smoothed_raw_list)

        # update prevs
        self.prev_pose = adj_raw
        self.prev_pose_smoothed = smoothed_raw

        # Calibrate relative to baseline (so straight -> 0)
        if self.pose_baseline is None:
            self.pose_baseline = smoothed_raw

        # Compute calibrated values:
        # yaw: subtract baseline yaw (keep sign)
        yaw_cal = None if smoothed_raw[0] is None else (smoothed_raw[0] - self.pose_baseline[0])
        # pitch: we want straight=0 and down=positive -> flip sign relative to baseline
        # raw pitch increase direction depends on convention; flipping ensures down=positive.
        pitch_cal = None if smoothed_raw[1] is None else (smoothed_raw[1] - self.pose_baseline[1])
        # roll: subtract baseline
        roll_cal = None if smoothed_raw[2] is None else (smoothed_raw[2] - self.pose_baseline[2])

        return (yaw_cal, pitch_cal, roll_cal)

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
        r = self.face_model.predict(source=frame, verbose=False, conf=0.3)[0]
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
            pose_result = get_head_pose(pts, frame.shape)  # returns (yaw, pitch, roll) in degrees
            if pose_result:
                # Adjust for continuity + smoothing + baseline + pitch flip (down positive)
                adjusted = self._adjust_angles_continuity_and_smooth(pose_result)
                if adjusted is not None:
                    yaw, pitch, roll = adjusted  # yaw/pitch/roll calibrated: straight=0, pitch down positive

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

        # if ep is not None:
        #     eyes_closed = (ep[0] == 0)
        #     eye_conf = ep[2]
        # ---- YAWN LOGIC (STRICT: mouth must open WIDE) ----
        MAR_THRESHOLD = 0.75      # càng lớn → càng phải há to
        YAWN_CONF_TH = 0.80       # tránh nhầm nói chuyện

        mar = self._mouth_open_ratio(pts)

        if mp is not None:
            yawn_conf = mp[2]

            yawn_open = (
                mp[0] == 1 and
                yawn_conf >= YAWN_CONF_TH and
                mar >= MAR_THRESHOLD
            )
        else:
            yawn_open = False


        # --- UPDATE HISTORY ---
        current_yaw = yaw if yaw is not None else 0.0
        current_pitch = pitch if pitch is not None else 0.0
        now = time.time()

        # Lưu tuple: (eyes_closed, yawn_open, yaw, pitch, phone_detected, timestamp)
        self.history.append((eyes_closed, yawn_open, current_yaw, current_pitch, phone_detected, now))

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
        idx: index của flag (0:eyes_closed, 1:yawn_open, 2:yaw, 3:pitch, 4:phone, 5:ts)
        """
        if not hist:
            return 0.0
        total = 0.0
        last_ts = hist[0][5]
        for rec in hist:
            flag = rec[idx]
            ts = rec[5]
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
        last_ts = hist[0][5]
        for closed, yawn, yaw_angle, pitch_angle, phone, ts in hist:
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
        window_records = [(rec[1], rec[5]) for rec in self.history if rec[5] >= window_start_ts]
        if not window_records:
            return 0.0
        prev_ts = window_records[0][1]
        for yawn_flag, ts in window_records:
            if yawn_flag:
                yawn_accum += ts - prev_ts
            prev_ts = ts
        return yawn_accum

    def _calc_head_down(self, hist, pitch_threshold_deg):
        """
        Tính thời gian cúi đầu (pitch > threshold).
        Cho phép jitter nhỏ, không break ngay lập tức.
        """
        if not hist:
            return 0.0

        total = 0.0
        last_ts = hist[0][5]

        # cho phép dưới ngưỡng tối đa 0.2s để chống nhiễu
        grace_time = 0.2
        grace_used = 0.0

        for closed, yawn, yaw_angle, pitch_angle, phone, ts in hist:
            dt = max(last_ts - ts, 0.0)

            if pitch_angle > pitch_threshold_deg:
                # đang cúi → tăng thời gian
                total += dt
                grace_used = 0.0
            else:
                # jitter → chỉ cho phép tích lũy tối đa 0.2s
                grace_used += dt
                if grace_used > grace_time:
                    break

            last_ts = ts

        return total


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

# 2) phone duration (cộng dồn + grace 0.5s)
        phone_flag = self.history[-1][4] if len(self.history) > 0 else False

        if phone_flag:
            if self.phone_start_ts is None:
                self.phone_start_ts = now
            self.phone_last_seen_ts = now
            phone_dur = now - self.phone_start_ts
        else:
            if self.phone_last_seen_ts is not None:
                # chưa quá 0.5s → giữ nguyên
                if (now - self.phone_last_seen_ts) <= self.phone_reset_grace:
                    phone_dur = now - self.phone_start_ts
                else:
                    # quá 0.5s → reset
                    self.phone_start_ts = None
                    self.phone_last_seen_ts = None
                    phone_dur = 0.0
            else:
                phone_dur = 0.0


        # 3) head-away duration (liên tục)
        head_away_dur = self._calc_head_away(hist_rev, yaw_threshold_deg)

        # 4) yawn_accum trong cửa sổ
        window_start = now - self.window_accum_sec
        yawn_accum = self._calc_yawn_accum(window_start)

        # 5) head-down (cúi đầu)
        pitch_threshold_deg = 20.0   # góc cúi đầu ngưỡng (deg)
        head_down_dur = self._calc_head_down(hist_rev, pitch_threshold_deg)

        # 6) quyết định loại alarm (ưu tiên sleepy > not_focus > phone)
        alarm_type = None
        if (eyes_dur >= self.eye_closed_sec) or (yawn_dur >= self.yawn_sec) or (yawn_accum >= self.yawn_accum_total_sec):
            alarm_type = "sleepy"
        elif head_away_dur >= self.head_away_sec:
            alarm_type = "not_focus"
        elif head_down_dur >= self.head_away_sec:
            alarm_type = "not_focus"
        elif phone_dur >= self.phone_away_sec:
            alarm_type = "phone"

        # 7) phát âm thanh (cooldown tránh spam)
        if alarm_type is not None:
            # throttle phát để khỏi spam (ví dụ 1s)
            if (now - self.last_alarm_ts) > 1.0:
                try:
                    self.alarm_player.play(alarm_type)
                except Exception:
                    # giữ an toàn: không làm crash GUI nếu audio lỗi
                    pass
                self.last_alarm_ts = now

        # 8) trả kết quả (info cho GUI hiển thị debug)
        info = {
            "eyes_dur": eyes_dur,
            "yawn_dur": yawn_dur,
            "yawn_accum": yawn_accum,
            "head_away_dur": head_away_dur,
            "head_down_dur": head_down_dur,
            "phone_dur": phone_dur,
            "alarm_type": alarm_type
        }
        return alarm_type is not None, info
