# -*- coding: utf-8 -*-
"""
GUI tiếng Việt — hỗ trợ:
- Vẽ khung mặt (YOLO), mắt & miệng (landmarks)
- Hiển thị ROI thumbnails (Face / Eye / Mouth)
- Hiển thị % độ tin cậy của CNN
- Điều chỉnh ngưỡng cảnh báo trực tiếp trong GUI
"""

# =============================================================
# IMPORT — gom vào 1 chỗ cho dễ quản lý
# =============================================================
import sys, pathlib
import cv2
import numpy as np
import warnings
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor

# =============================================================
# FIX chạy trực tiếp hoặc chạy dạng module
# =============================================================
if __package__ is None or __package__ == "":
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    from app.core import DrowsinessSystem, DEVICE
    from app.landmarks import MOUTH_OUTER, MOUTH_INNER, LEFT_EYE, RIGHT_EYE
    from app.config import PHONE_HOLD_SEC
else:
    from .core import DrowsinessSystem, DEVICE
    from .landmarks import MOUTH_OUTER, MOUTH_INNER, LEFT_EYE, RIGHT_EYE
    from .config import PHONE_HOLD_SEC

# Ẩn cảnh báo protobuf (mediapipe)
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated",
    category=UserWarning
)


# =============================================================
# HÀM PHỤ: Hiển thị thumbnail ROI
# =============================================================
def _set_thumb(label: QLabel, roi):
    """Hiển thị ROI lên thumbnail; nếu None → chữ 'N/A' """
    if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
        label.setText("N/A")
        label.setPixmap(QPixmap())
        return

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg).scaled(160, 120, Qt.KeepAspectRatio))



# =============================================================
# CLASS GUI
# =============================================================
class MainWindow(QWidget):

    # ---------------------------------------------------------
    # KHỞI TẠO — Setup toàn bộ giao diện
    # ---------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Giám sát buồn ngủ | Thiết bị: {DEVICE}')

        self.sys = DrowsinessSystem()
        self.cap = None

        # ------------ KHU VIDEO + TRẠNG THÁI -------------
        self.video = QLabel()
        self.status = QLabel('Trạng thái: Sẵn sàng')
        self.info = QLabel('...')

        # ------------ THUMBNAIL ROI -------------
        self.face_thumb = QLabel('Khung mặt')
        self.eye_thumb  = QLabel('Vùng mắt')
        self.mouth_thumb= QLabel('Vùng miệng')

        # ------------ NÚT -------------
        self.btn_start = QPushButton('Bắt đầu')
        self.btn_stop  = QPushButton('Dừng')
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # ------------ TIMER LOOP -------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop)

        # ------------ TẠO FORM NGƯỠNG -------------
        box = self._build_threshold_box()

        # ------------ BỐ CỤC -------------
        left = QVBoxLayout()
        left.addWidget(self.video)
        left.addWidget(self.status)
        left.addWidget(self.info)

        right = QVBoxLayout()
        right.addWidget(box)
        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addWidget(QLabel('Khung mặt'));  right.addWidget(self.face_thumb)
        right.addWidget(QLabel('Vùng mắt'));   right.addWidget(self.eye_thumb)
        right.addWidget(QLabel('Vùng miệng')); right.addWidget(self.mouth_thumb)
        right.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(left, 3)
        layout.addLayout(right, 1)
        self.setLayout(layout)

        # Font đẹp và hỗ trợ tiếng Việt
        try:
            self.setFont(QFont('Segoe UI', 10))
        except:
            self.setFont(QFont('Arial', 10))



    # ---------------------------------------------------------
    # TẠO HỘP CÁC NGƯỠNG CẢNH BÁO
    # ---------------------------------------------------------
    def _build_threshold_box(self):
        self.spin_eye  = QDoubleSpinBox()
        self.spin_yawn = QDoubleSpinBox()
        self.spin_acc  = QDoubleSpinBox()
        self.spin_head = QDoubleSpinBox()
        self.spin_phone= QDoubleSpinBox()

        # Set ranges + default values
        settings = [
            (self.spin_eye ,   0.5, 5.0, self.sys.eye_closed_sec,      0.1),
            (self.spin_yawn,   0.5, 5.0, self.sys.yawn_sec,            0.1),
            (self.spin_acc ,   1.0, 60.0, self.sys.yawn_accum_total_sec,0.5),
            (self.spin_head,   1.0, 10.0, self.sys.head_away_sec,      0.5),
            (self.spin_phone,  1.0, 15.0, PHONE_HOLD_SEC,              0.5),
        ]
        for spin, mn, mx, val, step in settings:
            spin.setRange(mn, mx)
            spin.setValue(val)
            spin.setSingleStep(step)

        form = QFormLayout()
        form.addRow('Ngưỡng mắt nhắm (giây)', self.spin_eye)
        form.addRow('Ngưỡng ngáp (giây)', self.spin_yawn)
        form.addRow('Tổng ngáp tích lũy (giây)', self.spin_acc)
        form.addRow('Ngưỡng quay đầu (giây)', self.spin_head)
        form.addRow('Ngưỡng cầm ĐT (giây)', self.spin_phone)

        box = QGroupBox('Ngưỡng cảnh báo')
        box.setLayout(form)
        return box



    # ---------------------------------------------------------
    # START / STOP
    # ---------------------------------------------------------
    def start(self):
          # Reset tất cả giá trị tích lũy
        self.sys.yawn_accum_current = 0
        self.sys.phone_duration = 0
        self.sys.eye_close_duration = 0
        self.sys.head_away_duration = 0
        # Cập nhật tất cả ngưỡng
        self.sys.eye_closed_sec       = self.spin_eye.value()
        self.sys.yawn_sec             = self.spin_yawn.value()
        self.sys.yawn_accum_total_sec = self.spin_acc.value()
        self.sys.head_away_sec        = self.spin_head.value()
        self.sys.phone_away_sec       = self.spin_phone.value()

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.status.setText("Không mở được camera")
            return

        self.timer.start(0)

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status.setText("Đã dừng")



    # ---------------------------------------------------------
    # HÀM VẼ TEXT TIẾNG VIỆT (QPainter)
    # ---------------------------------------------------------
    def _paint_vn_text(self, qimg: QImage, alarm, eyes_closed, yawn_open, phone_detected):
        painter = QPainter(qimg)
        font_name = self.font().family()

        # Nền mờ
        painter.setOpacity(0.7)
        painter.fillRect(6, 6, 520, 70, QColor(0, 0, 0))
        painter.setOpacity(1.0)

        # Dòng 1
        painter.setPen(QColor(255, 255, 0))
        painter.setFont(QFont(font_name, 30, QFont.Bold))
        text1 = (
            f"Mắt nhắm: {'Có' if eyes_closed else 'Không'}   |   "
            f"Ngáp: {'Có' if yawn_open else 'Không'}   |   "
            f"Cầm ĐT: {'Có' if phone_detected else 'Không'}"
        )
        painter.drawText(QPoint(12, 30), text1)

        # Dòng 2
        painter.setFont(QFont(font_name, 30, QFont.Black))
        painter.setPen(QColor(255, 30, 30) if alarm else QColor(50, 255, 50))
        painter.drawText(
            QPoint(12, 60),
            "CẢNH BÁO!" if alarm else "Bình thường"
        )
        painter.end()



    # ---------------------------------------------------------
    # LOOP — mỗi frame
    # ---------------------------------------------------------
    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        # Lấy output từ core
        (draw_box, eyes_closed, yawn_open,
         face_roi, eye_roi, mouth_roi,
         eyes_box, mouth_box, face_xyxy,
         eye_conf, yawn_conf,
         pose_data, face_landmarks_2d,
         phone_detected, phone_xyxy,) = self.sys.step(frame)

        alarm, info = self.sys.should_alarm()

        
        # ===== VẼ KHUNG SAFE =====
        boxes_colors = [
            (face_xyxy, (0, 255, 0)),   # Xanh lá: mặt
            (eyes_box, (0, 255, 255)),  # Vàng: mắt
            (mouth_box, (255, 0, 0)),   # Xanh dương: miệng
            (phone_xyxy, (0, 0, 255))   # Đỏ: điện thoại
        ]

        for box, color in boxes_colors:
            if box is not None and len(box) == 4:
                try:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                except Exception as e:
                    print("Lỗi khi vẽ box:", box, e)

       
        # ===== VẼ LANDMARKS =====
        if face_landmarks_2d is not None:
            

            lm_list = landmark_pb2.NormalizedLandmarkList()
            h, w = frame.shape[:2]
            for x, y in face_landmarks_2d:
                lm = landmark_pb2.NormalizedLandmark()
                lm.x = float(x / w)
                lm.y = float(y / h)
                lm.z = 0.0
                lm_list.landmark.append(lm)
            
           
             
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=lm_list,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,255), thickness=1)
            )
            # Vẽ contour
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=lm_list,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
            )

        # ===== VẼ UNICODE TEXT =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._paint_vn_text(qimg, alarm, eyes_closed, yawn_open, phone_detected)

        # ===== UPDATE GUI =====
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(1080, 720, Qt.KeepAspectRatio , Qt.SmoothTransformation))
        self._update_info(info, pose_data, phone_detected)

        _set_thumb(self.face_thumb, face_roi)
        _set_thumb(self.eye_thumb , eye_roi)
        _set_thumb(self.mouth_thumb, mouth_roi)



    # ---------------------------------------------------------
    # LANDMARK SMOOTHING
    # ---------------------------------------------------------
    def _smooth_landmarks(self, lm):
        if lm is None:
            return getattr(self, "last_landmarks", None)

        lm = np.array(lm)
        if not hasattr(self, "smoothed_landmarks"):
            self.smoothed_landmarks = lm.copy()

        alpha = 0.9
        self.smoothed_landmarks = alpha * lm + (1 - alpha) * self.smoothed_landmarks
        self.last_landmarks = self.smoothed_landmarks.astype(int)
        return self.last_landmarks



    # ---------------------------------------------------------
    # UPDATE BOX INFO
    # ---------------------------------------------------------
    def _update_info(self, info, pose_data, phone_detected):
        yaw = pose_data[0] if pose_data is not None else None
        yaw_str = f"{yaw:.1f}°" if yaw is not None else "N/A"

        phone_dur = info.get("phone_dur", None)
        phone_dur_str = f"{phone_dur:.2f}s" if phone_dur is not None else "N/A"

        # self.status.setText("Trạng thái: CẢNH BÁO" if info.get("alarm") else "Trạng thái: Bình thường")
        self.info.setText(
            f"Mắt nhắm: {info.get('eyes_dur',0):.2f}s | "
            f"Ngáp: {info.get('yawn_dur',0):.2f}s | "
            f"Ngáp tích lũy: {info.get('yawn_accum',0):.2f}s | "
            f"Quay đầu: {info.get('head_away_dur',0):.2f}s (Góc: {yaw_str}) | "
            f"Cầm ĐT: {'Có' if phone_detected else 'Không'} ({phone_dur_str})"
        )



# =============================================================
# MAIN
# =============================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
