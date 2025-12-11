# gui_new.py
# Giao diện đẹp theo layout mẫu, logic giữ nguyên 100%

import sys, pathlib
import cv2
import numpy as np
import warnings
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QFrame
)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor

# ------------------------------------------------------------
# IMPORT CORE
# ------------------------------------------------------------
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

warnings.filterwarnings("ignore")


# ============================================================
# HÀM SET THUMB (GIỮ NGUYÊN)
# ============================================================
def _set_thumb(label: QLabel, roi):
    if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
        label.setText("N/A")
        label.setPixmap(QPixmap())
        return

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg).scaled(160, 120, Qt.KeepAspectRatio))


# ============================================================
# GUI CLASS (GIAO DIỆN MỚI)
# ============================================================
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Giám sát buồn ngủ | Thiết bị: {DEVICE}")
        self.setMinimumSize(1200, 750)
        self.resize(1600, 950)
        self.sys = DrowsinessSystem()
        self.cap = None

        self._setup_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loop)

    # ---------------------------------------------------------
    # SETUP UI
    # ---------------------------------------------------------
    def _setup_ui(self):
        # VIDEO
        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setFrameShape(QFrame.NoFrame)

        # INFO HEADER
        self.header = QLabel("Trạng thái: Sẵn sàng")
        self.header.setObjectName("header")

        self.info = QLabel("...")
        self.info.setObjectName("info")

        # THUMBNAILS
        self.face_thumb = QLabel("Face")
        self.eye_thumb = QLabel("Eye")
        self.mouth_thumb = QLabel("Mouth")

        # BUTTONS
        self.btn_start = QPushButton("Bắt đầu")
        self.btn_stop = QPushButton("Dừng")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # NGƯỠNG (LẤY NGUYÊN FILE CŨ)
        box = self._build_threshold_box()

        # ======= LAYOUT =======
        left = QVBoxLayout()
        left.addWidget(self.header)
        left.addWidget(self.video, 1)
        left.addWidget(self.info)

        # Right sidebar kiểu card
        card = QVBoxLayout()
        card.addWidget(box)
        card.addSpacing(12)
        card.addWidget(self.btn_start)
        card.addWidget(self.btn_stop)
        card.addSpacing(20)

        card.addWidget(QLabel("Khung mặt"))
        card.addWidget(self.face_thumb)
        card.addWidget(QLabel("Vùng mắt"))
        card.addWidget(self.eye_thumb)
        card.addWidget(QLabel("Vùng miệng"))
        card.addWidget(self.mouth_thumb)
        card.addStretch(1)

        right_card = QFrame()
        right_card.setObjectName("card")
        right_card.setLayout(card)

        layout = QHBoxLayout()
        layout.addLayout(left, 3)
        layout.addWidget(right_card, 1)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # STYLE (THEO MẪU)
    # ---------------------------------------------------------
    def _apply_style(self):
        self.setStyleSheet("""
    QWidget {
        background-color: #1E1E2E;
        color: #E0E0E0;
        font-size: 14px;
    }

    QLabel {
        color: #E0E0E0;
    }

    QGroupBox {
        background-color: #2A2A3C;
        border: 1px solid #3A3A4F;
        border-radius: 8px;
        margin-top: 10px;
        padding: 10px;
        font-weight: bold;
    }

    QDoubleSpinBox {
        background-color: #2A2A3C;
        color: #E0E0E0;
        border: 1px solid #3A3A4F;
        border-radius: 6px;
        padding: 4px;
    }

    QPushButton {
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: bold;
        color: white;
        background-color: #7C4DFF;
    }

    QPushButton:hover {
        background-color: #6C3FE6;
    }
""")


        self.video.setStyleSheet("""
                border: 3px solid #2196F3;
                border-radius: 6px;
            """)
        self.btn_start.setStyleSheet("""
            background-color: #4CAF50;
            """)

        self.btn_start.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 6px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #43A047;
                }
            """)

        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)



    # ---------------------------------------------------------
    # NGƯỠNG – LẤY ĐÚNG Y NHƯ FILE CŨ
    # ---------------------------------------------------------
    def _build_threshold_box(self):
        self.spin_eye = QDoubleSpinBox()
        self.spin_yawn = QDoubleSpinBox()
        self.spin_acc = QDoubleSpinBox()
        self.spin_head = QDoubleSpinBox()
        self.spin_phone = QDoubleSpinBox()

        settings = [
            (self.spin_eye, 0.5, 5.0, self.sys.eye_closed_sec, 0.1),
            (self.spin_yawn, 0.5, 5.0, self.sys.yawn_sec, 0.1),
            (self.spin_acc, 1.0, 60.0, self.sys.yawn_accum_total_sec, 0.5),
            (self.spin_head, 1.0, 10.0, self.sys.head_away_sec, 0.5),
            (self.spin_phone, 1.0, 15.0, PHONE_HOLD_SEC, 0.5),
        ]

        for spin, mn, mx, val, step in settings:
            spin.setRange(mn, mx)
            spin.setValue(val)
            spin.setSingleStep(step)

        form = QFormLayout()
        form.addRow("Ngưỡng mắt nhắm (giây)", self.spin_eye)
        form.addRow("Ngưỡng ngáp (giây)", self.spin_yawn)
        form.addRow("Tổng ngáp tích lũy (giây)", self.spin_acc)
        form.addRow("Ngưỡng quay đầu (giây)", self.spin_head)
        form.addRow("Ngưỡng cầm ĐT (giây)", self.spin_phone)

        box = QGroupBox("Ngưỡng cảnh báo")
        box.setLayout(form)
        return box

    # ---------------------------------------------------------
    # START / STOP
    # ---------------------------------------------------------
    def start(self):
        self.header.setText("Trạng thái: Đang chạy...")

        self.sys.yawn_accum_current = 0
        self.sys.phone_duration = 0
        self.sys.eye_close_duration = 0
        self.sys.head_away_duration = 0

        self.sys.eye_closed_sec       = self.spin_eye.value()
        self.sys.yawn_sec             = self.spin_yawn.value()
        self.sys.yawn_accum_total_sec = self.spin_acc.value()
        self.sys.head_away_sec        = self.spin_head.value()
        self.sys.phone_away_sec       = self.spin_phone.value()

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.header.setText("Không mở được camera")
            return

        self.timer.start(0)


    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        self.header.setText("Trạng thái: Đã dừng")


    # ---------------------------------------------------------
    # VẼ TEXT
    # ---------------------------------------------------------
    def _paint_vn_text(self, qimg: QImage, alarm, eyes_closed, yawn_open, phone_detected):
        painter = QPainter(qimg)
        font_name = self.font().family()

        w = qimg.width()
        h = qimg.height()

        box_width  = 600
        box_height = 90

        center_x = (w - box_width) // 2
        center_y = 20   # khoảng cách từ viền trên

        # ===== Nền mờ =====
        painter.setOpacity(0.65)
        painter.fillRect(center_x, center_y, box_width, box_height, QColor(0, 0, 0))
        painter.setOpacity(1.0)

        # ===== TEXT 1 =====
        painter.setPen(QColor(255, 255, 0))
        painter.setFont(QFont(font_name, 26, QFont.Bold))

        text1 = (
            f"Mắt nhắm: {'Có' if eyes_closed else 'Không'}   |   "
            f"Ngáp: {'Có' if yawn_open else 'Không'}   |   "
            f"Cầm ĐT: {'Có' if phone_detected else 'Không'}"
        )

        # căn giữa
        text1_width = painter.fontMetrics().boundingRect(text1).width()
        painter.drawText(center_x + (box_width - text1_width) // 2, center_y + 35, text1)

        # ===== TEXT 2 =====
        painter.setFont(QFont(font_name, 26, QFont.Black))
        painter.setPen(QColor(255, 30, 30) if alarm else QColor(50, 255, 50))

        text2 = "CẢNH BÁO!" if alarm else "Bình thường"
        text2_width = painter.fontMetrics().boundingRect(text2).width()
        painter.drawText(center_x + (box_width - text2_width) // 2, center_y + 70, text2)

        painter.end()


    # ---------------------------------------------------------
    # LOOP
    # ---------------------------------------------------------
    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        (draw_box, eyes_closed, yawn_open,
         face_roi, eye_roi, mouth_roi,
         eyes_box, mouth_box, face_xyxy,
         eye_conf, yawn_conf,
         pose_data, face_landmarks_2d,
         phone_detected, phone_xyxy) = self.sys.step(frame)

        alarm, info = self.sys.should_alarm()

        # Vẽ box giữ nguyên logic
        for box, color in [
            (face_xyxy, (0, 255, 0)),
            (eyes_box, (0, 255, 255)),
            (mouth_box, (255, 0, 0)),
            (phone_xyxy, (0, 0, 255)),
        ]:
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Landmarks
        if face_landmarks_2d is not None:
            lm = landmark_pb2.NormalizedLandmarkList()
            h, w = frame.shape[:2]
            for x, y in face_landmarks_2d:
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x = x / w
                landmark.y = y / h
                lm.landmark.append(landmark)

            mp_drawing.draw_landmarks(
                frame, lm,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
            )
             # Vẽ contour
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=lm,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format_RGB888)
        self._paint_vn_text(qimg, alarm, eyes_closed, yawn_open, phone_detected)

        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            1080, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        self._update_info(info, pose_data, phone_detected)

        _set_thumb(self.face_thumb, face_roi)
        _set_thumb(self.eye_thumb, eye_roi)
        _set_thumb(self.mouth_thumb, mouth_roi)

    # ---------------------------------------------------------
    # UPDATE INFO
    # ---------------------------------------------------------
    def _update_info(self, info, pose_data, phone_detected):
        yaw = pose_data[0] if pose_data is not None else None
        yaw_str = f"{yaw:.1f}°" if yaw is not None else "N/A"

        phone_dur = info.get("phone_dur", 0)

        self.info.setText(
            f"Mắt nhắm: {info.get('eyes_dur',0):.2f}s | "
            f"Ngáp: {info.get('yawn_dur',0):.2f}s | "
            f"Tổng ngáp: {info.get('yawn_accum',0):.2f}s | "
            f"Quay đầu: {info.get('head_away_dur',0):.2f}s (Yaw: {yaw_str}) | "
            f"ĐT: {'Có' if phone_detected else 'Không'} ({phone_dur:.2f}s)"
        )


# MAIN
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
