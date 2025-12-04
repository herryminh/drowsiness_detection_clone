import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_EYE    = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
RIGHT_EYE   = [263, 362, 387, 386, 385, 373, 380, 381, 382, 390]

class FaceLandmarker:
    def __init__(self, static=False, max_faces=1):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    def __del__(self):
        try: self.mesh.close()
        except Exception: pass
    def _landmarks_to_np(self, face_landmarks, w, h):
        return np.array([(int(lm.x*w), int(lm.y*h)) for lm in face_landmarks.landmark])
    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        h,w = bgr.shape[:2]
        return self._landmarks_to_np(res.multi_face_landmarks[0], w, h)

def bbox_from_points(points, scale=1.2, img_shape=None):
    pts = np.array(points)
    x1, y1 = pts[:,0].min(), pts[:,1].min()
    x2, y2 = pts[:,0].max(), pts[:,1].max()
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = max(x2-x1,1), max(y2-y1,1)
    w *= scale; h *= scale
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    if img_shape is not None:
        H, W = img_shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)
    return x1, y1, x2, y2



def get_head_pose(landmarks_2d, img_shape):
    """
    Tính toán góc Yaw, Pitch, Roll từ 478 điểm mốc của MediaPipe.
    """
    h, w = img_shape[:2]

    # Ma trận camera
    focal_length = w
    cam_center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal_length, 0, cam_center[0]],
        [0, focal_length, cam_center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    face_3d_model_points = np.array([
        [0.0, 0.0, 0.0],           # Điểm 1 (Mũi)
        [-225.0, 170.0, -135.0],   # Điểm 33 (Mắt trái)
        [225.0, 170.0, -135.0],    # Điểm 263 (Mắt phải)
        [-150.0, -150.0, -125.0],  # Điểm 61 (Miệng trái)
        [150.0, -150.0, -125.0],   # Điểm 291 (Miệng phải)
        [0.0, -330.0, -65.0]       # Điểm 152 (Cằm)
    ], dtype=np.float32)

    face_2d_points = np.array([
        landmarks_2d[1],
        landmarks_2d[33],
        landmarks_2d[263],
        landmarks_2d[61],
        landmarks_2d[291],
        landmarks_2d[152],
    ], dtype=np.float32)

    try:
        success, rvec, tvec = cv2.solvePnP(
            face_3d_model_points, face_2d_points, cam_matrix, dist_coeffs
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rmat[2,1], rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0

        yaw = np.degrees(y)
        pitch = np.degrees(x)
        roll = np.degrees(z)
        return yaw, pitch, roll

    except Exception as e:
        # print(f"Lỗi PnP: {e}")
        return None