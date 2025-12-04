# Hệ thống cảnh báo buồn ngủ tài xế (YOLO + Landmarks + CNN + GUI) thực hiện bởi nhóm 4
## Sinh viên nhóm
22h1320012	Tôn Thất Bảo
22h1320023	Hồ Đăng Nguyên
22h1320034	Phạm Đăng Trình
	        Nguyễn Đình	Phương

Giải pháp phát hiện buồn ngủ theo thời gian thực từ camera:
- **YOLO** phát hiện **khuôn mặt** (hỗ trợ hiển thị & fallback).
- **MediaPipe FaceMesh (landmarks)** cắt **ROI mắt & miệng** chính xác.
- **2 CNN (ResNet18)** phân loại **mắt (Closed/Open)** và **ngáp (no_yawn/yawn)**.
- **GUI (PyQt5)** hiển thị khung hình, 3 khung ROI (face/eyes/mouth), 3 thumbnail ROI, % độ tin cậy, nút/điều chỉnh ngưỡng.


## Tính năng chính
- Cảnh báo khi:
  - **Mắt nhắm liên tục** ≥ ngưỡng.
  - **Ngáp liên tục** ≥ ngưỡng.
  - **Tổng thời gian ngáp** trong cửa sổ trượt ≥ ngưỡng.
- Hiển thị **% độ tin cậy** cho dự đoán mắt/ngáp.
- **Fallback**: mất landmarks → cắt ROI theo tỷ lệ trong khung mặt YOLO để không “mất khung”.

## Chạy nhanh
```bash
pip install -r requirements.txt
python -m train.train_eye -> huấn luyện mô hình phát hiện nhắm mắt
python -m train.train_yawn -> huấn luyện mô hình phát hiện ngáp
python app.gui -> để mở hệ thống của dự án