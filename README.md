# Dự báo giá cổ phiếu sử dụng nhiều mô hình

Dự án này sử dụng nhiều mô hình deep learning và machine learning khác nhau để dự báo giá cổ phiếu.

## Các mô hình được sử dụng

1. LSTM (Long Short-Term Memory)
2. TCN (Temporal Convolutional Network)
3. Transformer
4. Hybrid (LSTM + Attention)
5. Informer
6. Random Forest
7. SVR (Support Vector Regression)

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử dụng

1. Chạy ứng dụng:
```bash
streamlit run app.py
```

2. Upload file dữ liệu CSV với các cột sau:
- Date: Ngày
- Open: Giá mở cửa
- High: Giá cao nhất
- Low: Giá thấp nhất
- Close: Giá đóng cửa
- Volume: Khối lượng
- VWAP: Giá trung bình có trọng số theo khối lượng

3. Cấu hình mô hình:
- Chọn loại mô hình
- Điều chỉnh các tham số
- Nhấn nút "Huấn luyện mô hình"

4. Xem kết quả:
- Các chỉ số đánh giá (RMSE, MAE, MAPE, R²)
- Biểu đồ so sánh giá thực tế và dự báo

## Cấu trúc thư mục

```
.
├── app.py              # Ứng dụng Streamlit
├── models/             # Thư mục chứa các mô hình
│   ├── lstm_model.py
│   ├── tcn_model.py
│   ├── transformer_model.py
│   ├── hybrid_model.py
│   ├── informer_model.py
│   ├── rf_model.py
│   └── svr_model.py
├── requirements.txt    # Các thư viện cần thiết
└── README.md          # Hướng dẫn sử dụng
```

## Tính năng

1. Phân tích dữ liệu:
- Thống kê mô tả
- Kiểm tra giá trị thiếu
- Ma trận tương quan
- Biểu đồ phân phối

2. Tiền xử lý dữ liệu:
- Chuẩn hóa dữ liệu
- Tạo chuỗi dữ liệu cho dự báo nhiều bước
- Chia tập train/val/test

3. Huấn luyện và đánh giá:
- Huấn luyện mô hình với validation
- Tính toán các chỉ số đánh giá
- Vẽ biểu đồ so sánh

4. Lưu và tải mô hình:
- Lưu trọng số mô hình
- Lưu scaler để tái sử dụng

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để đóng góp.

## Giấy phép

Dự án này được cấp phép theo giấy phép MIT. Xem file LICENSE để biết thêm chi tiết. 