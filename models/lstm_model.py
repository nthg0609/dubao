import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import timedelta

# ================== 1. Cố định seed reproducibility ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

# ================== 2. Lớp Dataset ==================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ================== 3. Model LSTM PyTorch ==================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Lấy hidden state cuối cùng
        return self.fc(out[:, -1, :]) # (batch_size, output_size)

# ================== 4. Phân tích thống kê & mô tả dữ liệu ==================
def analyze_data(df, features):
    print("\n=== PHÂN TÍCH THỐNG KÊ DỮ LIỆU ===")
    stat = df[features].describe().T
    stat = stat[['count','mean','std','min','25%','50%','75%','max']]
    stat.columns = [
        "Số lượng quan sát", "Giá trị trung bình", "Độ lệch chuẩn", 
        "Giá trị nhỏ nhất", "Phân vị 25%", "Trung vị (50%)",
        "Phân vị 75%", "Giá trị lớn nhất"
    ]
    print(stat)
    print("\nKiểm tra giá trị thiếu:")
    print(df[features].isnull().sum())
    print("\n🔗 Ma trận tương quan:")
    plt.figure(figsize=(10,7))
    sns.heatmap(df[features].corr(), annot=True, cmap='Blues', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    for col in features:
        plt.figure(figsize=(5,2))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f'Phân phối {col}')
        plt.tight_layout()
        plt.show()

# ================== 5. Tiền xử lý, chuẩn hóa, tạo window ==================
def preprocess_data(
    df, 
    features, 
    univariate=True, 
    look_back=30, 
    pred_steps=7
):
    # -- Chỉ lấy các cột cần thiết
    feature_cols = ['Close'] if univariate else features
    scaler_dict = {}
    scaled_data = df[feature_cols].copy()
    # Chuẩn hóa từng biến riêng
    for col in feature_cols:
        scaler = MinMaxScaler()
        scaled_data[col] = scaler.fit_transform(df[[col]])
        scaler_dict[col] = scaler
    values = scaled_data.values
    # Tạo sliding window cho multi-step
    X, y = [], []
    for i in range(len(values) - look_back - pred_steps + 1):
        X.append(values[i:i+look_back, :])              # shape: (look_back, n_features)
        y.append(values[i+look_back:i+look_back+pred_steps, 0]) # chỉ lấy cột 'Close' cho nhãn (dù multivariate)
    X, y = np.array(X), np.array(y)
    # Lấy các ngày tương ứng cho y (dùng để vẽ/truy vết)
    date_targets = df['Date'].iloc[look_back: look_back+len(y)].reset_index(drop=True)
    return X, y, scaler_dict, date_targets

# ================== 6. Chia train/val/test theo thời gian ==================
def split_data(X, y, dates, train_ratio=0.8, val_ratio=0.1):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    date_test = dates[n_train+n_val:].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test, date_test

# ================== 7. Huấn luyện model ==================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} — Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# ================== 8. Đánh giá model và phân tích lỗi ==================
def evaluate_and_plot(model, test_loader, scaler_close, device, date_test, pred_steps):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            trues = y_batch.numpy()
            all_preds.append(preds)
            all_trues.append(trues)
    preds = np.vstack(all_preds)     # (N, pred_steps)
    trues = np.vstack(all_trues)     # (N, pred_steps)
    # Inverse transform cột Close
    preds_inv = scaler_close.inverse_transform(preds)
    trues_inv = scaler_close.inverse_transform(trues)
    # ==== Đánh giá per-step ====
    print("\n=== KẾT QUẢ ĐÁNH GIÁ MULTI-STEP ===")
    metrics = []
    for i in range(pred_steps):
        rmse = np.sqrt(mean_squared_error(trues_inv[:,i], preds_inv[:,i]))
        mae = mean_absolute_error(trues_inv[:,i], preds_inv[:,i])
        mape = mean_absolute_percentage_error(trues_inv[:,i], preds_inv[:,i])
        r2 = r2_score(trues_inv[:,i], preds_inv[:,i])
        print(f"Bước t+{i+1}: RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE={mape:.2%} | R2={r2:.4f}")
        metrics.append([rmse, mae, mape, r2])
    # ==== Tổng hợp bảng kết quả ====
    res_df = pd.DataFrame({
        "Ngày": [d+timedelta(days=k) for d in date_test for k in range(1, pred_steps+1)],
        "Giá thực tế": trues_inv.flatten(),
        "Giá dự báo": preds_inv.flatten(),
        "Sai số tuyệt đối": np.abs(trues_inv.flatten() - preds_inv.flatten())
    })
    print("\n== Bảng kết quả (mẫu):")
    print(res_df.head(10))
    # ==== Biểu đồ scatter giá thực vs giá dự báo ====
    plt.figure(figsize=(6,6))
    plt.scatter(trues_inv.flatten(), preds_inv.flatten(), alpha=0.5)
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự báo")
    plt.title("Scatter: Giá thực tế vs Giá dự báo (toàn bộ multi-step)")
    plt.grid(True)
    plt.show()
    # ==== Histogram error ====
    plt.figure(figsize=(7,3))
    plt.hist(trues_inv.flatten()-preds_inv.flatten(), bins=40, alpha=0.7)
    plt.title("Histogram sai số (error = thực tế - dự báo)")
    plt.xlabel("Error")
    plt.ylabel("Số lượng")
    plt.show()
    # ==== Plot từng bước dự báo ====
    for step in range(pred_steps):
        plt.figure(figsize=(11,4))
        plt.plot(trues_inv[:,step], label=f'Thực tế t+{step+1}')
        plt.plot(preds_inv[:,step], label=f'Dự báo t+{step+1}')
        plt.legend()
        plt.title(f"Biểu đồ dự báo bước t+{step+1}")
        plt.tight_layout()
        plt.show()
    return res_df, metrics

# ================== 9. Hàm main dùng cho giao diện hoặc CLI ==================
def run_lstm_model(
    df=None,
    csv_path=None,
    univariate=True,
    look_back=30,
    pred_steps=7,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    batch_size=32,
    epochs=50,
    lr=0.001,
    plot=True
):
    print(f"\n🔵 Khởi động mô hình LSTM {'univariate' if univariate else 'multivariate'} — Dự báo {pred_steps} bước, look_back={look_back} ngày")
    # ===== 1. Load dữ liệu
    if df is None:
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        else:
            raise ValueError("Bạn phải truyền vào df hoặc csv_path!")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values(by='Date')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']

    if plot:
        analyze_data(df, features)
    # ===== 3. Tạo window + chuẩn hóa + lấy scaler
    X, y, scaler_dict, date_targets = preprocess_data(df, features, univariate, look_back, pred_steps)
    scaler_close = scaler_dict['Close']
    # ===== 4. Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test, date_test = split_data(X, y, date_targets)
    # ===== 5. DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)
    # ===== 6. Build model
    input_size = X.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, pred_steps, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ===== 7. Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    # ===== 8. Đánh giá và phân tích lỗi
    res_df, metrics = evaluate_and_plot(model, test_loader, scaler_close, device, date_test, pred_steps)
    # ===== 9. Lưu model (nếu cần)
    torch.save(model.state_dict(), f"lstm_{'uni' if univariate else 'multi'}_{look_back}win_{pred_steps}step.pth")
    print("✅ Đã lưu model vào file.")
    return model, res_df, metrics

# ============= 10. Ví dụ chạy từ CLI hoặc gọi từ app.py/streamlit =============
if __name__ == "__main__":
    # Ví dụ: Dự báo 7 ngày, đơn biến, window 30, multivariate=True/False đều được
    run_lstm_model(
        csv_path="AXISBANK.csv",      # Đường dẫn tới dữ liệu
        univariate=False,             # True: chỉ 'Close', False: đa biến (các đặc trưng)
        look_back=30,                 # Độ dài chuỗi vào
        pred_steps=7,                 # Số ngày dự báo
        hidden_size=64,               # Số neuron ẩn
        num_layers=2,                 # Số lớp LSTM
        dropout=0.2,                  # Dropout giữa các lớp
        batch_size=32,
        epochs=50,
        lr=0.001,
        plot=True                     # Hiển thị biểu đồ/phân tích
    )