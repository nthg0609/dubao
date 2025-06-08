import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, output_size=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_size)
    def forward(self, src):
        # src: (batch, seq_len, input_size)
        x = self.embedding(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.decoder(x)
        return out[:, -1, :]

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def analyze_data(df, features):
    print("\n=== PHÂN TÍCH THỐNG KÊ DỮ LIỆU ===")
    desc = df[features].describe().T
    print(desc)
    print("\nMissing values:\n", df[features].isnull().sum())
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Ma trận tương quan')
    plt.show()
    for col in features:
        sns.histplot(df[col], kde=True, bins=40)
        plt.title(f'Phân phối {col}')
        plt.show()

def create_windows(data, look_back, pred_steps):
    X, y = [], []
    for i in range(len(data) - look_back - pred_steps + 1):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+pred_steps, 0])
    return np.array(X), np.array(y)

def split_data(X, y, dates, train_ratio=0.8, val_ratio=0.1):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    date_test = dates.iloc[n_train+n_val:].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test, date_test

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
    preds = np.vstack(all_preds)
    trues = np.vstack(all_trues)
    preds_inv = scaler_close.inverse_transform(preds)
    trues_inv = scaler_close.inverse_transform(trues)
    metrics = []
    for i in range(pred_steps):
        rmse = np.sqrt(mean_squared_error(trues_inv[:,i], preds_inv[:,i]))
        mae = mean_absolute_error(trues_inv[:,i], preds_inv[:,i])
        mape = mean_absolute_percentage_error(trues_inv[:,i], preds_inv[:,i])
        r2 = r2_score(trues_inv[:,i], preds_inv[:,i])
        print(f"Bước t+{i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2%}, R2={r2:.4f}")
        metrics.append([rmse, mae, mape, r2])
    # Tạo bảng kết quả cho 10 dòng đầu tiên
    res_df = pd.DataFrame({
        "Ngày": date_test[:10],
        "Giá thực tế": trues_inv[:10,0],
        "Giá dự báo": preds_inv[:10,0],
        "Sai số tuyệt đối": np.abs(trues_inv[:10,0] - preds_inv[:10,0])
    })
    print("\n=== KẾT QUẢ ĐÁNH GIÁ MULTI-STEP (Transformer)===")
    for i in range(pred_steps):
        rmse = np.sqrt(mean_squared_error(trues_inv[:,i], preds_inv[:,i]))
        mae = mean_absolute_error(trues_inv[:,i], preds_inv[:,i])
        mape = mean_absolute_percentage_error(trues_inv[:,i], preds_inv[:,i])
        r2 = r2_score(trues_inv[:,i], preds_inv[:,i])
        print(f"Bước t+{i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2%}, R2={r2:.4f}")
    for step in range(pred_steps):
        plt.figure(figsize=(11,4))
        plt.plot(trues_inv[:,step], label=f'Thực tế t+{step+1}')
        plt.plot(preds_inv[:,step], label=f'Dự báo t+{step+1}')
        plt.title(f"Biểu đồ Transformer t+{step+1}")
        plt.legend()
        plt.show()
    plt.figure(figsize=(5,5))
    plt.scatter(trues_inv.flatten(), preds_inv.flatten(), alpha=0.5)
    plt.xlabel("Giá thực tế"); plt.ylabel("Giá dự báo"); plt.title("Scatter: Actual vs Predicted")
    plt.show()
    plt.figure(figsize=(7,3))
    plt.hist((trues_inv-preds_inv).flatten(), bins=40)
    plt.title("Histogram lỗi"); plt.show()
    return model, res_df, metrics
def run_transformer(
    df=None,
    csv_path=None,
    univariate=True,
    look_back=30,
    pred_steps=7,
    d_model=64,
    nhead=8,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    batch_size=32,
    epochs=50,
    lr=0.001
):
    print(f"\n🔵 Transformer {'univariate' if univariate else 'multivariate'} — Dự báo {pred_steps} bước, window={look_back}")
    # --- Đọc dữ liệu ---
    if df is None:
        if csv_path is None:
            raise ValueError("Cần truyền vào một trong hai: df hoặc csv_path")
        df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') 
    df = df.sort_values(by='Date')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
    analyze_data(df, features)
    feature_cols = ['Close'] if univariate else features
    scaler_dict = {}
    scaled_data = df[feature_cols].copy()
    for col in feature_cols:
        scaler = MinMaxScaler()
        scaled_data[col] = scaler.fit_transform(df[[col]])
        scaler_dict[col] = scaler
    values = scaled_data.values
    X, y = create_windows(values, look_back, pred_steps)
    scaler_close = scaler_dict['Close']
    dates = df['Date']
    X_train, X_val, X_test, y_train, y_val, y_test, date_test = split_data(X, y, dates)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = torch.utils.data.DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)
    input_size = X.shape[2]
    model = TransformerModel(input_size, d_model, nhead, num_layers, dim_feedforward, dropout, pred_steps).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    model, res_df, metrics = evaluate_and_plot(model, test_loader, scaler_close, device, date_test, pred_steps)
    torch.save(model.state_dict(), f"transformer_{'uni' if univariate else 'multi'}_{look_back}win_{pred_steps}step.pth")
    print("✅ Đã lưu model.")
    return model, res_df, metrics

if __name__ == "__main__":
    run_transformer(
        csv_path="AXISBANK.csv",
        univariate=True,     # Hoặc False cho đa biến
        look_back=30,
        pred_steps=7,
        d_model=64,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        batch_size=32,
        epochs=50,
        lr=0.001
    )