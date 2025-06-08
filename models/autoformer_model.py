import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask):
        B, L, D = queries.shape
        q_fft = torch.fft.rfft(queries, dim=1)
        k_fft = torch.fft.rfft(keys, dim=1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=1)
        new_x = corr + values
        return new_x, None

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x.transpose(1,2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1,2)
        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class AutoformerModel(nn.Module):
    def __init__(self, input_size, d_model, n_layers, d_ff, dropout=0.1, output_size=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = Encoder(
            [EncoderLayer(AutoCorrelation(False, attention_dropout=dropout), d_model, d_ff, dropout) for _ in range(n_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.output_proj = nn.Linear(d_model, output_size)
    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.encoder(x)
        output = self.output_proj(enc_out)
        return output[:, -1, :]

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
    date_test = dates[n_train+n_val:].reset_index(drop=True)
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
    res_df_list = []
    for i in range(pred_steps):
        rmse = np.sqrt(mean_squared_error(trues_inv[:,i], preds_inv[:,i]))
        mae = mean_absolute_error(trues_inv[:,i], preds_inv[:,i])
        mape = mean_absolute_percentage_error(trues_inv[:,i], preds_inv[:,i])
        r2 = r2_score(trues_inv[:,i], preds_inv[:,i])
        metrics.append([rmse, mae, mape, r2])
        n_sample = trues_inv.shape[0]  # hoặc preds_inv.shape[0], luôn đúng
        step_df = pd.DataFrame({
            "Ngày": date_test[:n_sample],  # chỉ lấy đúng số sample test
            "Giá thực tế": trues_inv[:, i],
            "Giá dự báo": preds_inv[:, i],
            "Sai số tuyệt đối": np.abs(trues_inv[:, i] - preds_inv[:, i]),
            "Bước dự báo": f"t+{i+1}"
        })
        res_df_list.append(step_df)
    res_df = pd.concat(res_df_list, axis=0).reset_index(drop=True)
    return res_df, metrics

def run_autoformer(
    df=None,
    csv_path=None,
    univariate=True,
    look_back=30,
    pred_steps=7,
    d_model=64,
    n_layers=2,
    d_ff=256,
    dropout=0.1,
    batch_size=32,
    epochs=50,
    lr=0.001
):
    print(f"\n🔵 Autoformer {'univariate' if univariate else 'multivariate'} — Dự báo {pred_steps} bước, window={look_back}")
    if df is None:
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        else:
            raise ValueError("Bạn phải truyền vào df hoặc csv_path!")
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
    model = AutoformerModel(input_size, d_model, n_layers, d_ff, dropout, output_size=pred_steps).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    res_df, metrics = evaluate_and_plot(model, test_loader, scaler_close, device, date_test, pred_steps)
    torch.save(model.state_dict(), f"autoformer_{'uni' if univariate else 'multi'}_{look_back}win_{pred_steps}step.pth")
    print("✅ Đã lưu model.")
    return model, res_df, metrics

if __name__ == "__main__":
    model, res_df, metrics = run_autoformer(
        csv_path="AXISBANK.csv",
        univariate=True,     # Hoặc False cho đa biến
        look_back=30,
        pred_steps=7,
        d_model=64,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        batch_size=32,
        epochs=50,
        lr=0.001
    )
    print(res_df.head())
    print(metrics)