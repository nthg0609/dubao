import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

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
        y.append(data[i+look_back:i+look_back+pred_steps, 3])  # chỉ lấy cột Close
    return np.array(X), np.array(y)

def split_data(X, y, dates, train_ratio=0.8, val_ratio=0.1):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    date_test = dates.iloc[n_train+n_val+len(X[0])+len(y[0])-1:].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test, date_test

def train_rf_multi_step(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5):
    models = []
    n_steps = y_train.shape[1]
    for step in range(n_steps):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train[:, step])
        models.append(model)
    return models

def predict_rf_multi_step(models, X_test):
    preds = []
    for model in models:
        preds.append(model.predict(X_test))
    return np.stack(preds, axis=1)

def evaluate_and_plot_rf(models, X_test, y_test, scaler_close, date_test):
    n_steps = y_test.shape[1]
    preds = predict_rf_multi_step(models, X_test)
    # Inverse scaler chỉ cho cột 'Close'
    preds_inv = scaler_close.inverse_transform(preds)
    trues_inv = scaler_close.inverse_transform(y_test)
    metrics = []
    res_df_list = []
    for i in range(n_steps):
        rmse = np.sqrt(mean_squared_error(trues_inv[:, i], preds_inv[:, i]))
        mae = mean_absolute_error(trues_inv[:, i], preds_inv[:, i])
        mape = mean_absolute_percentage_error(trues_inv[:, i], preds_inv[:, i])
        r2 = r2_score(trues_inv[:, i], preds_inv[:, i])
        metrics.append([rmse, mae, mape, r2])
        step_df = pd.DataFrame({
            "Ngày": date_test,
            "Giá thực tế": trues_inv[:, i],
            "Giá dự báo": preds_inv[:, i],
            "Sai số tuyệt đối": np.abs(trues_inv[:, i] - preds_inv[:, i]),
            "Bước dự báo": f"t+{i+1}"
        })
        res_df_list.append(step_df)
    res_df = pd.concat(res_df_list, axis=0).reset_index(drop=True)
    return res_df, metrics

def run_rf(
    df=None,
    csv_path=None,
    look_back=30,
    pred_steps=7,
    n_estimators=100,
    max_depth=10,
    min_samples_split=5
):
    if df is None:
        if csv_path is None:
            raise ValueError("Cần truyền vào một trong hai: df hoặc csv_path")
        df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
    analyze_data(df, features)
    scaler_close = MinMaxScaler()
    df['Close'] = scaler_close.fit_transform(df[['Close']])
    df[features] = df[features]
    dates = df['Date'].reset_index(drop=True)

    values = df[features].values
    X, y = create_windows(values, look_back, pred_steps)
    X = X.reshape(X.shape[0], -1)  # RF cần input 2D (samples, features)
    # chia tay chuẩn
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    # cắt lại date_test cho phù hợp
    date_test = dates[n_train+n_val+look_back+pred_steps-1:].reset_index(drop=True)

    models = train_rf_multi_step(X_train, y_train, n_estimators, max_depth, min_samples_split)
    res_df, metrics = evaluate_and_plot_rf(models, X_test, y_test, scaler_close, date_test)
    return models, res_df, metrics

if __name__ == "__main__":
    models, res_df, metrics = run_rf(
        csv_path="AXISBANK.csv",
        look_back=30,
        pred_steps=7,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5
    )
    print(res_df.head())
    print(metrics)