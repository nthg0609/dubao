import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def create_windows(data, look_back, pred_steps, univariate=True):
    X, y = [], []
    if univariate:
        data_close = data[:, 3].reshape(-1, 1)  # chỉ lấy cột Close
        for i in range(len(data_close) - look_back - pred_steps + 1):
            X.append(data_close[i:i+look_back])
            y.append(data_close[i+look_back:i+look_back+pred_steps, 0])
    else:
        for i in range(len(data) - look_back - pred_steps + 1):
            X.append(data[i:i+look_back, :])
            y.append(data[i+look_back:i+look_back+pred_steps, 3])  # vẫn lấy giá trị Close cho nhãn
    return np.array(X), np.array(y)

def run_xgb(
    df,
    univariate=True,
    look_back=30,
    pred_steps=7,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
    feature_cols = ['Close'] if univariate else features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    values = df[features].values

    X, y = create_windows(values, look_back, pred_steps, univariate=univariate)

    # --- Kiểm tra lỗi để tránh crash ---
    if X.shape[0] == 0 or y.shape[0] == 0:
        empty_df = pd.DataFrame(columns=["Ngày", "Giá thực tế", "Giá dự báo", "Sai số tuyệt đối", "Bước dự báo"])
        return [], empty_df, []

    X = X.reshape(X.shape[0], -1)  # XGB yêu cầu 2D
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    dates = df['Date'].reset_index(drop=True)
    date_test = dates[n_train+n_val+look_back+pred_steps-1:].reset_index(drop=True)

    models = []
    for i in range(pred_steps):
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train[:, i])
        models.append(model)

    preds = np.column_stack([m.predict(X_test) for m in models])
    # Chuẩn hóa ngược
    n_cols = len(features)
    close_idx = 3
    temp_pred = np.zeros((preds.shape[0], n_cols))
    temp_true = np.zeros((y_test.shape[0], n_cols))
    metrics = []
    res_df_list = []
    for i in range(pred_steps):
        temp_pred[:, close_idx] = preds[:, i]
        pred_inv = scaler.inverse_transform(temp_pred)[:, close_idx]
        temp_true[:, close_idx] = y_test[:, i]
        actual_inv = scaler.inverse_transform(temp_true)[:, close_idx]

        rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
        mae = mean_absolute_error(actual_inv, pred_inv)
        mape = mean_absolute_percentage_error(actual_inv, pred_inv)
        r2 = r2_score(actual_inv, pred_inv)
        metrics.append([rmse, mae, mape, r2])
        valid_len = len(actual_inv)
        step_df = pd.DataFrame({
            "Ngày": date_test.iloc[:valid_len].reset_index(drop=True),
            "Giá thực tế": actual_inv,
            "Giá dự báo": pred_inv,
            "Sai số tuyệt đối": np.abs(actual_inv - pred_inv),
            "Bước dự báo": f"t+{i+1}"
        })
        res_df_list.append(step_df)
    res_df = pd.concat(res_df_list, axis=0).reset_index(drop=True)
    return models, res_df, metrics