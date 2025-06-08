import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import os

# --- Import các model đúng chuẩn interface (theo file bạn gửi) ---
from models.lstm_model import run_lstm_model
from models.tcn_model import run_tcn
from models.transformer_model import run_transformer
from models.hybrid_model import run_hybrid
from models.informer_model import run_informer
from models.autoformer_model import run_autoformer
from models.arima_model import ARIMAModel
# from models.svr_model import run_svr, train_svr_multi_step, predict_svr_multi_step
from models.rf_model import train_rf_multi_step, predict_rf_multi_step
from models.rf_model import run_rf
from models.xgb_model import run_xgb

# ---- CONFIG STREAMLIT ----
st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)
st.title("📈 Dự báo giá cổ phiếu với các mô hình Machine Learning & Deep Learning hiện đại")

# ---- SIDEBAR: UPLOAD VÀ CHỌN MODEL ----
st.sidebar.header("Cấu hình")
uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu", type=['csv'])

# ==== XỬ LÝ DỮ LIỆU ====
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].dropna()
    st.subheader("🔎 Xem dữ liệu đầu vào")
    st.dataframe(df.head())
    # ==== THÊM ĐOẠN NÀY ====
    unique_dates = df["Date"].dt.strftime("%Y-%m-%d").unique()
    st.sidebar.markdown("### Chọn khoảng ngày dự báo")
    start_date = st.sidebar.selectbox("Từ ngày", options=unique_dates, index=0)
    end_date = st.sidebar.selectbox("Đến ngày", options=unique_dates, index=len(unique_dates)-1)

    start_date_obj = pd.to_datetime(start_date)
    end_date_obj = pd.to_datetime(end_date)

    # ==== ĐOẠN SỬA 2: KIỂM TRA SỐ NGÀY TỐI THIỂU ====
    # Các giá trị mặc định cho look_back và pred_steps (lấy ở đây để check)
    look_back = st.sidebar.slider("Window size", 10, 60, 30, key="look_back_check")
    pred_steps = st.sidebar.slider("Số bước dự báo", 1, 14, 7, key="pred_steps_check")
    min_days = look_back + pred_steps - 1
    num_days = (end_date_obj - start_date_obj).days + 1
    if num_days < min_days:
        st.warning(f"Khoảng ngày bạn chọn phải có ít nhất {min_days} ngày (Window size = {look_back}, Số bước dự báo = {pred_steps})")
        st.stop()

    # Lọc lại dữ liệu theo ngày
    df_filtered = df[(df["Date"] >= start_date_obj) & (df["Date"] <= end_date_obj)]
    st.subheader("📊 Phân tích thống kê dữ liệu")
    st.write(df.describe())
    st.write("Missing values:", df.isnull().sum())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # ==== SIDEBAR: CONFIG MODEL ====
    st.sidebar.subheader("Cấu hình dự báo")
    model_type = st.sidebar.selectbox(
        "Chọn mô hình", [
            "LSTM", "TCN", "Transformer", "Hybrid", "Informer", "Autoformer", "ARIMA", "Random Forest", "XGBoost"
        ]
    )
    forecast_type = st.sidebar.selectbox(
        "Kiểu dự báo", [
            "Đơn biến (Univariate)", "Đa biến (Multivariate)"
        ]
    )
    look_back = st.sidebar.slider("Window size", 10, 60, 30)
    pred_steps = st.sidebar.slider("Số bước dự báo", 1, 14, 7)

    # ==== CẤU HÌNH RIÊNG TỪNG MODEL ====
    if model_type == "LSTM":
        hidden_size = st.sidebar.slider("Hidden size", 32, 256, 64)
        num_layers = st.sidebar.slider("Số lớp LSTM", 1, 4, 2)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)
    elif model_type == "TCN":
        num_channels = st.sidebar.text_input("Số channel (vd: 64,32,16)", "64,32,16")
        num_channels = [int(i) for i in num_channels.split(',')]
        kernel_size = st.sidebar.slider("Kernel size", 2, 8, 3)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)
    elif model_type == "Transformer":
        d_model = st.sidebar.slider("d_model", 32, 256, 64)
        nhead = st.sidebar.slider("Số head", 1, 16, 8)
        num_layers = st.sidebar.slider("Số lớp", 1, 6, 2)
        dim_feedforward = st.sidebar.slider("Feedforward dim", 32, 512, 256)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)
    elif model_type == "Hybrid":
        hidden_size = st.sidebar.slider("Hidden size", 32, 256, 64)
        num_layers = st.sidebar.slider("Số lớp", 1, 4, 2)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)
    elif model_type == "Informer":
        d_model = st.sidebar.slider("d_model", 32, 256, 64)
        n_layers = st.sidebar.slider("Số lớp", 1, 6, 2)
        d_ff = st.sidebar.slider("d_ff", 64, 512, 256)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)
    elif model_type == "Autoformer":
        d_model = st.sidebar.slider("d_model", 32, 256, 64)
        n_layers = st.sidebar.slider("Số lớp", 1, 6, 2)
        d_ff = st.sidebar.slider("d_ff", 64, 512, 256)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)
    elif model_type == "ARIMA":
        p = st.sidebar.slider("p", 0, 5, 1)
        d = st.sidebar.slider("d", 0, 2, 1)
        q = st.sidebar.slider("q", 0, 5, 1)
    elif model_type == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
        max_depth = st.sidebar.slider("max_depth", 3, 30, 10)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 5)
    # elif model_type == "SVR":
    #     kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    #     C = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    #     epsilon = st.sidebar.slider("epsilon", 0.01, 0.5, 0.1)

    elif model_type == "XGBoost":
        n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
        max_depth = st.sidebar.slider("max_depth", 2, 16, 6)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1)
        subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8)

    # ==== PHẦN CHẠY MODEL THEO LỰA CHỌN ====
    st.sidebar.markdown("---")
    st.sidebar.write("### Huấn luyện mô hình")
    if st.sidebar.button("Bắt đầu huấn luyện và dự báo"):
        with st.spinner("Đang xử lý và huấn luyện mô hình..."):
            # === Univariate/multivariate config ===
            univariate = (forecast_type == "Đơn biến (Univariate)")
            features = ['Open','High','Low','Close','Volume','VWAP']
            res_df, metrics = None, None  # Khởi tạo giá trị mặc định
            if model_type == "LSTM":
                _, res_df, metrics = run_lstm_model(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, plot=False
                )
            elif model_type == "TCN":
                _, res_df, metrics = run_tcn(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    num_channels=num_channels, kernel_size=kernel_size, dropout=dropout,
                    batch_size=32, epochs=50, lr=0.001
                )
            elif model_type == "Transformer":
                _, res_df, metrics = run_transformer(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                    dim_feedforward=dim_feedforward, dropout=dropout,
                    batch_size=32, epochs=50, lr=0.001
                )
            elif model_type == "Hybrid":
                _, res_df, metrics = run_hybrid(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                    batch_size=32, epochs=50, lr=0.001
                )
            elif model_type == "Informer":
                _, res_df, metrics = run_informer(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    d_model=d_model, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
                    batch_size=32, epochs=50, lr=0.001
                )
            elif model_type == "Autoformer":
                _, res_df, metrics = run_autoformer(
                    df=df_filtered, univariate=univariate, look_back=look_back, pred_steps=pred_steps,
                    d_model=d_model, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
                    batch_size=32, epochs=50, lr=0.001
                )
            elif model_type == "ARIMA":
                arima = ARIMAModel(p=p, d=d, q=q, window_size=look_back)
                arima_result = arima.run(
                    df=df_filtered, univariate=univariate, window_size=look_back,
                    pred_steps=pred_steps, features=features
                )
                res_df = arima_result["res_df"]
                metrics = arima_result["metrics"]
                # Show bảng thống kê, heatmap, histogram nếu muốn:
                st.subheader("Thống kê mô tả dữ liệu")
                st.dataframe(arima_result["desc"])
                st.write("Missing values:", arima_result["missing"])
                st.pyplot(arima_result["fig_corr"])
                for fig in arima_result["hist_figs"]:
                    st.pyplot(fig)
                for fig in arima_result["figs"]:
                    st.pyplot(fig)
                st.write("Baseline RMSE (Naive):", arima_result["baseline"])
            # elif model_type == "SVR":
            # #     _, res_df, metrics = run_svr(
            # #         df=df, look_back=look_back, pred_steps=pred_steps
            #     )
            elif model_type == "Random Forest":
                _, res_df, metrics = run_rf(
                    df=df_filtered,
                    look_back=look_back,
                    pred_steps=pred_steps,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
            
            elif model_type == "XGBoost":
                _, res_df, metrics = run_xgb(
                    df=df_filtered,
                    univariate=univariate,
                    look_back=look_back,
                    pred_steps=pred_steps,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8
                )

            st.success("✅ Huấn luyện và dự báo hoàn tất.")
            if metrics is not None:
                st.subheader("=== KẾT QUẢ ĐÁNH GIÁ MULTI-STEP ===")
                for i, (rmse, mae, mape, r2) in enumerate(metrics):
                    st.write(
                        f"Bước t+{i+1}: "
                        f"RMSE={rmse:.2f} | "
                        f"MAE={mae:.2f} | "
                        f"MAPE={mape:.2%} | "
                        f"R2={r2:.4f}"
                    )
            # ==== ĐOẠN SỬA 3: HIỂN THỊ TOÀN BỘ KẾT QUẢ VÀ BIỂU ĐỒ ====
            if res_df is not None:
                # Chuyển 'Ngày' sang datetime nếu cần
                if not np.issubdtype(res_df['Ngày'].dtype, np.datetime64):
                    res_df['Ngày'] = pd.to_datetime(res_df['Ngày'])
                # Lọc đúng khoảng ngày đã chọn (dù model đã dự báo đúng khoảng này)
                mask = (res_df['Ngày'] >= start_date_obj) & (res_df['Ngày'] <= end_date_obj)
                show_df = res_df[mask] if mask.any() else res_df

                st.markdown("#### == Bảng kết quả dự báo:")
                st.dataframe(show_df)

                # Biểu đồ thực tế & dự báo
                st.markdown("#### == Biểu đồ giá thực tế và giá dự báo:")
                fig, ax = plt.subplots(figsize=(12,5))
                ax.plot(show_df['Ngày'], show_df['Giá thực tế'], label="Giá thực tế", marker='o')
                ax.plot(show_df['Ngày'], show_df['Giá dự báo'], label="Giá dự báo", marker='o')
                ax.set_xlabel("Ngày")
                ax.set_ylabel("Giá")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)

else:
    st.info("Vui lòng tải lên file dữ liệu .csv để bắt đầu.")