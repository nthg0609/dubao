import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

class ARIMAModel:
    def __init__(self, p=1, d=1, q=1, window_size=30):
        self.p = p
        self.d = d
        self.q = q
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.forecast_type = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def analyze_data(self, df, features):
        desc = df[features].describe().T
        desc = desc[['count','mean','std','min','25%','50%','75%','max']]
        desc.columns = [
            "Số lượng quan sát", "Giá trị trung bình", "Độ lệch chuẩn",
            "Giá trị nhỏ nhất", "Phân vị 25%", "Trung vị (50%)",
            "Phân vị 75%", "Giá trị lớn nhất"
        ]
        # Biểu đồ heatmap và hist cần trả về Figure
        fig_corr, ax_corr = plt.subplots(figsize=(9,6))
        sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Ma trận tương quan')
        hist_figs = []
        for col in features:
            fig_hist, ax_hist = plt.subplots(figsize=(5,2))
            sns.histplot(df[col], kde=True, bins=40, ax=ax_hist)
            ax_hist.set_title(f'Phân phối {col}')
            hist_figs.append(fig_hist)
        return desc, df[features].isnull().sum(), fig_corr, hist_figs

    def preprocess_data(self, df, univariate=True, features=None):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') 
        df = df.sort_values(by='Date')
        # Chỉ lấy các cột phù hợp
        if univariate:
            # Fit scaler với toàn bộ 'Close'
            self.scaler.fit(df[['Close']])  # SỬA Ở ĐÂY!
            scaled = self.scaler.transform(df[['Close']])
            return scaled.flatten(), df['Date'], None
        else:
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].dropna()
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
            scaled_all = {}
            # Fit riêng từng scaler cho từng feature (nếu cần cho multivariate)
            for feat in features:
                scaler = MinMaxScaler()
                scaler.fit(df[[feat]])
                scaled_all[feat] = scaler.transform(df[[feat]]).flatten()
                if feat == 'Close':
                    self.scaler = scaler  # Đảm bảo self.scaler đã fit với 'Close'
            return scaled_all, df['Date'], features
    def split_data(self, series, ratio=(0.8, 0.1)):
        N = len(series)
        n_train = int(N * ratio[0])
        n_val = int(N * ratio[1])
        train_idx = (0, n_train)
        val_idx = (n_train, n_train+n_val)
        test_idx = (n_train+n_val, N)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        return train_idx, val_idx, test_idx

    def naive_forecast(self, series, n_test, pred_steps):
        preds = []
        for i in range(n_test):
            preds.append([series[-n_test+i-1]]*pred_steps)
        return np.array(preds)

    def train(self, df, window_size=30, pred_steps=7, univariate=True, features=None):
        self.forecast_type = "Đơn biến" if univariate else "Đa biến"
        self.window_size = window_size
        self.pred_steps = pred_steps
        desc, missing, fig_corr, hist_figs = self.analyze_data(df, features or ['Open','High','Low','Close','Volume','VWAP'])
        if univariate:
            series, dates, _ = self.preprocess_data(df, univariate=True)
            idx_train, idx_val, idx_test = self.split_data(series)
            train_series = series[idx_train[0]:idx_val[1]]
            self.model = pm.auto_arima(
                train_series,
                start_p=self.p, start_q=self.q, start_d=self.d,
                max_p=5, max_d=2, max_q=5,
                seasonal=False,
                stepwise=True, error_action="ignore", suppress_warnings=True
            )
            self.test_series = series[idx_test[0]:idx_test[1]]
            self.test_dates = dates.iloc[idx_test[0]:idx_test[1]].reset_index(drop=True)
            self.train_series = train_series
        else:
            scaled_dict, dates, features = self.preprocess_data(df, univariate=False, features=features)
            idx_train, idx_val, idx_test = self.split_data(scaled_dict['Close'])
            train_series = scaled_dict['Close'][idx_train[0]:idx_val[1]]
            self.model = pm.auto_arima(
                train_series,
                start_p=self.p, start_q=self.q, start_d=self.d,
                max_p=5, max_d=2, max_q=5,
                seasonal=False,
                stepwise=True, error_action="ignore", suppress_warnings=True
            )
            self.test_series = scaled_dict['Close'][idx_test[0]:idx_test[1]]
            self.test_dates = dates.iloc[idx_test[0]:idx_test[1]].reset_index(drop=True)
            self.train_series = train_series
        return desc, missing, fig_corr, hist_figs

    def predict(self):
        y_preds, y_trues, y_dates = [], [], []
        test_series = self.test_series
        win, steps = self.window_size, self.pred_steps
        n_sample = len(test_series) - win - steps + 1
        for i in range(n_sample):
            history = test_series[i:i+win]
            true_vals = test_series[i+win:i+win+steps]
            model = pm.ARIMA(order=self.model.order)
            model.fit(history)
            pred = model.predict(n_periods=steps)
            y_preds.append(pred)
            y_trues.append(true_vals)
            y_dates.append(self.test_dates[i+win+steps-1])
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        y_preds_inv = self.scaler.inverse_transform(y_preds)
        y_trues_inv = self.scaler.inverse_transform(y_trues)
        return y_preds_inv, y_trues_inv, y_dates

    def evaluate(self, y_preds_inv, y_trues_inv):
        pred_steps = y_preds_inv.shape[1]
        metrics = []
        for i in range(pred_steps):
            rmse = np.sqrt(mean_squared_error(y_trues_inv[:,i], y_preds_inv[:,i]))
            mae = mean_absolute_error(y_trues_inv[:,i], y_preds_inv[:,i])
            mape = mean_absolute_percentage_error(y_trues_inv[:,i], y_preds_inv[:,i])
            r2 = r2_score(y_trues_inv[:,i], y_preds_inv[:,i])
            metrics.append([rmse, mae, mape, r2])
        return metrics

    def plot_results(self, y_preds_inv, y_trues_inv, y_dates):
        pred_steps = y_preds_inv.shape[1]
        figs = []
        for step in range(pred_steps):
            fig, ax = plt.subplots(figsize=(11,4))
            ax.plot(range(len(y_trues_inv)), y_trues_inv[:,step], label=f'Thực tế t+{step+1}')
            ax.plot(range(len(y_preds_inv)), y_preds_inv[:,step], label=f'Dự báo t+{step+1}')
            ax.set_title(f"Biểu đồ dự báo ARIMA bước t+{step+1}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Giá đóng cửa")
            ax.legend()
            figs.append(fig)
        fig_scatter, ax_scatter = plt.subplots(figsize=(5,5))
        ax_scatter.scatter(y_trues_inv.flatten(), y_preds_inv.flatten(), alpha=0.5)
        ax_scatter.set_xlabel("Giá thực tế")
        ax_scatter.set_ylabel("Giá dự báo")
        ax_scatter.set_title("Scatter: Giá thực tế vs Dự báo (all steps)")
        figs.append(fig_scatter)
        fig_hist, ax_hist = plt.subplots(figsize=(7,3))
        ax_hist.hist((y_trues_inv-y_preds_inv).flatten(), bins=40, alpha=0.7)
        ax_hist.set_title("Histogram sai số (thực tế - dự báo)")
        ax_hist.set_xlabel("Error")
        figs.append(fig_hist)
        return figs

    def summary_table(self, y_preds_inv, y_trues_inv, y_dates):
        pred_steps = y_preds_inv.shape[1]
        res_df = pd.DataFrame({
            "Ngày": [d for d in y_dates for _ in range(pred_steps)],
            "Step": [f"t+{i+1}" for _ in y_dates for i in range(pred_steps)],
            "Giá thực tế": y_trues_inv.flatten(),
            "Giá dự báo": y_preds_inv.flatten(),
            "Sai số tuyệt đối": np.abs(y_trues_inv.flatten()-y_preds_inv.flatten())
        })
        return res_df

    def compare_baseline(self, y_trues_inv):
        n = len(y_trues_inv)
        pred_steps = y_trues_inv.shape[1]
        naive = np.repeat(y_trues_inv[:,0][:,None], pred_steps, axis=1)
        baseline = []
        for i in range(pred_steps):
            rmse = np.sqrt(mean_squared_error(y_trues_inv[:,i], naive[:,i]))
            baseline.append(rmse)
        return baseline

    # ========= Hàm gọi tổng thể (gắn cho UI hoặc CLI) ===========
    def run(
        self,
        df=None,
        csv_path=None,
        univariate=True,
        window_size=30,
        pred_steps=7,
        features=None
    ):
        if df is None:
            if csv_path is not None:
                df = pd.read_csv(csv_path)
            else:
                raise ValueError("Bạn phải truyền vào df hoặc csv_path!")
        desc, missing, fig_corr, hist_figs = self.train(
            df, window_size=window_size, pred_steps=pred_steps,
            univariate=univariate, features=features
        )
        y_preds_inv, y_trues_inv, y_dates = self.predict()
        metrics = self.evaluate(y_preds_inv, y_trues_inv)
        res_df = self.summary_table(y_preds_inv, y_trues_inv, y_dates)
        figs = self.plot_results(y_preds_inv, y_trues_inv, y_dates)
        baseline = self.compare_baseline(y_trues_inv)
        # Trả về mọi thứ cần show ở Streamlit:
        return {
            "desc": desc,
            "missing": missing,
            "fig_corr": fig_corr,
            "hist_figs": hist_figs,
            "res_df": res_df,
            "metrics": metrics,
            "figs": figs,
            "baseline": baseline
        }

# ========== Chạy demo file hoặc import cho giao diện ==========
if __name__ == "__main__":
    arima = ARIMAModel()
    result = arima.run(
        csv_path="AXISBANK.csv",
        univariate=True,        # hoặc False nếu muốn thử đa biến
        window_size=30,
        pred_steps=7,
        features=['Open','High','Low','Close','Volume','VWAP']
    )
    print(result["desc"])
    print(result["missing"])
    print(result["res_df"].head())
    print(result["metrics"])
    print("Baseline RMSE (naive):", result["baseline"])