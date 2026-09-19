import numpy as np
import pandas as pd

class AnomalyDetector:
    def __init__(self, window_size=16):
        """
        初始化檢測器，由它自己管理滑動窗口與狀態
        """
        self.window_size = window_size
        self.sliding_window = []
        self.baseline_std = None

    def create_decision(self, is_anomaly, anomaly_type="Normal", action="none", reason=""):
        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "action": action,
            "reason": reason
        }

    def process_new_data(self, site, value, param_name="220_Main.Suite1#CP"):
        """
        給正賢呼叫的唯一介面。每收到一筆資料，就呼叫一次這個函式。
        回傳值：一個 List，包含當下觸發的所有異常決策（若無異常則為空 List）
        """
        # 1. 將新資料推入狀態中
        self.sliding_window.append({"site": site, "value": value})
        
        # 2. 維持窗口大小
        if len(self.sliding_window) > self.window_size:
            self.sliding_window.pop(0)
            
        # 3. 若窗口未滿，直接回傳空陣列 (不檢查)
        if len(self.sliding_window) < self.window_size:
            return []
            
        # 4. 窗口滿載，開始轉換並檢查
        df_window = pd.DataFrame(self.sliding_window)
        
        if self.baseline_std is None:
            self.baseline_std = df_window['value'].std()
            if self.baseline_std == 0:
                self.baseline_std = 0.01 
                
        # 執行四大檢查
        decisions = [
            self.check_site_unbalance(df_window, param_name),
            self.check_trend(df_window, threshold=0.1),
            self.check_std_trend(df_window, std_multiplier_threshold=2.0),
            self.check_value_shift(df_window, shift_threshold=0.2)
        ]
        
        # 只回傳被觸發的異常
        return [d for d in decisions if d["is_anomaly"]]

    # ---------------- 內部檢測邏輯 (需加上 self) ---------------- #
    def check_site_unbalance(self, df_window, param_name, std_multiplier=1.5):
        site_means = df_window.groupby('site')['value'].mean()
        overall_mean = df_window['value'].mean()
        overall_std = df_window['value'].std()
        
        for site, mean_val in site_means.items():
            if overall_std > 0 and abs(mean_val - overall_mean) > (std_multiplier * overall_std):
                return self.create_decision(
                    is_anomaly=True, anomaly_type="Site 2 Site Unbalance",
                    action="set_message", 
                    reason=f"異常參數: {param_name}, 異常 Site: {site}, 該 Site 均值: {mean_val:.3f}, 整體均值: {overall_mean:.3f}"
                )
        return self.create_decision(is_anomaly=False)

    def check_trend(self, df_window, threshold=0.1):
        for site, group in df_window.groupby('site'):
            if len(group) < 3: continue
            y = group['value'].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            if abs(slope) > threshold:
                direction = "up" if slope > 0 else "down"
                return self.create_decision(
                    is_anomaly=True, anomaly_type=f"Trend {direction}",
                    action="set_message", reason=f"Site {site} 出現顯著 {direction} 趨勢，斜率達 {slope:.3f}"
                )
        return self.create_decision(is_anomaly=False)

    def check_std_trend(self, df_window, std_multiplier_threshold=2.0):
        for site, group in df_window.groupby('site'):
            current_std = group['value'].std()
            if pd.isna(current_std): continue
            if current_std > (self.baseline_std * std_multiplier_threshold):
                return self.create_decision(
                    is_anomaly=True, anomaly_type="Standard Deviation Trend Change",
                    action="set_message", reason=f"Site {site} 標準差異常放大：當前 {current_std:.3f}，基準 {self.baseline_std:.3f}"
                )
        return self.create_decision(is_anomaly=False)

    def check_value_shift(self, df_window, shift_threshold=0.2):
        for site, group in df_window.groupby('site'):
            if len(group) < 4: continue
            mid_point = len(group) // 2
            diff = group['value'].iloc[mid_point:].mean() - group['value'].iloc[:mid_point].mean()
            if abs(diff) > shift_threshold:
                direction = "向上" if diff > 0 else "向下"
                return self.create_decision(
                    is_anomaly=True, anomaly_type="Measure Value Shift",
                    action="set_pause", reason=f"Site {site} 發生 {direction} 斷崖式偏移，落差達 {abs(diff):.3f}"
                )
        return self.create_decision(is_anomaly=False)


# ---------------- 測試區塊 (模擬正賢在 sample.py 的用法) ---------------- #
if __name__ == "__main__":
    def simulate_oneapi_stream(csv_path):
        df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
        target_param = "220_Main.Suite1#CP"
        
        # 1. 正賢初始化你的檢測器
        detector = AnomalyDetector(window_size=16)
        
        # 2. 模擬 OneAPI 監聽到機台不斷送出新資料
        for index, row in df_data.iterrows():
            site = int(row["Site"])
            value = float(row[target_param])
            
            # 3. 正賢呼叫你的 API，並拿到決策清單
            alerts = detector.process_new_data(site, value, target_param)
            
            if alerts:
                print(f"========== 收到 OneAPI 事件 (第 {index + 1} 筆, Site {site}) ==========")
                for alert in alerts:
                    print(f"🚨 呼叫 ActionManager.{alert['action']} -> {alert['reason']}")
                print("-" * 60 + "\n")

    simulate_oneapi_stream("data/example.csv")