import os
import numpy as np
import pandas as pd
import joblib
import json
import time

class AnomalyDetector:
    def __init__(self, window_size=16):
        self.window_size = window_size
        # 結構優化：依 Site 獨立維護歷史數據 (針對 Site 1~4)
        self.history = {1: [], 2: [], 3: [], 4: []}
        self.baseline_std = None
        
        # 預留載入場景一與場景二的 ML 模型擴充點
        try:
            self.iso_forest = joblib.load("models/isolation_forest.pkl")
            self.temp_model = joblib.load("models/temperature_predictor.pkl")
        except FileNotFoundError:
            self.iso_forest = None
            self.temp_model = None

    def create_decision(self, is_anomaly, anomaly_type="Normal", action="none", reason=""):
        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "action": action,
            "reason": reason
        }

    # ================= 場景一：異常偵測 (被動接收測試數據) ================= #
    def process_new_data(self, site, value, param_name, high_limit, low_limit):
        # 1. 更新單一 Site 的滑動窗口
        self.history[site].append(value)
        if len(self.history[site]) > self.window_size:
            self.history[site].pop(0)
            
        # 確保所有 Site 都有足夠的資料才開始進行分析
        if any(len(v) < self.window_size for v in self.history.values()):
            return []
            
        # 建立 DataFrame，欄位為 Site 1, 2, 3, 4
        df_window = pd.DataFrame(self.history)
        
        if self.baseline_std is None:
            self.baseline_std = df_window.values.std()
            if self.baseline_std == 0:
                self.baseline_std = 0.01 
                
        # 2. Rule-based 保底防禦
        decisions = [
            self.check_site_unbalance(df_window, param_name),
            self.check_trend(df_window, threshold=0.1),
            self.check_std_trend(df_window, std_multiplier_threshold=2.0),
            self.check_value_shift(df_window, shift_threshold=0.2)
        ]
        
        # 3. ML 模型推論 (Isolation Forest 或 Autoencoder)
        if self.iso_forest is not None:
            # 萃取 4 個 Site 最新的一筆資料作為特徵
            latest_features = [self.history[s][-1] for s in [1, 2, 3, 4]]
            prediction = self.iso_forest.predict([latest_features])
            if prediction[0] == -1: 
                decisions.append(self.create_decision(True, "ML Anomaly Detected", "set_message", "機器學習模型判定異常"))

        return [d for d in decisions if d["is_anomaly"]]

    # ----------------- 傳統統計規則實作 ----------------- #
    def check_site_unbalance(self, df_window, param_name, std_multiplier=1.5):
        site_means = df_window.mean() 
        overall_mean = df_window.values.mean()
        overall_std = df_window.values.std()
        
        for site, mean_val in site_means.items():
            if overall_std > 0 and abs(mean_val - overall_mean) > (std_multiplier * overall_std):
                return self.create_decision(True, "Site 2 Site Unbalance", "set_message", f"異常 Site: {site}")
        return self.create_decision(False)

    def check_trend(self, df_window, threshold=0.1):
        if len(df_window) < 3: 
            return self.create_decision(False)
            
        x = np.arange(len(df_window))
        for site in df_window.columns:
            y = df_window[site].values
            slope, _ = np.polyfit(x, y, 1)
            if abs(slope) > threshold:
                direction = "up" if slope > 0 else "down"
                return self.create_decision(True, f"Trend {direction}", "set_message", f"Site {site} 斜率達 {slope:.3f}")
        return self.create_decision(False)

    def check_std_trend(self, df_window, std_multiplier_threshold=2.0):
        if self.baseline_std is None: 
            return self.create_decision(False)
            
        site_stds = df_window.std()
        for site, current_std in site_stds.items():
            if pd.isna(current_std): continue
            if current_std > (self.baseline_std * std_multiplier_threshold):
                return self.create_decision(True, "Standard Deviation Trend Change", "set_message", f"Site {site} 標準差異常")
        return self.create_decision(False)

    def check_value_shift(self, df_window, shift_threshold=0.2):
        if len(df_window) < 4: 
            return self.create_decision(False)
            
        mid_point = len(df_window) // 2
        for site in df_window.columns:
            diff = df_window[site].iloc[mid_point:].mean() - df_window[site].iloc[:mid_point].mean()
            if abs(diff) > shift_threshold:
                direction = "向上" if diff > 0 else "向下"
                return self.create_decision(True, "Measure Value Shift", "set_pause", f"Site {site} 發生 {direction} 偏移")
        return self.create_decision(False)

    # ================= 場景二：溫度預測 (主動攔截請求) ================= #
    def predict_temperature(self, target_sensor_num):
        """
        處理機台發送的 consumeTPRequest，預測指定 sensor 的溫度並回傳字串。
        """
        if self.temp_model is None:
            return "ML Model not loaded"
            
        features = self._extract_features_for_sensor(target_sensor_num)
        predictions = self.temp_model.predict(features) 
        
        message = ""
        for site, val in enumerate(predictions, start=1):
            message += f"site[{site}]: {val:.2f} "
            
        return message
        
    def _extract_features_for_sensor(self, target_sensor_num):
        """
        從 self.history 中萃取預測特徵，絕對不可取用大於 target_sensor_num 的未來數據。
        """
        return np.zeros((4, 10)) 


# ================= 執行主程式 (測試與產生 JSON 檔) ================= #
if __name__ == "__main__":
    def simulate_oneapi_stream(base_dir="data/Data"):
        # 定義給前端的共用狀態字典
        dashboard_state = {
            "current_window": [],  
            "alerts": [],          
            "limits": {            
                "high": 1.8, 
                "low": 0.6
            }
        }
        
        detector = AnomalyDetector(window_size=16)
        print("⏳ 開始模擬機台生產數據，並即時覆寫 dashboard_status.json ...\n")
        
        target_param = "220_Main.Suite1#CP"

        # 遍歷 1 到 25 的真實 Wafer 資料
        for i in range(1, 26):
            file_name = f"A12345_W{i:02d}_RawResult.csv"
            csv_path = os.path.join(base_dir, file_name)

            try:
                df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
                print(f"\n📂 成功載入 Wafer: {file_name}，準備進行模擬。")
            except FileNotFoundError:
                print(f"\n⚠️ 找不到測試檔案 {csv_path}，跳過此 Wafer。")
                continue
            
            # 模擬 OneAPI 持續拋出資料
            for index, row in df_data.iterrows():
                # 1. 防呆：跳過 Site 為空值 (NaN) 的髒數據
                if pd.isna(row.get("Site")):
                    continue
                
                site = int(row["Site"])
                
                # 2. 防呆：跳過目標測試項不存在或為空值的數據
                if target_param not in row or pd.isna(row[target_param]):
                    continue
                    
                value = float(row[target_param])
                
                # 在真實環境中，此處 limits 會由 OneAPI 即時獲取
                current_high_limit = 1.8
                current_low_limit = 0.6
                
                # 1. 執行偵測
                alerts = detector.process_new_data(site, value, target_param, current_high_limit, current_low_limit)
                
                # 2. 更新全域字典 (轉換為扁平格式供前端讀圖)
                flattened_window = []
                for s, vals in detector.history.items():
                    for v in vals:
                        flattened_window.append({"site": s, "value": v})
                        
                dashboard_state["current_window"] = flattened_window
                dashboard_state["alerts"] = alerts
                dashboard_state["limits"]["high"] = current_high_limit
                dashboard_state["limits"]["low"] = current_low_limit
                
                # 3. 寫出實體 JSON 供 File Polling 讀取
                with open("dashboard_status.json", "w", encoding="utf-8") as f:
                    json.dump(dashboard_state, f, ensure_ascii=False, indent=2)
                
                # 4. 終端機模擬 Log 輸出 (同一列覆寫，避免 25 片洗版)
                if alerts:
                    print(f"\n[{file_name} - {index+1}] 🚨 觸發異常: {alerts[0]['anomaly_type']}")
                else:
                    print(f"[{file_name} - {index+1}] ✅ 正常: Site {site} 測出 {value:.3f}", end="\r")
                
                # 微量延遲確保前端拿得到檔案，但不會因 25 片 Wafer 而跑太久
                time.sleep(0.01)

    # 執行模擬
    simulate_oneapi_stream()