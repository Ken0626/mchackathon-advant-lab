import numpy as np
import pandas as pd
from flask import Flask, jsonify
import threading
import time

# ================= 1. Flask API 設定區 ================= #
app = Flask(__name__)

# 這是要拋轉給心靈的全域狀態字典，確保她隨時抓到最新快照
dashboard_state = {
    "current_window": [],  # 讓心靈畫最新折線圖與散佈圖用
    "alerts": [],          # 觸發的警報 (陣列若為空代表正常)
    "limits": {            # 規格上下限 (讓心靈在圖表上畫紅線用)
        "high": 1.8, 
        "low": 0.6
    }
}

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    心靈的前端只要發送 GET 請求到這個網址，
    Flask 就會立刻把 dashboard_state 轉成 JSON 傳給她。
    """
    return jsonify(dashboard_state)

def run_flask():
    # 關閉除錯模式與重新載入，避免在執行緒中報錯
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ================= 2. 演算法核心 (維持不變) ================= #
class AnomalyDetector:
    def __init__(self, window_size=16):
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
        self.sliding_window.append({"site": site, "value": value})
        
        if len(self.sliding_window) > self.window_size:
            self.sliding_window.pop(0)
            
        if len(self.sliding_window) < self.window_size:
            return []
            
        df_window = pd.DataFrame(self.sliding_window)
        
        if self.baseline_std is None:
            self.baseline_std = df_window['value'].std()
            if self.baseline_std == 0:
                self.baseline_std = 0.01 
                
        decisions = [
            self.check_site_unbalance(df_window, param_name),
            self.check_trend(df_window, threshold=0.1),
            self.check_std_trend(df_window, std_multiplier_threshold=2.0),
            self.check_value_shift(df_window, shift_threshold=0.2)
        ]
        
        return [d for d in decisions if d["is_anomaly"]]

    def check_site_unbalance(self, df_window, param_name, std_multiplier=1.5):
        site_means = df_window.groupby('site')['value'].mean()
        overall_mean = df_window['value'].mean()
        overall_std = df_window['value'].std()
        for site, mean_val in site_means.items():
            if overall_std > 0 and abs(mean_val - overall_mean) > (std_multiplier * overall_std):
                return self.create_decision(True, "Site 2 Site Unbalance", "set_message", f"異常 Site: {site}")
        return self.create_decision(False)

    def check_trend(self, df_window, threshold=0.1):
        for site, group in df_window.groupby('site'):
            if len(group) < 3: continue
            y = group['value'].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            if abs(slope) > threshold:
                direction = "up" if slope > 0 else "down"
                return self.create_decision(True, f"Trend {direction}", "set_message", f"Site {site} 斜率達 {slope:.3f}")
        return self.create_decision(False)

    def check_std_trend(self, df_window, std_multiplier_threshold=2.0):
        for site, group in df_window.groupby('site'):
            current_std = group['value'].std()
            if pd.isna(current_std): continue
            if current_std > (self.baseline_std * std_multiplier_threshold):
                return self.create_decision(True, "Standard Deviation Trend Change", "set_message", f"Site {site} 標準差異常")
        return self.create_decision(False)

    def check_value_shift(self, df_window, shift_threshold=0.2):
        for site, group in df_window.groupby('site'):
            if len(group) < 4: continue
            mid_point = len(group) // 2
            diff = group['value'].iloc[mid_point:].mean() - group['value'].iloc[:mid_point].mean()
            if abs(diff) > shift_threshold:
                direction = "向上" if diff > 0 else "向下"
                return self.create_decision(True, "Measure Value Shift", "set_pause", f"Site {site} 發生 {direction} 偏移")
        return self.create_decision(False)


import json
import time
# (記得確認檔案最上方有 import json 與 import time)

# ... (中間的 AnomalyDetector 類別維持原樣，不需要動) ...

# ================= 3. 執行主程式 (B 計畫：讀檔案輪詢版) ================= #
if __name__ == "__main__":
    def simulate_oneapi_stream(csv_path):
        df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
        target_param = "220_Main.Suite1#CP"
        
        # 定義給心靈的共用狀態字典
        dashboard_state = {
            "current_window": [],  
            "alerts": [],          
            "limits": {            
                "high": 1.8, 
                "low": 0.6
            }
        }
        
        # 1. 正賢初始化你的檢測器
        detector = AnomalyDetector(window_size=16)
        print("⏳ 開始模擬機台生產數據，並即時覆寫 dashboard_status.json ...\n")
        
        # 2. 模擬 OneAPI 監聽到機台不斷送出新資料
        for index, row in df_data.iterrows():
            site = int(row["Site"])
            value = float(row[target_param])
            
            # 正賢呼叫你的 API，取得決策清單
            alerts = detector.process_new_data(site, value, target_param)
            
            # --- 更新全域字典 ---
            dashboard_state["current_window"] = detector.sliding_window
            dashboard_state["alerts"] = alerts
            
            # =======================================================
            # 🌟 放這裡！每次狀態一更新，就立刻覆寫存成實體 JSON 檔
            # =======================================================
            with open("dashboard_status.json", "w", encoding="utf-8") as f:
                json.dump(dashboard_state, f, ensure_ascii=False, indent=2)
            
            # 在終端機印出提示，讓你知道跑到哪了
            if alerts:
                print(f"[{index+1}] 🚨 觸發異常: {alerts[0]['anomaly_type']} (已寫入 JSON)")
            else:
                print(f"[{index+1}] ✅ 正常: Site {site} 測出 {value} (已寫入 JSON)")
            
            # 放慢迴圈速度，模擬真實機台運作節奏
            time.sleep(0.5) 

    simulate_oneapi_stream("data/example.csv")