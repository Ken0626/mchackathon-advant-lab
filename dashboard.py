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
    def __init__(self, window_size=16, model_path=None):
        self.window_size = window_size
        self.sliding_window = []
        self.baseline_std = None
        
        # 🌟 ML 核心：在這裡載入你預先訓練好的模型
        self.ml_model = None
        if model_path:
            print(f"正在載入 ML 模型: {model_path}")
            # [PyTorch 範例] self.ml_model = torch.load(model_path)
            # [XGBoost/Sklearn 範例] self.ml_model = joblib.load(model_path)
            
            # 這裡我們先放一個 Dummy Flag，代表模型已準備就緒
            self.ml_model = "Model_Loaded"

    def create_decision(self, is_anomaly, anomaly_type="Normal", action="none", reason=""):
        return {"is_anomaly": is_anomaly, "anomaly_type": anomaly_type, "action": action, "reason": reason}

    def process_new_data(self, site, value, limits, param_name="220_Main.Suite1#CP"):
        self.sliding_window.append({"site": site, "value": value})
        if len(self.sliding_window) > self.window_size:
            self.sliding_window.pop(0)
        if len(self.sliding_window) < self.window_size:
            return []
            
        df_window = pd.DataFrame(self.sliding_window)
        if self.baseline_std is None:
            self.baseline_std = df_window['value'].std() or 0.01 
                
        # 1. 傳統的規則防呆 (Rule-based) 依然保留，當作第一道防線
        decisions = [
            self.check_site_unbalance(df_window, param_name),
            self.check_value_shift(df_window, shift_threshold=0.2)
        ]
        
        # 2. 🌟 呼叫 ML 模型進行深度特徵推論
        if self.ml_model:
            ml_decision = self.predict_with_ml(df_window)
            decisions.append(ml_decision)
            
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

    def predict_with_ml(self, df_window):
        """
        將滑動窗口的資料轉換成特徵 (Feature Engineering)，
        然後餵給 ML 模型 (Autoencoder / XGBoost / Isolation Forest) 進行異常預測。
        """
        # 步驟 1: 特徵工程 (Feature Extraction)
        # 模型通常不吃 raw data，而是吃特徵。這裡我們把 16 筆資料轉換成一個特徵向量。
        features = []
        for site, group in df_window.groupby('site'):
            if len(group) >= 3:
                y = group['value'].values
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
                
                # 萃取該 Site 的特徵：最新值、平均、標準差、斜率
                features.extend([
                    y[-1],                # Current Value
                    group['value'].mean(),# Mean
                    group['value'].std(), # Std Dev
                    slope                 # Trend Slope
                ])
                
        # 如果資料不足以產生完整特徵，先跳過推論
        if len(features) < 16:  # 4 個 Site * 4 個特徵 = 16 維向量
            return self.create_decision(False)

        # 步驟 2: 執行模型推論
        # 轉換成模型吃得下的格式 (例如 numpy array 或 PyTorch Tensor)
        # X_input = np.array([features])
        # [PyTorch 範例] 
        # tensor_input = torch.FloatTensor(X_input)
        # reconstruction = self.ml_model(tensor_input)
        # loss = torch.nn.functional.mse_loss(reconstruction, tensor_input)
        # is_anomaly = loss.item() > 0.05 (你的報警閾值)
        
        # [Isolation Forest / XGBoost 範例]
        # prediction = self.ml_model.predict(X_input)
        # is_anomaly = (prediction[0] == -1)  # -1 通常代表異常
        
        # ⚠️ 這裡先寫死一個隨機模擬的判定，等你真的把模型訓練好再把上面的註解打開替換掉！
        mock_ml_anomaly_score = np.random.rand() 
        is_ml_anomaly = mock_ml_anomaly_score > 0.95 # 有 5% 機率觸發

        if is_ml_anomaly:
            return self.create_decision(
                is_anomaly=True, 
                anomaly_type="ML Model Detection (預知異常)", 
                action="set_message", 
                reason=f"ML 模型偵測到深層特徵異常，異常分數達 {mock_ml_anomaly_score:.3f}"
            )
            
        return self.create_decision(False)


# ================= 3. 執行主程式 (模擬與正賢、心靈介接) ================= #
if __name__ == "__main__":
    def simulate_oneapi_stream(csv_path):
        df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
        target_param = "220_Main.Suite1#CP"
        
        # 1. 啟動 Flask 背景執行緒，負責對外發送 JSON
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("🌐 Dashboard API 已啟動！心靈可以連線至: http://127.0.0.1:5000/api/status")
        print("⏳ 開始模擬機台生產數據，每 0.5 秒進件一次...\n")
        
        # 2. 正賢初始化你的檢測器
        detector = AnomalyDetector(window_size=16)
        
        # 3. 模擬 OneAPI 監聽到機台不斷送出新資料
        for index, row in df_data.iterrows():
            site = int(row["Site"])
            value = float(row[target_param])
            
            # 取得決策清單
            alerts = detector.process_new_data(site, value, target_param)
            
            # --- 關鍵：將最新狀態寫入全域字典，給 Flask 取用 ---
            dashboard_state["current_window"] = detector.sliding_window
            dashboard_state["alerts"] = alerts
            
            if alerts:
                print(f"[{index+1}] 🚨 觸發異常: {alerts[0]['anomaly_type']}")
            else:
                print(f"[{index+1}] ✅ 正常: Site {site} 測出 {value}")
            
            # 放慢迴圈速度，讓你有時間打開瀏覽器看 JSON 變化
            time.sleep(0.5) 

    simulate_oneapi_stream("data/example.csv")