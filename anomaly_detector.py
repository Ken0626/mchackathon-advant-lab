import os
import numpy as np
import pandas as pd
import joblib
import json
import time

pd.set_option('future.no_silent_downcasting', True)

class AnomalyDetector:
    def __init__(self, window_size=16):
        self.window_size = window_size
        self.history = {1: [], 2: [], 3: [], 4: []}
        
        # 暫存各個 Site 最新一筆的完整 row 資料，供溫度預測使用
        self.latest_full_row = {1: None, 2: None, 3: None, 4: None}
        
        self.baseline_std = None
        self.feature_cols = None
        
        # ---------------------------------------------------------
        # 修改點 1：獨立載入各個模型，避免「一壞全壞」的骨牌效應
        # ---------------------------------------------------------
        
        # 1. 載入 Isolation Forest
        try:
            self.iso_forest = joblib.load("models/isolation_forest.pkl")
        except FileNotFoundError:
            self.iso_forest = None
            
        # 2. 載入 溫度預測模型包
        try:
            self.temp_package = joblib.load("models/temperature_predictors.pkl")
        except FileNotFoundError:
            self.temp_package = None

        # 3. 載入 輕量化 Random Forest 模型 (把你第二支程式產出的模型接上)
        try:
            self.rf_model = joblib.load("models/edge_lightweight_rf.pkl")
            self.rf_features = joblib.load("models/feature_names.pkl")
        except FileNotFoundError:
            self.rf_model, self.rf_features = None, None

        # 4. 載入 舊版的 PCA & Scaler (由於你目前的訓練檔沒產出，這裡出錯也不會影響上面)
        try:
            self.scaler = joblib.load("models/scaler.pkl")
            self.pca = joblib.load("models/autoencoder_pca.pkl")
            with open("models/threshold.json", "r") as f:
                self.pca_threshold = json.load(f)["reconstruction_threshold"]
        except FileNotFoundError:
            self.scaler, self.pca = None, None


    def create_decision(self, is_anomaly, anomaly_type="Normal", action="none", reason=""):
        return {"is_anomaly": is_anomaly, "anomaly_type": anomaly_type, "action": action, "reason": reason}


    def process_new_data(self, site, target_value, target_param, full_row, high_limit, low_limit):
        decisions = []
        self.latest_full_row[site] = full_row 
        
        # --- 雙軌防禦 Track A: PCA 全特徵重建誤差檢查 ---
        if self.pca is not None and self.scaler is not None:
            if self.feature_cols is None:
                self.feature_cols = [col for col in full_row.index if '_Main.' in col]
            
            raw_features = full_row[self.feature_cols].fillna(0).infer_objects(copy=False).values.reshape(1, -1)
            features = self.scaler.transform(raw_features) 
            
            proj = self.pca.transform(features)
            recon = self.pca.inverse_transform(proj)
            mse = np.mean(np.power(features - recon, 2))
            
            if mse > self.pca_threshold:
                decisions.append(self.create_decision(
                    True, "Global Feature Anomaly", "set_pause", f"MSE {mse:.4f} 超出閾值 (Site {site})"
                ))

        # --- 雙軌防禦 Track B: Isolation Forest 關鍵測項與滑動窗口檢查 ---
        self.history[site].append(target_value)
        if len(self.history[site]) > self.window_size:
            self.history[site].pop(0)
            
        if all(len(v) == self.window_size for v in self.history.values()):
            df_window = pd.DataFrame(self.history)
            
            if self.baseline_std is None:
                self.baseline_std = df_window.values.std() or 0.01 
                
            if self.iso_forest is not None and site == 4:
                latest_features = [self.history[s][-1] for s in [1, 2, 3, 4]]
                prediction = self.iso_forest.predict([latest_features])
                if prediction[0] == -1: 
                    decisions.append(self.create_decision(
                        True, "Point Anomaly (IF)", "set_message", f"關鍵測項 {target_param} 發生極端異常"
                    ))

        # ---------------------------------------------------------
        # 修改點 2：輕量化 Random Forest 已經成功載入！
        # 若你未來要在這裡啟用 Track C 即時預測，需在這裡補上 
        # _site_diff 與 _rolling_mean 的即時運算邏輯，並丟入 self.rf_model 預測
        # ---------------------------------------------------------

        return [d for d in decisions if d["is_anomaly"]]


    def predict_temperature(self, target_sensor_num):
        if self.temp_package is None:
            return "Error: ML Model not loaded"
            
        model = self.temp_package["models"][target_sensor_num]
        valid_features = self.temp_package["features"][target_sensor_num]
        
        message = ""
        for site in [1, 2, 3, 4]:
            row = self.latest_full_row[site]
            if row is None:
                message += f"site[{site}]: 0.00 "
                continue
                
            feature_vector = row[valid_features].fillna(0).values.reshape(1, -1)
            prediction = model.predict(feature_vector)[0]
            message += f"site[{site}]: {prediction:.2f} "
            
        return message


# ================= 執行主程式 (模擬 OneAPI 資料流) ================= #
if __name__ == "__main__":
    def simulate_oneapi_stream(base_dir="data/Data"):
        dashboard_state = {"current_window": [], "alerts": [], "limits": {"high": 1.8, "low": 0.6}}
        detector = AnomalyDetector(window_size=16)
        target_param = "220_Main.Suite1#CP"

        print("⏳ 開始模擬機台生產數據雙軌防禦 ...\n")
        for i in range(1, 26):
            file_name = f"A12345_W{i:02d}_RawResult.csv"
            csv_path = os.path.join(base_dir, file_name)

            if not os.path.exists(csv_path):
                continue
                
            df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
            print(f"\n📂 載入 Wafer: {file_name}")
            
            for index, row in df_data.iterrows():
                if pd.isna(row.get("Site")) or target_param not in row or pd.isna(row[target_param]):
                    continue
                    
                site = int(row["Site"])
                target_value = float(row[target_param])
                
                # 執行雙軌 ML 偵測 (傳入整筆 row 供 PCA 掃描)
                alerts = detector.process_new_data(site, target_value, target_param, row, 1.8, 0.6)
                
                # 終端機 Log
                if alerts:
                    # 若有多個模型同時報警，將名稱串接印出
                    alert_types = " & ".join([a['anomaly_type'] for a in alerts])
                    print(f"[{file_name} - {index+1}] 🚨 觸發異常: {alert_types}")
                else:
                    print(f"[{file_name} - {index+1}] ✅ 正常: Site {site} 測出 {target_value:.3f}", end="\r")
                
                time.sleep(0.01)

    simulate_oneapi_stream()