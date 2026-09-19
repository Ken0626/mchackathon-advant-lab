import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_temperature_models(data_path="data/training_data_cleaned.csv", model_dir="models"):
    print("⏳ 開始載入資料，準備訓練溫度預測模型...")
    df = pd.read_csv(data_path)
    
    meta_cols = ['PID', 'Lot', 'Wafer', 'Site', 'X', 'Y', 'PF', 'SBin', 'HBin', 'Test Time', 'Wafer_ID', 'Label', 'Anomaly_Type']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    sensors = [
        "100_Main.sensor1#CP", "120_Main.sensor2#DS0", "140_Main.sensor3#IO4",
        "160_Main.sensor4#IO1", "180_Main.sensor5#IO2", "200_Main.sensor6#IO3"
    ]
    
    predictor_package = {"models": {}, "features": {}}
    
    for i, target_sensor in enumerate(sensors, start=1):
        print(f"\n⚙️ 正在訓練模型 {i}/6: 預測 {target_sensor} ...")
        
        if target_sensor not in feature_cols:
            continue
            
        target_idx = feature_cols.index(target_sensor)
        valid_features = feature_cols[:target_idx]
        
        if not valid_features:
            print(f"   ⚠️ {target_sensor} 之前沒有特徵可用，將使用 Dummy。")
            predictor_package["models"][i] = "DUMMY"
            predictor_package["features"][i] = []
            continue
            
        X = df[valid_features].fillna(0).values
        y = df[target_sensor].fillna(0).values
        
        # 1. 拆分訓練與測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. 只在訓練集上學習
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 3. 在測試集上評估真實誤差
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   ✅ 訓練完成！特徵數: {len(valid_features)}")
        print(f"   📊 測試集真實誤差 -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
        predictor_package["models"][i] = model
        predictor_package["features"][i] = valid_features
        
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(predictor_package, os.path.join(model_dir, "temperature_predictors.pkl"))

if __name__ == "__main__":
    train_temperature_models()