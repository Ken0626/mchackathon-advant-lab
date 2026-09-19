import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_lightweight_edge_model(data_path="data/training_data_cleaned.csv", model_dir="models"):
    print("⏳ 開始載入資料並進行特徵工程...")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if '_Main.' in col]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # ---------------------------------------------------------
    # 核心改良：建立時間序列與空間特徵 (針對 Trend 與 Site Unbalance)
    # (已修正 DataFrame 記憶體破碎化警告)
    # ---------------------------------------------------------
    frames_to_concat = [df] # 用來收集所有 DataFrame 進行一次性合併
    
    # 1. 捕捉 Site Unbalance: 計算每個 Site 的歷史平均偏離度
    if 'Site' in df.columns:
        site_means = df.groupby('Site')[feature_cols].transform('mean')
        site_diff_df = df[feature_cols] - site_means
        site_diff_cols = [f"{c}_site_diff" for c in feature_cols]
        site_diff_df.columns = site_diff_cols
        frames_to_concat.append(site_diff_df)
    else:
        site_diff_cols = []

    # 2. 捕捉 Trend Issue: 計算最近 5 筆測試的移動平均 (Rolling Mean)
    rolling_df = df[feature_cols].rolling(window=5, min_periods=1).mean()
    rolling_cols = [f"{c}_rolling_mean" for c in feature_cols]
    rolling_df.columns = rolling_cols
    frames_to_concat.append(rolling_df)
    
    # 一次性在橫向 (axis=1) 合併所有欄位，徹底解決 PerformanceWarning
    df = pd.concat(frames_to_concat, axis=1)

    # 合併所有要餵給模型的特徵名稱
    all_features = feature_cols + site_diff_cols + rolling_cols
    
    X_all = df[all_features].values
    y_all = df['Label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    
    # ---------------------------------------------------------
    # 核心改良：使用極輕量的 Random Forest
    # n_estimators=30, max_depth=5 可以將模型大小壓縮在幾百 KB 內，推論時間極短
    # ---------------------------------------------------------
    print(f"⚙️ 訓練輕量化 Random Forest 模型 (使用 {len(all_features)} 個特徵)...")
    rf_model = RandomForestClassifier(
        n_estimators=30,     # 減少樹的數量以降低模型體積
        max_depth=5,         # 限制深度以加快推論速度
        class_weight="balanced", 
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 驗證測試集
    y_pred_test = rf_model.predict(X_test)
    print("\n=== 測試集 (Test Set) 評估結果 ===")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["Normal (0)", "Anomaly (1)"], zero_division=0))
    
    os.makedirs(model_dir, exist_ok=True)
    # 儲存輕量化模型
    joblib.dump(rf_model, os.path.join(model_dir, "edge_lightweight_rf.pkl"))
    
    # 為了邊緣推論，我們需要把特徵名稱也存下來
    joblib.dump(all_features, os.path.join(model_dir, "feature_names.pkl"))
    print("\n✅ 輕量化模型與特徵清單已儲存！適合部署於 ACS Edge Server。")

if __name__ == "__main__":
    train_lightweight_edge_model()