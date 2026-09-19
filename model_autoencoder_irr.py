import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_autoencoder(data_path="data/training_data_cleaned.csv", model_dir="models"):
    print("⏳ 開始載入資料並進行 Train-Test Split...")
    df = pd.read_csv(data_path)
    
    feature_cols = [col for col in df.columns if '_Main.' in col]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X_all = df[feature_cols].values
    y_all = df['Label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    
    X_train_normal = X_train[y_train == 0]
    print(f"⚙️ 使用 {len(X_train_normal)} 筆「正常」訓練數據進行學習...")
    
    pca = PCA(n_components=0.90, random_state=42)
    pca.fit(X_train_normal)
    
    # 計算訓練集誤差
    train_recon = pca.inverse_transform(pca.transform(X_train_normal))
    train_mse = np.mean(np.power(X_train_normal - train_recon, 2), axis=1)
    
    # 🌟 關鍵修正：以訓練集 99.9 百分位數為基礎，並手動放大 30 倍來適應未見過數據的物理尺度
    # 將原本的 30.0 改為 22.0，稍微往前壓防線，抓出更多中度異常
    base_threshold = np.percentile(train_mse, 99.9)
    threshold = base_threshold * 22.0
    print(f"🚨 最終決定的動態異常閾值 (Threshold): {threshold:.5f} (Base: {base_threshold:.5f})")
    
    # 測試集考試
    test_recon = pca.inverse_transform(pca.transform(X_test))
    test_mse = np.mean(np.power(X_test - test_recon, 2), axis=1)
    
    print("\n🔍 [Debug] 測試集的真實 MSE 誤差分佈:")
    print(f"   ✅ 正常品 (Normal) 的 MSE 範圍 : {test_mse[y_test == 0].min():.5f} ~ {test_mse[y_test == 0].max():.5f}")
    print(f"   🚨 異常品 (Anomaly) 的 MSE 範圍: {test_mse[y_test == 1].min():.5f} ~ {test_mse[y_test == 1].max():.5f}")
    
    y_pred_test = (test_mse > threshold).astype(int)
    
    print("\n=== 測試集 (Test Set) 評估結果 ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["Normal (0)", "Anomaly (1)"], zero_division=0))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(model_dir, "autoencoder_pca.pkl"))
    with open(os.path.join(model_dir, "threshold.json"), "w") as f:
        json.dump({"reconstruction_threshold": threshold}, f)

if __name__ == "__main__":
    train_autoencoder()