import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_isolation_forest(data_path="data/training_data_cleaned.csv", model_dir="models"):
    print("⏳ 開始載入資料並進行特徵工程...")
    df = pd.read_csv(data_path)
    target_param = "220_Main.Suite1#CP"
    
    df['Touchdown_ID'] = df.groupby(['Wafer_ID', 'Site']).cumcount()
    df_pivot = df.pivot_table(index=['Wafer_ID', 'Touchdown_ID', 'Label'], columns='Site', values=target_param).reset_index().dropna()
    
    X = df_pivot[[1, 2, 3, 4]].values
    y = df_pivot['Label'].values
    
    # 1. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"✅ 特徵準備完成！訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")
    
    # 2. 在訓練集上建立森林
    iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.10, # [修改] 從 0.01 提高到 0.10 (抓前 10% 極端值)
    random_state=42,
    n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # 3. 在測試集上進行考試
    preds_test = iso_forest.predict(X_test)
    y_pred_test = np.where(preds_test == 1, 0, 1)
    
    print("\n=== 測試集 (Test Set) 評估結果 ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["Normal (0)", "Anomaly (1)"]))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(model_dir, "isolation_forest.pkl"))

if __name__ == "__main__":
    train_isolation_forest()