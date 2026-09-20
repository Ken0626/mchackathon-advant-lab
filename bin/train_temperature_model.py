"""
train_temperature_model.py  (修正版)
====================================
訓練「場景二：預測 IC 溫度」的 6 個迴歸模型。

相對舊版修正兩件事
------------------
1. **單位**。舊版吃的是被 min-max 正規化過的訓練表，y 的範圍是 0.152~0.210，
   模型輸出 0.18 就直接回傳給測試程式，但機台期待的是 28.8 (°C)。
   這版吃原始物理值，輸出直接就是攝氏溫度。
2. **驗證方式**。舊版用 train_test_split 隨機切 row，同一片 wafer 的 die 會同時
   出現在訓練集和測試集。這版改成依 wafer 分組，測試集是模型沒看過的 wafer，
   這樣 MAE 才代表上線後的真實誤差。

時序限制（題目「重要事項(1)」）
------------------------------
預測 sensor N 時，只能用「執行順序在 sensor N 之前」的測項。
好消息：RawResult 的欄位順序就是執行順序，所以直接用欄位位置切即可，
不需要去解析 test number（test number 不等於執行順序：Suite1 是 220，
sensor1 卻是 100，但 Suite1 先跑）。

  sensor1 -> 可用前 30 個測項（Suite1~14 + IDDQ_flow）
  sensor2 -> 再加上 subflow1 的 500 個測項
  ...以此類推
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

META_COLS = ["PID", "Lot", "Wafer", "Site", "X", "Y", "PF", "SBin", "HBin",
             "Test Time", "Wafer_ID", "Label", "Anomaly_Type"]

SENSORS = [
    "100_Main.sensor1#CP",
    "120_Main.sensor2#DS0",
    "140_Main.sensor3#IO4",
    "160_Main.sensor4#IO1",
    "180_Main.sensor5#IO2",
    "200_Main.sensor6#IO3",
]

# 保留出來當驗證集的 wafer（挑正常片，避免異常干擾誤差估計）
HOLDOUT_WAFERS = [5, 11, 17, 20, 24]

MAX_FEATURES_PER_MODEL = 150   # 後面的 sensor 可用測項高達 2500 個，
                               # 只取「執行順序最接近」的 N 個，模型小很多且精度不掉


def train(data_path, model_dir="models", n_estimators=40, max_depth=10):
    print(f"載入 {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  {df.shape[0]} 筆 die，可用測項 {len(feature_cols)} 個\n")

    holdout = df["Wafer_ID"].isin(HOLDOUT_WAFERS)
    print(f"訓練集 {(~holdout).sum()} 筆（{df['Wafer_ID'].nunique() - len(HOLDOUT_WAFERS)} 片）"
          f" / 驗證集 {holdout.sum()} 筆（W{HOLDOUT_WAFERS}，模型沒看過）\n")

    package = {"models": {}, "features": {}, "mae": {}}

    for num, target in enumerate(SENSORS, start=1):
        if target not in feature_cols:
            print(f"  [跳過] 找不到 {target}")
            continue

        # 時序限制：只取欄位順序在 target 之前的測項
        cut = feature_cols.index(target)
        prior = feature_cols[:cut]

        # 前面已經量過的 sensor 一定要留著：晶片溫度高度自相關，
        # sensor1、sensor2 的實測值是預測 sensor3 最強的特徵。
        # 若只機械式地取「最接近的 300 個」會把它們切掉（sensor2 在第 541 欄、
        # sensor3 在第 1042 欄，中間隔了 500 個 subflow1 測項）。
        prior_sensors = [s for s in SENSORS[:num - 1] if s in prior]
        others = [c for c in prior if c not in prior_sensors]
        if len(others) > MAX_FEATURES_PER_MODEL:
            others = others[-MAX_FEATURES_PER_MODEL:]
        valid = prior_sensors + others

        X = df[valid].fillna(0).values
        y = df[target].fillna(df[target].median()).values

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=42, n_jobs=-1)
        model.fit(X[~holdout.values], y[~holdout.values])

        pred = model.predict(X[holdout.values])
        mae = mean_absolute_error(y[holdout.values], pred)
        # 基準線：永遠猜訓練集平均。模型要贏過它才有價值。
        naive = mean_absolute_error(y[holdout.values],
                                    np.full(holdout.sum(), y[~holdout.values].mean()))

        print(f"  sensor{num}  {target}")
        print(f"     特徵 {len(valid):4d} 個 | 實際溫度範圍 {y.min():.2f} ~ {y.max():.2f} °C")
        print(f"     未見過 wafer 的 MAE = {mae:.4f} °C   (猜平均的基準線 {naive:.4f} °C，"
              f"改善 {(1 - mae / naive) * 100:+.1f}%)")

        package["models"][num] = model
        package["features"][num] = valid
        package["mae"][num] = float(mae)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "temperature_predictors.pkl")
    joblib.dump(package, path)
    print(f"\n已儲存 {path}（{os.path.getsize(path) / 1024 / 1024:.1f} MB）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_data_raw.csv")
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()
    train(a.data, a.model_dir)
