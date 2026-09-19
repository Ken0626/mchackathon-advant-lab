"""
prepare_dataset.py  (重寫版)
============================
把 25 片 A12345_W##_RawResult.csv 合併成一張訓練表。

跟舊版最大的差別：**完全不做正規化**。

舊版第 73~79 行的 min-max 正規化是整串問題的源頭：
  - 訓練表裡 220_Main.Suite1#CP 的平均值是 0.582
  - Edge 端從 RawResult 讀到的同一欄同一片 wafer 是 1.102
  - scaler / PCA 在 0.58 的世界 fit，推論吃 1.10 → 重建誤差永遠爆表 → 全紅
而且 RandomForest / IsolationForest 對單調線性縮放本來就不敏感，
正規化沒有帶來任何好處，只帶來這個 train/serve skew。

limits 仍然萃取並存成 JSON，但只用於：
  (1) 前端畫規格紅線
  (2) 溫度預測結果的合理性檢查
不再拿來改動資料本身。
"""

import os
import json
import argparse
import pandas as pd

# TrainDataInfo.txt 的內容
WAFER_INFO = {
    1: "Site unbalance", 2: "Normal", 3: "Low yield", 4: "Normal",
    5: "Normal", 6: "Normal", 7: "Normal", 8: "Normal",
    9: "Low yield", 10: "Normal", 11: "Normal", 12: "Normal",
    13: "Normal", 14: "Mean Trend Up", 15: "Normal", 16: "Normal",
    17: "Normal", 18: "Mean Trend Down", 19: "Normal", 20: "Normal",
    21: "Normal", 22: "Normal", 23: "Stdev Trend Up", 24: "Normal",
    25: "Stdev Trend Down",
}

# RawResult CSV 的 meta 列數：Pin / Test Num / High Limit / Low Limit
# 舊版 anomaly_detector.py 第 148 行寫 skiprows=[1,2,3] 少跳一行，
# 害 "Low Limit" 那列被當成第一筆資料讀進來。
META_ROWS = [1, 2, 3, 4]


def read_wafer(path):
    """回傳 (limits_dict, 資料 DataFrame)。"""
    head = pd.read_csv(path, nrows=4, low_memory=False)
    data = pd.read_csv(path, skiprows=META_ROWS, low_memory=False)

    feature_cols = [c for c in data.columns if "_Main." in c]

    # 注意：檔案裡標成 "High Limit" 的那列實際值比 "Low Limit" 那列小
    #      (Suite1: High=0.6, Low=1.8)，命名是反的。
    #      這裡一律取 min / max，不管它標什麼。
    row_a, row_b = head.iloc[2], head.iloc[3]
    limits = {}
    for c in feature_cols:
        try:
            a, b = float(row_a[c]), float(row_b[c])
        except (ValueError, TypeError):
            continue
        if pd.isna(a) or pd.isna(b) or a == b:
            continue
        limits[c] = {"lo": min(a, b), "hi": max(a, b)}

    data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    data = data[pd.to_numeric(data["Site"], errors="coerce").notna()].copy()
    data["Site"] = data["Site"].astype(int)
    return limits, data


def prepare(base_dir, out_csv, model_dir="models"):
    frames, limits = [], {}
    print("開始合併 wafer 資料（不做正規化）\n")

    for i in range(1, 26):
        path = os.path.join(base_dir, f"A12345_W{i:02d}_RawResult.csv")
        if not os.path.exists(path):
            print(f"  [跳過] 找不到 {path}")
            continue

        lim, df = read_wafer(path)
        if not limits:
            limits = lim
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "global_limits.json"), "w", encoding="utf-8") as f:
                json.dump(limits, f, indent=2)
            print(f"  已擷取 {len(limits)} 個測項的規格上下限 -> {model_dir}/global_limits.json")

        status = WAFER_INFO[i]
        df["Wafer_ID"] = i
        df["Anomaly_Type"] = status
        df["Label"] = 0 if status == "Normal" else 1
        frames.append(df)

        yield_pct = (df["PF"] == 0).mean() * 100 if "PF" in df else float("nan")
        print(f"  W{i:02d}  {status:18s}  {len(df):3d} die  yield={yield_pct:5.1f}%")

    final = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    final.to_csv(out_csv, index=False)
    print(f"\n完成：{final.shape[0]} 筆 × {final.shape[1]} 欄 -> {out_csv}")
    print("（數值為原始物理值，未經任何縮放）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Data")
    ap.add_argument("--out", default="data/training_data_raw.csv")
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()
    prepare(a.data_dir, a.out, a.model_dir)
