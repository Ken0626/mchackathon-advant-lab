"""
train_anomaly_model.py  (取代 model_autoencoder.py 與 model_isolated_forest.py)
==============================================================================
訓練「場景一：即時異常偵測」所需的基準與模型。

核心觀念的改變
--------------
舊版把偵測單位放在「單顆 die」：用 3036 維的單顆 die 特徵去分類這片 wafer 有沒有異常。
但 TrainDataInfo 列出的 6 種異常 —— Site unbalance / Low yield / Mean Trend Up,Down /
Stdev Trend Up,Down —— 沒有任何一種是單顆 die 的異常，全部都是「一群 die 的分布長相」異常。
一顆 die 的測值本身完全正常，異常只存在於它跟前後數十顆的關係裡。
（實測：舊版逐 die RF 的 anomaly F1，隨機切 row 是 0.604、依 wafer 切是 0.623，兩邊都無效。）

所以這版把偵測單位改成 **滑動視窗**（40 顆 die），特徵改成視窗的統計量。

產出
----
models/anomaly_baseline.pkl   正常基準 + 門檻 + 欄位清單 + 視窗參數
models/anomaly_iforest.pkl    IsolationForest（4 維綜合分數，只用正常資料訓練）

驗證
----
Leave-One-Wafer-Out：25 片輪流當測試，基準永遠只由「其他正常片」建立。
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from window_features import (
    WINDOW_SIZE, EVAL_EVERY, STAT_NAMES, STAT_TO_ANOMALY,
    window_stats, anomaly_scores,
)

NORMAL_WAFERS = [2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24]
ANOMALY_WAFERS = {1: "Site unbalance", 3: "Low yield", 9: "Low yield",
                  14: "Mean Trend Up", 18: "Mean Trend Down",
                  23: "Stdev Trend Up", 25: "Stdev Trend Down"}

YIELD_THRESHOLD = 80.0   # 題目定義：yield 低於 80 視為 Low yield


def extract_all_windows(df, feature_cols):
    """對每片 wafer 切滑動視窗，回傳 {wafer_id: (視窗統計量陣列, 視窗良率陣列)}。"""
    out = {}
    for w in sorted(df["Wafer_ID"].unique()):
        sub = df[df["Wafer_ID"] == w].sort_values("PID")
        X = sub[feature_cols].fillna(0).values.astype(np.float32)
        S = sub["Site"].values
        PF = sub["PF"].values if "PF" in sub else np.zeros(len(sub))

        stats_list, yield_list = [], []
        for a in range(0, len(X) - WINDOW_SIZE + 1, EVAL_EVERY):
            stats_list.append(window_stats(X[a:a + WINDOW_SIZE], S[a:a + WINDOW_SIZE]))
            # 良率用「從 wafer 開始到目前為止的累計值」，不是單一視窗。
            # 單一 40 顆的視窗良率抖動太大，會把正常片（如 W24 的 77.5%）誤判成 Low yield。
            yield_list.append((PF[:a + WINDOW_SIZE] == 0).mean() * 100)
        out[w] = (np.stack(stats_list), np.array(yield_list))
    return out


def build_baseline(windows, wafer_ids):
    """由指定 wafer 的所有視窗算出每測項、每統計量的正常基準 (mu, sd)。"""
    A = np.concatenate([windows[w][0] for w in wafer_ids])   # (視窗數, 4, M)
    return A.mean(axis=0), A.std(axis=0) + 1e-9


def train(data_path, model_dir="models", z_cut=4.0, margin=1.3):
    print(f"載入 {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    feature_cols = [c for c in df.columns if "_Main." in c]
    print(f"  {df.shape[0]} 筆 die，{len(feature_cols)} 個測項")
    print(f"  視窗大小 {WINDOW_SIZE}、每 {EVAL_EVERY} 顆 die 評估一次\n")

    print("切滑動視窗並計算統計量（約需 20~40 秒）...")
    windows = extract_all_windows(df, feature_cols)
    print(f"  每片 wafer {len(windows[list(windows)[0]][0])} 個視窗\n")

    # ---------- Leave-One-Wafer-Out 驗證 ----------
    print("=" * 74)
    print("Leave-One-Wafer-Out 驗證（基準永遠只由其他正常片建立）")
    print("=" * 74)
    header = "wafer  標註               " + "".join(f"{n:>12s}" for n in STAT_NAMES) + "   yield"
    print(header)
    print("-" * 74)

    lowo_rows = []
    for w in sorted(windows):
        base_ids = [x for x in NORMAL_WAFERS if x != w]
        mu, sd = build_baseline(windows, base_ids)
        stats_arr, yields = windows[w]
        sc = np.stack([anomaly_scores(s, mu, sd, z_cut) for s in stats_arr])  # (視窗數, 4)
        peak = sc.max(axis=0)
        label = ANOMALY_WAFERS.get(w, "Normal")
        lowo_rows.append((w, label, peak, yields.min()))
        print(f"W{w:<4d} {label:18s}" + "".join(f"{v:12.2f}" for v in peak) + f"{yields.min():8.1f}%")

    # ---------- 用「正常片的最大視窗分數」訂門檻 ----------
    normal_peaks = np.stack([r[2] for r in lowo_rows if r[0] in NORMAL_WAFERS])
    thresholds = normal_peaks.max(axis=0) * margin
    print("\n門檻（= 正常片視窗最高分 × {:.2f} 安全係數）：".format(margin))
    for n, t in zip(STAT_NAMES, thresholds):
        print(f"  {n:12s} > {t:6.2f} %   -> {STAT_TO_ANOMALY[n]}")
    print(f"  window_yield < {YIELD_THRESHOLD:.0f} %   -> Low yield")

    # ---------- 用門檻回測 ----------
    print("\n" + "=" * 74)
    print("回測結果（門檻套回 25 片）")
    print("=" * 74)
    tp = fp = fn = 0
    for w, label, peak, min_yield in lowo_rows:
        hits = [STAT_TO_ANOMALY[n] for n, v, t in zip(STAT_NAMES, peak, thresholds) if v > t]
        if min_yield < YIELD_THRESHOLD:
            hits.append("Low yield")
        fired = len(hits) > 0
        truth = label != "Normal"
        mark = {(True, True): "TP 正確抓到", (False, False): "TN 正確放行",
                (True, False): "FP 誤報 !!", (False, True): "FN 漏抓 !!"}[(fired, truth)]
        tp += fired and truth
        fp += fired and not truth
        fn += (not fired) and truth
        print(f"W{w:<4d} {label:18s} {mark:12s} {' + '.join(hits) if hits else '-'}")
    print(f"\n  抓到 {tp}/{tp + fn} 片異常，誤報 {fp} 片")

    # ---------- 產出正式基準（用全部 18 片正常）----------
    mu, sd = build_baseline(windows, NORMAL_WAFERS)
    normal_scores = np.concatenate(
        [np.stack([anomaly_scores(s, mu, sd, z_cut) for s in windows[w][0]]) for w in NORMAL_WAFERS]
    )

    iforest = IsolationForest(n_estimators=200, contamination=0.02,
                              random_state=42, n_jobs=-1).fit(normal_scores)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({
        "feature_cols": feature_cols,       # 欄位清單：推論時靠它 reindex，不再盲切
        "mu": mu, "sd": sd,
        "z_cut": z_cut,
        "thresholds": dict(zip(STAT_NAMES, thresholds)),
        "yield_threshold": YIELD_THRESHOLD,
        "window_size": WINDOW_SIZE,
        "eval_every": EVAL_EVERY,
    }, os.path.join(model_dir, "anomaly_baseline.pkl"))
    joblib.dump(iforest, os.path.join(model_dir, "anomaly_iforest.pkl"))

    size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in
               ["anomaly_baseline.pkl", "anomaly_iforest.pkl"]) / 1024
    print(f"\n已儲存 anomaly_baseline.pkl + anomaly_iforest.pkl（合計 {size:.0f} KB）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training_data_raw.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--z-cut", type=float, default=4.0)
    ap.add_argument("--margin", type=float, default=1.3)
    a = ap.parse_args()
    train(a.data, a.model_dir, a.z_cut, a.margin)
