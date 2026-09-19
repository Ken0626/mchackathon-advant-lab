"""
verify_setup.py
===============
上 Edge VM 之前的自我檢查。逐項驗證每個容易出錯的環節，印出 PASS / FAIL。

每一項都對應一個真實踩過的坑：
  1. 檔案齊不齊
  2. 模型檔在不在、載不載得動
  3. 欄位對齊（盲切錯位不會報錯，是最難抓的 bug）
  4. train/serve 數值尺度一致（上一版 80/80 全紅的原因）
  5. meta 列有沒有多讀一行
  6. 溫度預測是不是攝氏而不是正規化值
  7. 時序限制有沒有被違反
  8. 正常片會不會誤報、異常片抓不抓得到
  9. dashboard JSON 格式
 10. 推論速度夠不夠即時

用法：
    python verify_setup.py                       # 自動找路徑
    python verify_setup.py --data-dir ..\\data\\Data --model-dir models
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PASS, FAIL, WARN = [], [], []


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}" + (f"\n         {detail}" if detail else ""))
    (PASS if ok else (WARN if warn_only else FAIL)).append(name)
    return ok


def resolve(path, *fallbacks):
    for p in (path,) + fallbacks:
        if p and os.path.exists(p):
            return p
    return path


def main(data_dir, model_dir):
    print("=" * 70)
    print("環境自我檢查")
    print("=" * 70)

    # ---------- 1. 程式檔 ----------
    print("\n[1] 程式檔")
    needed = ["window_features.py", "anomaly_detector.py", "oneapi_monitor.py"]
    for f in needed:
        check(f, os.path.exists(os.path.join(_HERE, f)))

    # ---------- 2. 模型檔 ----------
    print("\n[2] 模型檔")
    model_dir = resolve(model_dir, os.path.join(_HERE, "models"))
    print(f"  model_dir = {os.path.abspath(model_dir)}")
    for f, required in [("anomaly_baseline.pkl", True), ("anomaly_iforest.pkl", True),
                        ("temperature_predictors.pkl", False), ("global_limits.json", False)]:
        p = os.path.join(model_dir, f)
        ok = os.path.exists(p)
        size = f"{os.path.getsize(p) / 1024 / 1024:.1f} MB" if ok else "缺少"
        check(f, ok, size, warn_only=not required)

    # 舊世代殘骸必須不存在
    stale = ["scaler.pkl", "autoencoder_pca.pkl", "threshold.json",
             "edge_lightweight_rf.pkl", "feature_names.pkl", "isolation_forest.pkl"]
    left = [f for f in stale if os.path.exists(os.path.join(model_dir, f))]
    check("舊世代 pkl 已清除", not left,
          f"仍存在：{left}（會被舊程式誤載，請刪除）" if left else "")

    if FAIL:
        print("\n模型檔不齊，後續檢查無法進行。請先跑 train_anomaly_model.py。")
        return summary()

    # ---------- 3. 載入偵測器 ----------
    print("\n[3] 偵測器初始化")
    from anomaly_detector import AnomalyDetector
    t0 = time.time()
    det = AnomalyDetector(model_dir=model_dir, verbose=False)
    check("AnomalyDetector 載入成功", True, f"耗時 {time.time() - t0:.2f} 秒")
    check("基準維度一致",
          det.mu.shape == det.sd.shape == (4, len(det.feature_cols)),
          f"mu{det.mu.shape} / {len(det.feature_cols)} 個測項")

    # ---------- 4. 資料檔 ----------
    print("\n[4] 原始資料")
    data_dir = resolve(data_dir, os.path.join(_HERE, "..", "data", "Data"),
                       os.path.join(_HERE, "data", "Data"))
    print(f"  data_dir = {os.path.abspath(data_dir)}")
    found = [i for i in range(1, 26)
             if os.path.exists(os.path.join(data_dir, f"A12345_W{i:02d}_RawResult.csv"))]
    check("找到 RawResult CSV", len(found) > 0, f"共 {len(found)} 片：W{found[:5]}...")
    if not found:
        return summary()

    path = os.path.join(data_dir, f"A12345_W{found[0]:02d}_RawResult.csv")
    head = pd.read_csv(path, nrows=4, low_memory=False)
    df = pd.read_csv(path, skiprows=[1, 2, 3, 4], low_memory=False)

    # ---------- 5. meta 列 ----------
    print("\n[5] CSV 讀取（meta 列 off-by-one）")
    check("meta 列標籤正確", list(head.iloc[:, 0]) == ["Pin", "Test Num", "High Limit", "Low Limit"],
          f"實際：{list(head.iloc[:, 0])}")
    check("資料列數 = 80（沒把 Low Limit 讀成資料）", len(df) == 80, f"實際 {len(df)} 列")
    check("第一列是 die 而非文字", str(df.iloc[0, 0]).strip().isdigit(),
          f"第一列第 0 欄 = {df.iloc[0, 0]!r}")

    # ---------- 6. 欄位對齊 ----------
    print("\n[6] 欄位對齊（盲切錯位不會報錯，必須主動驗）")
    csv_cols = [c for c in df.columns if "_Main." in c]
    check("測項數量一致", len(csv_cols) == len(det.feature_cols),
          f"CSV {len(csv_cols)} 個 vs 模型 {len(det.feature_cols)} 個")
    missing = [c for c in det.feature_cols if c not in set(csv_cols)]
    check("模型要的測項 CSV 都有", not missing,
          f"缺 {len(missing)} 個，例如 {missing[:3]}" if missing else "")
    check("欄位順序一致", csv_cols[:50] == det.feature_cols[:50],
          "前 50 欄順序不同，表示欄位對應錯位" if csv_cols[:50] != det.feature_cols[:50] else "")

    # ---------- 7. 數值尺度 ----------
    print("\n[7] train/serve 數值尺度（上一版全紅的主因）")
    probe = "220_Main.Suite1#CP"
    if probe in df.columns and probe in det.col_index:
        csv_mean = float(pd.to_numeric(df[probe], errors="coerce").mean())
        # 基準的 std 統計量代表視窗內離散程度，量級應與原始值相符
        base_mean_scale = float(np.median(det.mu[0]))
        check(f"{probe} 為原始物理值（約 1.0~1.2，不是 0.5~0.6）",
              0.8 < csv_mean < 1.5,
              f"CSV 平均 = {csv_mean:.4f}。若落在 0.5~0.7 表示資料被正規化過，"
              f"與模型基準不相容")
        check("基準量級合理", base_mean_scale > 0, f"std 基準中位數 = {base_mean_scale:.5f}")

    # ---------- 8. 跑一片正常 + 一片異常 ----------
    print("\n[8] 端到端行為")
    feature_cols = csv_cols

    def run_wafer(wid):
        p = os.path.join(data_dir, f"A12345_W{wid:02d}_RawResult.csv")
        d = pd.read_csv(p, skiprows=[1, 2, 3, 4], low_memory=False)
        d = d[pd.to_numeric(d["Site"], errors="coerce").notna()]
        d[feature_cols] = d[feature_cols].apply(pd.to_numeric, errors="coerce")
        det.on_wafer_start(wid)
        fired, times = [], []
        for _, row in d.iterrows():
            t = time.time()
            al = det.on_device_end(int(row["Site"]), row[feature_cols].to_dict(),
                                   row.get("PF", 0) == 0)
            times.append(time.time() - t)
            for a in al:
                if a["anomaly_type"] not in fired:
                    fired.append(a["anomaly_type"])
        return d, fired, times

    quiet_candidates = [w for w in (4, 5, 8, 12, 20) if w in found]
    noisy_candidates = [w for w in (1, 14, 23, 18) if w in found]

    all_times = []
    if quiet_candidates:
        w = quiet_candidates[0]
        _, fired, times = run_wafer(w)
        all_times += times
        check(f"正常片 W{w:02d} 不誤報", len(fired) == 0,
              f"觸發了 {fired}" if fired else "全程安靜")
    else:
        check("有正常片可測", False, "找不到 W4/W5/W8/W12/W20 任一片", warn_only=True)

    if noisy_candidates:
        w = noisy_candidates[0]
        d, fired, times = run_wafer(w)
        all_times += times
        check(f"異常片 W{w:02d} 有抓到", len(fired) > 0,
              f"觸發 {len(fired)} 類：{fired}")

        # ---------- 9. 溫度預測 ----------
        print("\n[9] 溫度預測單位")
        if det.temp_package is not None:
            msg = det.predict_temperature(1)
            vals = [float(x) for x in msg.replace("site[", "").split()
                    if x.replace(".", "").replace("-", "").isdigit()]
            sensor_col = "100_Main.sensor1#CP"
            truth = float(pd.to_numeric(d[sensor_col], errors="coerce").mean()) \
                if sensor_col in d.columns else None
            check("輸出是攝氏溫度而非正規化值（20~40 之間）",
                  bool(vals) and all(15 < v < 50 for v in vals),
                  f"{msg}"
                  + (f"  |  該片實測平均 {truth:.2f} °C" if truth else ""))
            if truth and vals:
                err = max(abs(v - truth) for v in vals)
                check("預測值與實測平均接近（誤差 < 1 °C）", err < 1.0,
                      f"最大偏差 {err:.3f} °C", warn_only=True)

            # ---------- 10. 時序限制 ----------
            print("\n[10] 時序限制（不能用還沒執行的測項）")
            order = {c: i for i, c in enumerate(det.feature_cols)}
            sensors = ["100_Main.sensor1#CP", "120_Main.sensor2#DS0", "140_Main.sensor3#IO4",
                       "160_Main.sensor4#IO1", "180_Main.sensor5#IO2", "200_Main.sensor6#IO3"]
            bad = []
            for num, target in enumerate(sensors, start=1):
                feats = det.temp_package["features"].get(num, [])
                cut = order.get(target)
                leak = [f for f in feats if order.get(f, -1) >= cut]
                if leak:
                    bad.append((num, leak[:3]))
            check("6 個模型都沒用到未來測項", not bad,
                  f"洩漏：{bad}" if bad else "全部特徵的執行順序都在目標之前")
        else:
            check("溫度模型已載入", False, "找不到 temperature_predictors.pkl", warn_only=True)

    # ---------- 11. 速度 ----------
    if all_times:
        print("\n[11] 推論速度")
        arr = np.array(all_times) * 1000
        check("每顆 die 平均處理時間 < 50 ms", arr.mean() < 50,
              f"平均 {arr.mean():.1f} ms，最慢 {arr.max():.1f} ms（評估視窗那幾顆較慢）")

    # ---------- 12. dashboard ----------
    print("\n[12] Dashboard JSON")
    out = os.path.join(_HERE, "_verify_dashboard.json")
    det.dump_dashboard(out)
    with open(out, encoding="utf-8") as f:
        st = json.load(f)
    need = ["alerts", "current_window", "limits", "scores", "thresholds", "yield"]
    check("必要欄位齊全", all(k in st for k in need),
          f"缺 {[k for k in need if k not in st]}")
    check("current_window 有資料", len(st.get("current_window", [])) > 0,
          f"{len(st.get('current_window', []))} 筆")
    check("limits 有數值", st.get("limits", {}).get("high") is not None,
          f"{st.get('limits')}")
    os.remove(out)

    return summary()


def summary():
    print("\n" + "=" * 70)
    print(f"結果：{len(PASS)} 項通過、{len(WARN)} 項警告、{len(FAIL)} 項失敗")
    if FAIL:
        print("\n失敗項目：")
        for f in FAIL:
            print(f"  - {f}")
        print("\n上 VM 之前請先修掉上列項目。")
    elif WARN:
        print("\n警告項目（不影響核心功能）：")
        for w in WARN:
            print(f"  - {w}")
        print("\n可以進行部署。")
    else:
        print("\n全部通過，可以進行部署。")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()
    sys.exit(main(a.data_dir, a.model_dir))