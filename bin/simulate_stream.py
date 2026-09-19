"""
simulate_stream.py
==================
在本機把 RawResult CSV 一顆一顆餵給 AnomalyDetector，模擬機台的即時資料流。

用途：
  1. 上 Edge VM 之前先確認偵測器行為正確（不用等 Gemini 環境）
  2. 產生真實的 dashboard_status.json 給心靈開發前端
  3. Demo 時可以直接跑這支重現異常告警

用法：
    python3 simulate_stream.py --data-dir data/Data --wafers 1 2 14 23
    python3 simulate_stream.py --data-dir data/Data --all --quiet
"""

import os
import time
import argparse
import pandas as pd

from anomaly_detector import AnomalyDetector

META_ROWS = [1, 2, 3, 4]   # Pin / Test Num / High Limit / Low Limit
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(_HERE, "..", "data", "Data")


def load_wafer(path):
    df = pd.read_csv(path, skiprows=META_ROWS, low_memory=False)
    df = df[pd.to_numeric(df["Site"], errors="coerce").notna()].copy()
    df["Site"] = df["Site"].astype(int)
    feature_cols = [c for c in df.columns if "_Main." in c]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df, feature_cols


def run(data_dir, wafers, model_dir, dashboard, delay, quiet):
    det = AnomalyDetector(model_dir=model_dir, verbose=not quiet)
    monitor = None
    summary = []

    for w in wafers:
        path = os.path.join(data_dir, f"A12345_W{w:02d}_RawResult.csv")
        if not os.path.exists(path):
            print(f"  [跳過] 找不到 {path}")
            continue

        df, feature_cols = load_wafer(path)
        monitor = monitor or feature_cols[0]
        det.on_wafer_start(w)
        print(f"\n===== Wafer W{w:02d}  共 {len(df)} 顆 =====")

        fired = {}
        for _, row in df.iterrows():
            results = row[feature_cols].to_dict()
            passed = (row.get("PF", 0) == 0)
            alerts = det.on_device_end(int(row["Site"]), results, passed)

            if dashboard:
                det.dump_dashboard(dashboard, monitor_test=monitor)

            for a in alerts:
                key = a["anomaly_type"]
                if key not in fired:
                    fired[key] = det.total_count
                    print(f"  [第 {det.total_count:3d} 顆] {a['action']:11s} {key}")
                    print(f"            {a['reason']}")
            if delay:
                time.sleep(delay)

        yld = det.pass_count / det.total_count * 100
        print(f"  本片結束：良率 {yld:.1f}%，"
              f"觸發 {len(fired)} 類告警 {list(fired) if fired else '（無）'}")
        summary.append((w, yld, fired))

        # 順便驗證溫度預測可用
        if det.temp_package is not None:
            print(f"  溫度預測 sensor1 -> {det.predict_temperature(1)}")

    print("\n" + "=" * 70)
    print("總結")
    print("=" * 70)
    for w, yld, fired in summary:
        first = min(fired.values()) if fired else None
        status = f"{len(fired)} 類告警，最早在第 {first} 顆" if fired else "無告警"
        print(f"  W{w:02d}  良率 {yld:5.1f}%   {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--wafers", type=int, nargs="*", default=[1, 14, 23, 5])
    ap.add_argument("--all", action="store_true", help="跑全部 25 片")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--dashboard", default="dashboard_status.json",
                    help="設成空字串可關閉寫檔")
    ap.add_argument("--delay", type=float, default=0.0, help="每顆 die 之間的秒數")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, list(range(1, 26)) if a.all else a.wafers,
        a.model_dir, a.dashboard or None, a.delay, a.quiet)
