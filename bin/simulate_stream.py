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
import numpy as np
import pandas as pd

from anomaly_detector import AnomalyDetector

META_ROWS = [1, 2, 3, 4]   # Pin / Test Num / High Limit / Low Limit


def load_wafer(path):
    df = pd.read_csv(path, skiprows=META_ROWS, low_memory=False)
    df = df[pd.to_numeric(df["Site"], errors="coerce").notna()].copy()
    df["Site"] = df["Site"].astype(int)
    feature_cols = [c for c in df.columns if "_Main." in c]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df, feature_cols


SENSORS = [
    "100_Main.sensor1#CP", "120_Main.sensor2#DS0", "140_Main.sensor3#IO4",
    "160_Main.sensor4#IO1", "180_Main.sensor5#IO2", "200_Main.sensor6#IO3",
]


def parse_prediction(msg):
    """把 'site[1]: 29.16 site[2]: 28.46 ...' 拆成 {site: 溫度}。"""
    out = {}
    for token in msg.split("site[")[1:]:
        try:
            num, rest = token.split("]:", 1)
            out[int(num)] = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def run(data_dir, wafers, model_dir, dashboard, delay, quiet, check_temp=False):
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

        order = {c: i for i, c in enumerate(feature_cols)}
        cut = {n: order[s] for n, s in enumerate(SENSORS, start=1) if s in order}
        group, errors = {}, {n: [] for n in cut}

        fired = {}
        for _, row in df.iterrows():
            results = row[feature_cols].to_dict()
            passed = (row.get("PF", 0) == 0)
            site = int(row["Site"])
            alerts = det.on_device_end(site, results, passed)

            # 一個 touchdown 四個 site 測完 -> 重現測試程式的六次預測請求
            group[site] = results
            if len(group) == 4:
                for num, cut_idx in cut.items():
                    partial = {s: {c: v for c, v in r.items() if order[c] < cut_idx}
                               for s, r in group.items()}
                    msg = det.predict_temperature(num, current_results=partial)
                    det.record_temperature(num, msg)      # 寫進 dashboard
                    if check_temp:
                        pred = parse_prediction(msg)
                        for s, r in group.items():
                            truth = r.get(SENSORS[num - 1])
                            if s in pred and truth is not None and not pd.isna(truth):
                                errors[num].append(abs(pred[s] - truth))
                group = {}

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

        # ---- 場景二：時序模擬的準確度統計 ----
        if det.temp_package is None:
            pass
        elif check_temp:
            print("  溫度預測（依測試流程時序，與實測值比對）：")
            for num in sorted(errors):
                e = errors[num]
                if not e:
                    continue
                mae, worst = float(np.mean(e)), float(np.max(e))
                flag = "" if mae < 0.5 else "   <-- 偏大"
                print(f"    sensor{num}  MAE {mae:.3f} °C   最差 {worst:.3f} °C"
                      f"   ({len(e)} 筆){flag}")
        else:
            latest = det.temp_latest.get("sensor1", {})
            shown = " ".join(f"site[{k}]: {v:.2f}" for k, v in sorted(latest.items()))
            print(f"  溫度預測 sensor1 -> {shown or 'NA'}"
                  f"   （加 --check-temp 可看六個 sensor 的誤差）")

    print("\n" + "=" * 70)
    print("總結")
    print("=" * 70)
    for w, yld, fired in summary:
        first = min(fired.values()) if fired else None
        status = f"{len(fired)} 類告警，最早在第 {first} 顆" if fired else "無告警"
        print(f"  W{w:02d}  良率 {yld:5.1f}%   {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Data")
    ap.add_argument("--wafers", type=int, nargs="*", default=[1, 14, 23, 5])
    ap.add_argument("--all", action="store_true", help="跑全部 25 片")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--dashboard", default="dashboard_status.json",
                    help="設成空字串可關閉寫檔")
    ap.add_argument("--delay", type=float, default=0.0, help="每顆 die 之間的秒數")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check-temp", action="store_true",
                    help="依測試流程時序重現六次溫度預測並與實測值比對")
    a = ap.parse_args()
    run(a.data_dir, list(range(1, 26)) if a.all else a.wafers,
        a.model_dir, a.dashboard or None, a.delay, a.quiet, a.check_temp)
