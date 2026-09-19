"""
rtdi_bridge.py
==============
ONEAPI 事件  <->  Ken 的 AnomalyDetector / 溫度模型 之間的轉接層。

sample.py 只負責「把 ONEAPI 事件拆成 (site, 欄位名稱, 數值)」，
攢資料、交給模型、決定要不要通知機台，全部在這裡。
這個檔案不 import oneapi，所以可以在本機直接用 CSV 模擬測試
（見檔尾的 __main__）。

欄位名稱規則（對照 py-app.log 與 RawResult.csv 驗證過）
------------------------------------------------------
    MEASURED_MULTI_PARAM : f"{query_TestNumber}_{query_TestSuite}#{query_PinName}"
                           例：220_Main.Suite1#CP、100_Main.sensor1#CP
    MEASURED_FUNCTIONAL  : f"{query_TestNumber}_{query_TestSuite}"
                           例：540_Main.Suite13（CSV 裡沒有 pin，值 0 = pass）
    MEASURED_PARAMETRIC  : 同上規則；lotidTest / waferidTest 不在訓練欄位中，
                           會被 detector 自動忽略。

執行緒
------
consumeData（Kafka 資料流）和 consumeTPRequest（測試程式的 predict 請求）
很可能在不同執行緒被呼叫，所以所有共用狀態都在 self._cv（Condition）保護下。
"""

import os
import json
import time
import logging
import threading
from datetime import datetime

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# 題目「重要事項(2)」：測試程式送來的編號 -> 要預測的測項
SENSOR_COLUMNS = {
    1: "100_Main.sensor1#CP",
    2: "120_Main.sensor2#DS0",
    3: "140_Main.sensor3#IO4",
    4: "160_Main.sensor4#IO1",
    5: "180_Main.sensor5#IO2",
    6: "200_Main.sensor6#IO3",
}
FALLBACK_TEMPERATURE = 30.0   # 模型與歷史實測值都沒有時的最後手段


def column_name(test_number, test_suite, pin_name=None):
    """組出跟訓練表一致的欄位名稱：<test number>_<test suite name>#<pin name>"""
    if pin_name and pin_name != "NOT EXIST":
        return f"{test_number}_{test_suite}#{pin_name}"
    return f"{test_number}_{test_suite}"


class RtdiBridge:

    def __init__(self, model_dir=None, dashboard_path=None, alert_log_path=None,
                 predict_wait_s=None):
        env = os.environ
        self.model_dir = model_dir or env.get("MODEL_DIR", os.path.join(_HERE, "models"))
        self.dashboard_path = dashboard_path if dashboard_path is not None else \
            env.get("DASHBOARD_PATH", os.path.join(_HERE, "dashboard_status.json"))
        self.alert_log_path = alert_log_path if alert_log_path is not None else \
            env.get("ALERT_LOG_PATH", os.path.join(_HERE, "alerts_history.jsonl"))
        self.predict_wait_s = float(predict_wait_s if predict_wait_s is not None
                                    else env.get("PREDICT_WAIT_S", "2.0"))

        self._cv = threading.Condition()

        self.detector = None
        self.temp_package = None
        self._load_models()

        self.sites = [1, 2, 3, 4]      # TESTSTART 會更新成實際的 site 清單
        self.buffer = {}               # {site: {欄位名稱: 數值}}，目前這一次 touchdown
        self.last_complete = {}        # {site: {欄位名稱: 數值}}，該 site 上一顆完整的 die
        self.last_sensor = {}          # {(site, sensor_num): 最近一次實測溫度}，模型失效時的退路
        self.wafer_id = None
        self.notified = set()          # 本片已通知過的 anomaly_type，避免每 4 顆洗版一次
        self.paused = False            # 本片是否已送過 pause

    # ------------------------------------------------------------------
    # 模型載入：任何一個失敗都不能讓整個 App 起不來
    # ------------------------------------------------------------------
    def _load_models(self):
        try:
            from anomaly_detector import AnomalyDetector
            self.detector = AnomalyDetector(model_dir=self.model_dir, verbose=True)
            self.temp_package = self.detector.temp_package
        except Exception:
            log.exception("AnomalyDetector 載入失敗，場景一停用")

        if self.temp_package is None:
            try:
                import joblib
                path = os.path.join(self.model_dir, "temperature_predictors.pkl")
                self.temp_package = joblib.load(path)
            except Exception:
                log.exception("溫度模型載入失敗，場景二改用最近實測值")

    # ------------------------------------------------------------------
    # 生產事件（由 sample.py 的 consumeData 呼叫）
    # ------------------------------------------------------------------
    def on_wafer_start(self, wafer_id):
        try:
            wafer_id = int(str(wafer_id).strip())
        except (TypeError, ValueError):
            pass
        with self._cv:
            self.wafer_id = wafer_id
            self.buffer.clear()
            self.notified.clear()
            self.paused = False
            if self.detector is not None:
                self.detector.on_wafer_start(wafer_id)
                self._dump_dashboard()

    def on_test_start(self, sites):
        with self._cv:
            if sites:
                self.sites = sorted(set(int(s) for s in sites))
            self.buffer = {s: {} for s in self.sites}

    def add_result(self, site, name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        with self._cv:
            self.buffer.setdefault(int(site), {})[name] = value
            for num, col in SENSOR_COLUMNS.items():
                if col == name:
                    self.last_sensor[(int(site), num)] = value
            self._cv.notify_all()   # 叫醒正在等資料的 predict 請求

    def on_test_end(self, site_results):
        """
        site_results : [(site, passed), ...]，依 TESTEND 事件的順序
        回傳這次 touchdown 新產生、需要通知機台的告警 list[dict]
        """
        to_notify = []
        with self._cv:
            for site, passed in site_results:
                results = self.buffer.pop(int(site), {})
                if not results:
                    continue
                self.last_complete[int(site)] = results
                if self.detector is None:
                    continue
                for a in self.detector.on_device_end(int(site), results, passed):
                    if a["anomaly_type"] in self.notified:
                        continue
                    self.notified.add(a["anomaly_type"])
                    to_notify.append(a)
                    self._log_alert(a)
            if self.detector is not None:
                self._dump_dashboard()
        return to_notify

    def on_wafer_end(self):
        with self._cv:
            if self.detector is not None:
                self._dump_dashboard()

    def should_pause(self, alert):
        """同一片只送一次 pause。"""
        with self._cv:
            if alert.get("action") == "set_pause" and not self.paused:
                self.paused = True
                return True
            return False

    # ------------------------------------------------------------------
    # 場景二：溫度預測
    # ------------------------------------------------------------------
    def predict(self, sensor_num):
        """
        回傳 [(site, 溫度), ...]。

        預測用的是「這顆 die 到目前為止已經測到的值」（self.buffer），
        不是上一顆 die —— receive_temp_predictN 出現在 sensorN 之前，
        模型就是用 sensorN 之前的測項訓練的。

        Kafka 資料流跟 TP request 走不同通道，request 到的時候前面的測項
        可能還沒全部送達，所以先等最後一個需要的測項到齊（最多 predict_wait_s 秒）。
        """
        sensor_num = int(sensor_num)
        feats = None
        if self.temp_package is not None:
            feats = self.temp_package["features"].get(sensor_num)

        with self._cv:
            if feats:
                last_needed = feats[-1]
                deadline = time.monotonic() + self.predict_wait_s
                while not all(last_needed in self.buffer.get(s, {}) for s in self.sites):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        log.warning(f"predict {sensor_num}: 等 {self.predict_wait_s}s 仍缺 "
                                    f"{last_needed}，缺的特徵改用該 site 上一顆 die 的值")
                        break
                    self._cv.wait(remaining)
            rows = {s: (dict(self.buffer.get(s, {})), self.last_complete.get(s, {}))
                    for s in self.sites}
            sites = list(self.sites)
            history = dict(self.last_sensor)

        out = []
        for s in sites:
            current, previous = rows[s]
            value = None
            if feats:
                try:
                    model = self.temp_package["models"][sensor_num]
                    x = [[current.get(c, previous.get(c, 0.0)) for c in feats]]
                    value = float(model.predict(x)[0])
                except Exception:
                    log.exception(f"sensor{sensor_num} site{s} 模型預測失敗")
            if value is None:
                value = history.get((s, sensor_num), FALLBACK_TEMPERATURE)
            out.append((s, value))
        return out

    # ------------------------------------------------------------------
    # 輸出：dashboard 與告警歷史（供特定人員查詢）
    # ------------------------------------------------------------------
    def _dump_dashboard(self):
        if not self.dashboard_path:
            return
        try:
            self.detector.dump_dashboard(self.dashboard_path)
        except Exception:
            log.exception("寫入 dashboard 失敗")

    def _log_alert(self, alert):
        """dashboard 的 alerts 只保留最近一次評估；這裡另外留一份可查詢的完整歷史。"""
        if not self.alert_log_path:
            return
        rec = dict(alert, time=datetime.now().isoformat(timespec="seconds"),
                   wafer_id=self.wafer_id,
                   device_count=self.detector.total_count if self.detector else None)
        try:
            with open(self.alert_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            log.exception("寫入告警歷史失敗")


def format_prediction(sensor_num, site_values):
    """測試程式端期待的格式（題目「重要事項(2)」的範例）：'prediction 1: (1,28.95) (2,28.71) '"""
    return f"prediction {sensor_num}: " + "".join(f"({s},{v:.2f}) " for s, v in site_values)


def format_alert(alert, wafer_id=None, device_count=None):
    """送到機台的一行文字（set_message 長度有限，這裡控制在 480 字元內）。"""
    where = f"W{wafer_id} #{device_count} " if wafer_id is not None else ""
    return f"[{alert['anomaly_type']}] {where}{alert['reason']}"[:480]


# ----------------------------------------------------------------------
# 本機測試：用 RawResult.csv 模擬 ONEAPI 事件流（每次 touchdown 4 顆）
#   python rtdi_bridge.py --csv ../data/Data/A12345_W01_RawResult.csv
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--wafer", default="1")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, skiprows=[1, 2, 3, 4], low_memory=False)
    cols = [c for c in df.columns if "_Main." in c]
    bridge = RtdiBridge(dashboard_path="", alert_log_path="", predict_wait_s=0.2)
    bridge.on_wafer_start(a.wafer)

    sensor_at = {cols.index(c): n for n, c in SENSOR_COLUMNS.items() if c in cols}
    errs = {n: [] for n in SENSOR_COLUMNS}
    for start in range(0, len(df), 4):
        td = df.iloc[start:start + 4]
        sites = [int(s) for s in td["Site"]]
        bridge.on_test_start(sites)
        # 依欄位（= 執行順序）送事件；碰到 sensor 前先模擬 TP 的 predict 請求
        for i, c in enumerate(cols):
            if i in sensor_at:
                n = sensor_at[i]
                pred = dict(bridge.predict(n))
                for _, row in td.iterrows():
                    errs[n].append(abs(pred[int(row["Site"])] - row[c]))
            for _, row in td.iterrows():
                bridge.add_result(int(row["Site"]), c, row[c])
        for al in bridge.on_test_end([(int(r["Site"]), r["PF"] == 0) for _, r in td.iterrows()]):
            print(format_alert(al, bridge.wafer_id, bridge.detector.total_count))

    for n, e in errs.items():
        if e:
            print(f"sensor{n}: MAE {sum(e) / len(e):.3f} °C over {len(e)} dies")
