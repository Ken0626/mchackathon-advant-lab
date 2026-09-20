"""
rtdi_bridge.py
==============
ONEAPI 事件  <->  AnomalyDetector / 溫度模型 之間的轉接層。

sample.py 只負責「把 ONEAPI 事件拆成 (site, 欄位名稱, 數值)」，
攢資料、交給模型、決定要不要通知機台、寫 dashboard，全部在這裡。
這個檔案不 import oneapi，所以可以在本機直接用 CSV 模擬測試（見檔尾 __main__）。

欄位名稱規則（簡報第 28 頁，已對 models/global_limits.json 的 3035 個 key 驗證過）
--------------------------------------------------------------------------------
    MEASURED_MULTI_PARAM : f"{query_TestNumber}_{query_TestSuite}#{query_PinName}"
                           例：220_Main.Suite1#CP、100_Main.sensor1#CP
    MEASURED_FUNCTIONAL  : f"{query_TestNumber}_{query_TestSuite}"
                           例：540_Main.Suite13（CSV 裡沒有 pin，值 0 = pass）
    MEASURED_PARAMETRIC  : 同上規則；lotidTest / waferidTest 不在訓練欄位中，
                           會被 detector 的 reindex 自動忽略。

執行緒
------
手冊 1.1.3.2：consumeData 與 consumeTPRequest 各自會阻塞自己的執行緒，
兩者可能同時被呼叫，所以所有共用狀態都在 self._cv（Condition）保護下。
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
HISTORY_LEN = 40              # dashboard 溫度走勢保留幾個 touchdown


def column_name(test_number, test_suite, pin_name=None):
    """組出跟訓練表一致的欄位名稱：<test number>_<test suite name>#<pin name>

    手冊 p.63：query_PinName 查不到時回傳 "NOT EXIST"，不是空字串，要濾掉。
    """
    if pin_name and pin_name != "NOT EXIST":
        return f"{test_number}_{test_suite}#{pin_name}"
    return f"{test_number}_{test_suite}"


class RtdiBridge:

    def __init__(self, model_dir=None, dashboard_path=None, alert_log_path=None,
                 predict_wait_s=None, live_results_path=None, monitor_test=None):
        env = os.environ
        self.model_dir = model_dir or env.get("MODEL_DIR", os.path.join(_HERE, "models"))
        self.dashboard_path = dashboard_path if dashboard_path is not None else \
            env.get("DASHBOARD_PATH", os.path.join(_HERE, "dashboard_status.json"))
        self.alert_log_path = alert_log_path if alert_log_path is not None else \
            env.get("ALERT_LOG_PATH", os.path.join(_HERE, "alerts_history.jsonl"))
        # 前端 dashboard/index.html 輪詢的檔案（資料契約 v2）
        self.live_results_path = live_results_path if live_results_path is not None else \
            env.get("LIVE_RESULTS_PATH", os.path.join(_HERE, "live_results.json"))
        self.predict_wait_s = float(predict_wait_s if predict_wait_s is not None
                                    else env.get("PREDICT_WAIT_S", "2.0"))

        self._cv = threading.Condition()

        self.detector = None
        self.temp_package = None
        self._load_models()

        # dashboard 每個 site 顯示哪一個測項的量測值。預設用訓練表的第一欄
        # （220_Main.Suite1#CP，mock_results.json 裡的 test_number "220" 就是它）
        self.monitor_test = monitor_test or env.get("MONITOR_TEST") or (
            self.detector.feature_cols[0] if self.detector is not None
            else "220_Main.Suite1#CP")

        self.sites = [1, 2, 3, 4]      # TESTSTART 會更新成實際的 site 清單
        self.buffer = {}               # {site: {欄位名稱: 數值}}，目前這一次 touchdown
        self.last_complete = {}        # {site: {欄位名稱: 數值}}，該 site 上一顆完整的 die
        self.last_sensor = {}          # {(site, sensor_num): 最近一次實測溫度}，模型失效時的退路
        self.temp_latest = {}          # {"sensor1": {site: 溫度}}，給 dashboard
        self.temp_history = {}         # {"sensor1": [{"n": 第幾顆, "1": 溫度, ...}]}
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
            self.temp_latest.clear()
            self.temp_history.clear()
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
        snapshot = []
        with self._cv:
            for site, passed in site_results:
                results = self.buffer.pop(int(site), {})
                if not results:
                    continue
                self.last_complete[int(site)] = results
                snapshot.append((int(site), results.get(self.monitor_test)))
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
            self._dump_live_results(snapshot)
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
        不是上一顆 die —— receive_temp_predictN 出現在 sensorN 之前（簡報第 4 頁），
        模型也是用 sensorN 之前的測項訓練的（train_temperature_model.py 的欄位切法）。
        用上一顆的資料去預測這一顆，精度會退回接近猜平均。

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

        self._record_temperature(sensor_num, out)
        return out

    def _record_temperature(self, sensor_num, site_values):
        """把預測結果記進 dashboard 的溫度區塊。"""
        key = f"sensor{sensor_num}"
        with self._cv:
            self.temp_latest[key] = {str(s): round(v, 2) for s, v in site_values}
            hist = self.temp_history.setdefault(key, [])
            n = self.detector.total_count if self.detector is not None else len(hist)
            hist.append({"n": n, **{str(s): round(v, 2) for s, v in site_values}})
            if len(hist) > HISTORY_LEN:
                hist.pop(0)

    # ------------------------------------------------------------------
    # 輸出：dashboard 與告警歷史（供特定人員查詢）
    # ------------------------------------------------------------------
    def _dump_dashboard(self):
        """detector 的 dashboard_state() 再加上溫度區塊，原子寫入。

        自己寫檔而不呼叫 detector.dump_dashboard()，是為了不改動 anomaly_detector.py
        就能把場景二的結果一起放進同一份 JSON 給前端輪詢。
        呼叫端必須已持有 self._cv。
        """
        if not self.dashboard_path or self.detector is None:
            return
        try:
            state = self.detector.dashboard_state()
            state["temperature"] = {
                "latest": self.temp_latest,
                "history": self.temp_history,
                "unit": "C",
                "mae": (self.temp_package or {}).get("mae"),
            }
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            tmp = self.dashboard_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.dashboard_path)   # 避免前端讀到寫到一半的檔案
        except Exception:
            log.exception("寫入 dashboard 失敗")

    def _dump_live_results(self, snapshot):
        """寫出前端 dashboard/index.html 輪詢的檔案（validate_data.py 的「資料契約 v2」）。

        格式是每個 site 一筆的扁平陣列，欄位固定為
        site / test_number / value / is_anomaly / anomaly_type / action / reason。

        告警取 detector.last_alerts（最近一次評估的結果，會一直留著）而不是
        本次新產生的告警：dashboard 要顯示的是「目前狀態」，用新告警的話
        非評估回合會變成全部正常，畫面會一閃一閃。
        呼叫端必須已持有 self._cv。
        """
        if not self.live_results_path or not snapshot:
            return

        alert = None
        alerts = self.detector.last_alerts if self.detector is not None else []
        if alerts:
            # set_pause 比 set_message 嚴重，取最嚴重的那筆代表本視窗的狀態
            order = {"set_pause": 2, "set_message": 1}
            alert = max(alerts, key=lambda a: order.get(a.get("action"), 0))

        test_number = str(self.monitor_test).split("_", 1)[0]
        records = []
        for site, value in snapshot:
            rec = {
                "site": int(site),
                "test_number": test_number,
                "value": round(float(value), 6) if value is not None else 0.0,
                "is_anomaly": False,
                "anomaly_type": None,
                "action": "none",
                "reason": "",
            }
            if alert is not None:
                rec["is_anomaly"] = bool(alert.get("is_anomaly", True))
                rec["anomaly_type"] = alert.get("anomaly_type")
                rec["action"] = alert.get("action", "none")
                rec["reason"] = alert.get("reason", "")
            records.append(rec)

        try:
            tmp = self.live_results_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.live_results_path)
        except Exception:
            log.exception("寫入 live_results 失敗")

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
    """題目「重要事項(2)」範例的格式：'prediction 1: (1,28.95) (2,28.71) '"""
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
