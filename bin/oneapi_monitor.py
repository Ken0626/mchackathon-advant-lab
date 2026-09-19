"""
oneapi_monitor.py
=================
給正賢：把 AnomalyDetector 接到 ONEAPI 事件流的骨架。

放進 oneAPI_py3.10/bin/ 下，在 main.py 裡用
    Interface.registerMonitor(SmartMonitor())
註冊即可。

關鍵設計
--------
ONEAPI 是一個測項結果一個事件（MEASURED_PARAMETRIC / MEASURED_MULTI_PARAM），
但偵測器是以「一顆 die 測完」為單位判斷的。
所以這裡先用 self.buffer 把同一顆 die 的所有測項結果攢起來，
等 PRODUCTION_TESTEND 到了才整包交給 detector。

欄位名稱必須組成跟訓練時一致的格式：<test number>_<test suite name>#<pin name>
（簡報第 28 頁的命名規則）。組錯的話 detector 會 reindex 不到而補 0，
不會報錯但會靜默失準 —— 所以下面留了 _column_name() 讓你對照實際事件欄位調整。
"""

import os
import logging

from oneapi import Monitor, DataType, ActionManager   # ONEAPI 提供
from anomaly_detector import AnomalyDetector

DASHBOARD_PATH = os.environ.get("DASHBOARD_PATH", "/tmp/dashboard_status.json")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/nexus/OneAPI/bin/models")


class SmartMonitor(Monitor):

    def __init__(self):
        super().__init__()
        self.detector = AnomalyDetector(model_dir=MODEL_DIR)
        self.buffer = {}     # {site: {欄位名稱: 數值}}
        self.paused = False

    # ---------- 欄位名稱組裝 ----------
    @staticmethod
    def _column_name(data):
        """
        組出跟訓練表一致的欄位名稱。
        若實際事件的屬性名不同，只要改這一個函式，其他地方都不用動。
        """
        num = data.getTestNumber()
        suite = data.getTestSuiteName()
        pin = getattr(data, "getPinName", lambda: None)()
        return f"{num}_{suite}#{pin}" if pin else f"{num}_{suite}"

    # ---------- ONEAPI 進入點 ----------
    def consumeData(self, tc, data):
        try:
            dtype = data.getType()

            if dtype == DataType.DATA_TYP_PRODUCTION_WAFERSTART:
                self.detector.on_wafer_start(getattr(data, "getWaferId", lambda: None)())
                self.buffer.clear()
                self.paused = False

            elif dtype in (DataType.DATA_TYP_MEASURED_PARAMETRIC,
                           DataType.DATA_TYP_MEASURED_MULTI_PARAM):
                site = int(data.getSiteNumber())
                self.buffer.setdefault(site, {})[self._column_name(data)] = float(data.getValue())

            elif dtype == DataType.DATA_TYP_PRODUCTION_TESTEND:
                self._on_test_end(tc, data)

            elif dtype == DataType.DATA_TYP_PRODUCTION_WAFEREND:
                self.detector.dump_dashboard(DASHBOARD_PATH)

        except Exception:
            # consumeData 丟例外會卡住 ONEAPI 的執行緒，一律吞掉並記錄
            logging.exception("consumeData 發生例外")

    # ---------- 一顆 die 測完 ----------
    def _on_test_end(self, tc, data):
        site = int(data.getSiteNumber())
        results = self.buffer.pop(site, {})
        if not results:
            return

        passed = int(getattr(data, "getSoftBin", lambda: 1)()) == 1
        alerts = self.detector.on_device_end(site, results, passed)

        # 每顆 die 都更新 dashboard，心靈的前端 0.5 秒輪詢一次
        self.detector.dump_dashboard(DASHBOARD_PATH)

        for a in alerts:
            text = f"[{a['anomaly_type']}] {a['reason']}"
            ActionManager.set_message(tc.testerId, text[:480])
            logging.warning(text)

            if a["action"] == "set_pause" and not self.paused:
                # 同一片只停一次，避免洗版
                self.paused = True
                ActionManager.set_wait(tc.testerId, 10, text[:200])

    # ---------- 場景二：測試程式來要溫度預測 ----------
    def handleRequest(self, tc, request):
        """
        對應簡報第 5 頁：測試程式送編號 1~6，container 回傳各 site 的預測值。
        實際的 request 解析方式依你現有 sample.py 的 key/data 格式調整。
        """
        import json
        obj = json.loads(request)
        if obj.get("key") != "predict":
            return ""
        sensor_num = int(obj.get("data"))
        message = f"prediction {sensor_num}: " + self.detector.predict_temperature(sensor_num)
        ActionManager.set_wait(tc.testerId, 10, message)
        return ActionManager.get(tc.testerId)
