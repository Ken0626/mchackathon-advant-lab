"""
anomaly_detector.py  (重寫版)
=============================
Edge 端即時偵測器。由正賢的 consumeData 呼叫。

與舊版的差異
------------
1. 特徵計算改為呼叫 window_features.py，與訓練端是同一份程式碼，
   不可能再發生「訓練吃正規化值、推論吃原始值」的 train/serve skew。
2. 不再載入 scaler.pkl / autoencoder_pca.pkl / threshold.json。
   那三個檔案是舊世代的殘骸，且 autoencoder_pca 是在「未經 scaler」的資料上 fit 的，
   推論時卻先套 scaler.transform，兩邊對不上，所以每筆都爆表 -> 80/80 全紅。
3. 欄位對齊改用訓練時存下的 feature_cols 清單做 reindex，
   不再用 all_main_cols[:3036] 盲切（盲切錯位時不會報錯，是最難抓的 bug）。
4. 告警理由會指名「是哪幾個測項、偏離幾個 sigma」，不再只有一句 MSE 超標。

用法
----
    det = AnomalyDetector(model_dir="models")
    det.on_wafer_start(wafer_id)
    alerts = det.on_device_end(site=2, results={"220_Main.Suite1#CP": 1.10, ...}, passed=True)
    msg    = det.predict_temperature(1)      # 回傳攝氏溫度字串
    det.dump_dashboard("dashboard_status.json")
"""

import os
import json
from collections import deque

import numpy as np
import joblib

from window_features import (
    STAT_NAMES, STAT_TO_ANOMALY, window_stats, anomaly_scores, top_offenders,
)


class AnomalyDetector:

    def __init__(self, model_dir="models", verbose=True):
        self.verbose = verbose
        self.model_dir = model_dir

        base_path = os.path.join(model_dir, "anomaly_baseline.pkl")
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"找不到 {base_path}。請先執行 train_anomaly_model.py。"
            )
        base = joblib.load(base_path)

        self.feature_cols = base["feature_cols"]
        self.col_index = {c: i for i, c in enumerate(self.feature_cols)}
        self.mu = base["mu"]
        self.sd = base["sd"]
        self.z_cut = base["z_cut"]
        self.thresholds = np.array([base["thresholds"][n] for n in STAT_NAMES])
        self.yield_threshold = base["yield_threshold"]
        self.window_size = base["window_size"]
        self.eval_every = base["eval_every"]

        self.iforest = self._try_load("anomaly_iforest.pkl")
        self.temp_package = self._try_load("temperature_predictors.pkl")

        # 滑動視窗狀態
        self.win_values = deque(maxlen=self.window_size)
        self.win_sites = deque(maxlen=self.window_size)
        self.device_count = 0
        self.pass_count = 0
        self.total_count = 0

        # 每個 site 最新一筆完整結果，供溫度預測使用
        self.latest_row = {1: None, 2: None, 3: None, 4: None}

        self.wafer_id = None
        self.last_alerts = []
        self.last_scores = None

        # 溫度預測結果（給 dashboard 顯示，不影響回覆機台的內容）
        self.temp_latest = {}
        self.temp_history = {}

        if verbose:
            print(f"[Detector] 基準載入完成：{len(self.feature_cols)} 個測項，"
                  f"視窗 {self.window_size}，每 {self.eval_every} 顆評估一次")
            print(f"[Detector] 門檻 " +
                  ", ".join(f"{n}={t:.2f}%" for n, t in zip(STAT_NAMES, self.thresholds)) +
                  f", yield<{self.yield_threshold:.0f}%")

    def _try_load(self, name):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            return joblib.load(path)
        if self.verbose:
            print(f"[Detector] 提醒：找不到 {path}，相關功能停用")
        return None

    # ------------------------------------------------------------------
    # 生產事件
    # ------------------------------------------------------------------
    def on_wafer_start(self, wafer_id=None):
        """對應 PRODUCTION_WAFERSTART。清空視窗與累計良率。"""
        self.wafer_id = wafer_id
        self.win_values.clear()
        self.win_sites.clear()
        self.device_count = 0
        self.pass_count = 0
        self.total_count = 0
        self.last_alerts = []
        self.temp_latest = {}
        self.temp_history = {}

    def _to_vector(self, results):
        """把 {測項名稱: 數值} 對齊成訓練時的欄位順序。缺測項補 0。"""
        vec = np.zeros(len(self.feature_cols), dtype=np.float32)
        for name, value in results.items():
            idx = self.col_index.get(name)
            if idx is not None:
                try:
                    vec[idx] = float(value)
                except (TypeError, ValueError):
                    pass
        return vec

    def on_device_end(self, site, results, passed=True):
        """
        對應 PRODUCTION_TESTEND（一顆 die 測完）。

        site    : int 1~4
        results : dict {測項欄位名稱: 數值}，或 pandas Series
        passed  : 這顆 die 是否通過

        回傳 list[dict]，格式沿用你們既有的 JSON contract：
        {"is_anomaly", "anomaly_type", "action", "reason"}，另外多帶 score / details。
        """
        if hasattr(results, "to_dict"):
            results = results.to_dict()

        self.win_values.append(self._to_vector(results))
        self.win_sites.append(int(site))
        self.latest_row[int(site)] = results
        self.device_count += 1
        self.total_count += 1
        self.pass_count += bool(passed)

        # 視窗還沒滿 -> 不判斷（基準是用滿視窗算的，半滿的視窗統計量沒有可比性）
        if len(self.win_values) < self.window_size:
            return []
        # 每 eval_every 顆才算一次，避免每顆 die 都做一次 40×3036 的統計
        if self.device_count % self.eval_every != 0:
            return []

        return self._evaluate()

    # ------------------------------------------------------------------
    # 核心判斷
    # ------------------------------------------------------------------
    def _evaluate(self):
        values = np.stack(self.win_values)
        sites = np.array(self.win_sites)

        stats = window_stats(values, sites)
        scores = anomaly_scores(stats, self.mu, self.sd, self.z_cut)
        self.last_scores = {n: float(v) for n, v in zip(STAT_NAMES, scores)}

        alerts = []

        # --- 規則層：4 個統計量各自比門檻 ---
        for i, name in enumerate(STAT_NAMES):
            if scores[i] <= self.thresholds[i]:
                continue
            worst = top_offenders(stats, self.mu, self.sd, i, self.feature_cols, k=3)
            detail = "、".join(f"{c.split('_Main.')[-1]}({z:+.1f}σ)" for c, z in worst)
            alerts.append({
                "is_anomaly": True,
                "anomaly_type": STAT_TO_ANOMALY[name],
                "action": "set_message",
                "reason": (f"近 {self.window_size} 顆中有 {scores[i]:.2f}% 的測項"
                           f"在「{name}」上偏離正常基準超過 {self.z_cut:.0f}σ"
                           f"（門檻 {self.thresholds[i]:.2f}%）。"
                           f"偏離最大：{detail}"),
                "score": float(scores[i]),
                "threshold": float(self.thresholds[i]),
                "top_items": [{"test": c, "z": round(z, 2)} for c, z in worst],
            })

        # --- 規則層：累計良率 ---
        cum_yield = self.pass_count / self.total_count * 100
        if self.total_count >= self.window_size and cum_yield < self.yield_threshold:
            alerts.append({
                "is_anomaly": True,
                "anomaly_type": "Low Yield",
                "action": "set_pause",
                "reason": (f"本片累計良率 {cum_yield:.1f}%（已測 {self.total_count} 顆），"
                           f"低於 {self.yield_threshold:.0f}% 門檻"),
                "score": float(cum_yield),
                "threshold": float(self.yield_threshold),
                "top_items": [],
            })

        # --- ML 層：IsolationForest 對 4 維綜合分數做整體判斷 ---
        # 只用正常視窗訓練，補捉「每一項都沒單獨超標、但組合起來不像正常」的情況
        if self.iforest is not None:
            if self.iforest.predict(scores.reshape(1, -1))[0] == -1:
                raw = float(self.iforest.decision_function(scores.reshape(1, -1))[0])
                if not alerts:   # 規則沒抓到才發，避免同一件事重複告警
                    alerts.append({
                        "is_anomaly": True,
                        "anomaly_type": "ML Composite Anomaly",
                        "action": "set_message",
                        "reason": (f"IsolationForest 判定本視窗的統計組合偏離正常族群"
                                   f"（分數 {raw:+.4f}，越負越異常）：" +
                                   "、".join(f"{n}={v:.2f}%" for n, v in self.last_scores.items())),
                        "score": raw,
                        "threshold": 0.0,
                        "top_items": [],
                    })

        # 兩個以上同時觸發 -> 升級為停機
        if len(alerts) >= 2:
            for a in alerts:
                a["action"] = "set_pause"

        self.last_alerts = alerts
        return alerts

    # ------------------------------------------------------------------
    # 場景二：溫度預測
    # ------------------------------------------------------------------
    def predict_temperature(self, sensor_num, current_results=None):
        """
        測試程式送編號 1~6 過來，回傳四個 site 的預測溫度（攝氏）。
        格式沿用簡報第 5 頁的樣子： 'site[1]: 28.95 site[2]: 28.71 ...'

        current_results : {site: {測項名稱: 數值}}
            「這一顆 die 目前為止已經量到的結果」。必須由呼叫端傳入。

        為什麼一定要傳這個而不是用 self.latest_row：
        測試流程是 Suite1~14 -> IDDQ -> receive_temp_predict1 -> sensor1 -> subflow1 -> ...
        也就是說，預測請求發生在這顆 die「還沒測完」的時候。
        self.latest_row 要等 PRODUCTION_TESTEND 才會更新，那時候拿到的是「上一顆」的資料。
        用上一顆的測項去預測這一顆的溫度，精度會退回接近猜平均的水準。

        傳 buffer 進來還有一個額外好處：時序限制變成物理上保證的。
        buffer 裡根本不會有還沒執行的測項，想洩漏也洩漏不了。
        """
        if self.temp_package is None:
            return "Error: temperature model not loaded"
        if sensor_num not in self.temp_package["models"]:
            return f"Error: no model for sensor {sensor_num}"

        source = current_results if current_results is not None else self.latest_row

        model = self.temp_package["models"][sensor_num]
        feats = self.temp_package["features"][sensor_num]
        idx = [self.col_index.get(c) for c in feats]

        parts = []
        for site in (1, 2, 3, 4):
            row = source.get(site) if hasattr(source, "get") else None
            if not row:
                parts.append(f"site[{site}]: NA")
                continue
            vec = self._to_vector(row)
            x = np.array([[vec[i] if i is not None else 0.0 for i in idx]])
            parts.append(f"site[{site}]: {float(model.predict(x)[0]):.2f}")
        return " ".join(parts)

    def record_temperature(self, sensor_num, message):
        """
        把一次溫度預測的結果記下來，供 dashboard 顯示。
        由 oneapi_monitor 在回覆測試程式之後順手呼叫，不影響回覆本身的延遲。

        同時保留每個 sensor 的預測歷史（最近 40 個 touchdown），
        讓前端可以畫溫度隨時間的走勢，而不只是當下這一個數字。
        """
        parsed = {}
        for token in str(message).split("site[")[1:]:
            try:
                num, rest = token.split("]:", 1)
                parsed[int(num)] = float(rest.split()[0])
            except (ValueError, IndexError):
                continue
        if not parsed:
            return

        key = f"sensor{sensor_num}"
        self.temp_latest[key] = parsed
        hist = self.temp_history.setdefault(key, [])
        hist.append({"n": self.total_count, **{str(k): v for k, v in parsed.items()}})
        if len(hist) > 40:
            hist.pop(0)

    # ------------------------------------------------------------------
    # 給心靈的 Dashboard
    # ------------------------------------------------------------------
    def dashboard_state(self, monitor_test=None):
        """產生前端要輪詢的狀態。維持你們既有的 JSON contract 再加欄位。"""
        monitor_test = monitor_test or self.feature_cols[0]
        col = self.col_index.get(monitor_test, 0)

        window = [{"site": int(s), "value": float(v[col])}
                  for s, v in zip(self.win_sites, self.win_values)]

        return {
            "wafer_id": self.wafer_id,
            "device_count": self.total_count,
            "yield": round(self.pass_count / self.total_count * 100, 2) if self.total_count else None,
            "monitor_test": monitor_test,
            "alerts": self.last_alerts,
            "scores": self.last_scores,
            "thresholds": {n: float(t) for n, t in zip(STAT_NAMES, self.thresholds)},
            "current_window": window,
            "limits": self._limits_for(monitor_test),
            "temperature": {
                "latest": self.temp_latest,
                "history": self.temp_history,
                "unit": "C",
                "mae": self.temp_package.get("mae") if self.temp_package else None,
            },
        }

    def _limits_for(self, test_name):
        path = os.path.join(self.model_dir, "global_limits.json")
        if not os.path.exists(path):
            return {"high": None, "low": None}
        with open(path, encoding="utf-8") as f:
            lim = json.load(f)
        e = lim.get(test_name)
        return {"high": e["hi"], "low": e["lo"]} if e else {"high": None, "low": None}

    def dump_dashboard(self, path="dashboard_status.json", monitor_test=None):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.dashboard_state(monitor_test), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)   # 原子寫入，避免前端讀到寫到一半的檔案
