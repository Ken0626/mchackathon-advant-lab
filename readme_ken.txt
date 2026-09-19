Python 版本：請務必使用 Python 3.10（對齊官方 OneAPI 環境，避免套件編譯報錯）。

安裝套件：執行 pip install -r requirements.txt。

本地測試 (產生 Mock Data)：執行 python anomaly_detector.py。
終端機會開始模擬機台運作，這時你的資料夾會自動產生一個 dashboard_status.json，且裡面的數字會每 0.5 秒自動更新。這就是前端可以用來開發的真實資料源。

給 正賢 (後端 / OneAPI 整合)
演算法已經封裝為 AnomalyDetector 類別。你的任務是把它接上真實的機台資料流，並負責寫出 JSON 檔案給前端的公佈欄。

1. 初始化與全域狀態準備
在你的 sample.py (或是負責監聽機台事件的主程式) 最上方引入：

Python
import json
from anomaly_detector import AnomalyDetector

# 初始化檢測器 (window_size 預設 16，涵蓋約 4 個機台輪次)
detector = AnomalyDetector(window_size=16)

# 準備給心靈 Dashboard 讀取的全域公佈欄
dashboard_state = {
    "current_window": [], "alerts": [], "limits": {"high": 1.8, "low": 0.6}
}
2. 資料餵入與 JSON 輸出
在你的 consumeData (接收資料的事件迴圈) 中，加入以下邏輯：

Python
# 1. 取得機台資料後，丟給檢測器算數學
alerts = detector.process_new_data(site, value, param_name="220_Main.Suite1#CP")

# 2. 如果觸發異常，呼叫 ActionManager
if alerts:
    for alert in alerts:
        action_cmd = alert["action"]  # 例如 "set_pause" 或 "set_message"
        reason_msg = alert["reason"]
        
        # 依據 action_cmd 執行對應的機台動作
        if action_cmd == "set_pause":
            ActionManager.set_pause(tc.testerId, reason_msg)
        elif action_cmd == "set_message":
            ActionManager.set_message(tc.testerId, reason_msg)

# 3. 關鍵：將最新狀態寫入 dashboard_status.json，讓心靈的程式去讀
dashboard_state["current_window"] = detector.sliding_window
dashboard_state["alerts"] = alerts

with open("dashboard_status.json", "w", encoding="utf-8") as f:
    json.dump(dashboard_state, f, ensure_ascii=False, indent=2)