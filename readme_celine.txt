Python 版本：請務必使用 Python 3.10（對齊官方 OneAPI 環境，避免套件編譯報錯）。

安裝套件：執行 pip install -r requirements.txt。

本地測試 (產生 Mock Data)：執行 python anomaly_detector.py。
終端機會開始模擬機台運作，這時你的資料夾會自動產生一個 dashboard_status.json，且裡面的數字會每 0.5 秒自動更新。這就是前端可以用來開發的真實資料源。

給 心靈 (前端 / Dashboard 整合)
為了讓你不用等機台連線也能「平行開發」，我們的溝通橋樑就是 dashboard_status.json 這個檔案。你的 dashboard.py 只需要負責去「讀取這個公佈欄」即可。

1. 取得最新資料的方式
在你的 Dashboard 程式碼中，寫一個定時器（例如每 0.5 秒或 1 秒），不斷去讀取同一層資料夾底下的 dashboard_status.json 檔案。

2. JSON 資料結構
你讀到的 JSON 格式固定如下：

JSON
{
  "alerts": [
    {
      "action": "set_pause",
      "anomaly_type": "Measure Value Shift",
      "is_anomaly": true,
      "reason": "Site 3 發生 向上 斷崖式偏移，落差達 0.290"
    }
  ],
  "current_window": [
    {"site": 1, "value": 1.25},
    {"site": 2, "value": 1.28},
    {"site": 3, "value": 1.54},
    {"site": 4, "value": 1.26}
    // 最多 16 筆資料，供繪製折線圖或散佈圖 (用 site 分顏色)
  ],
  "limits": {
    "high": 1.8,
    "low": 0.6
  }
}
3. 前端實作建議
圖表視覺：用 current_window 畫出資料線，並用 limits 的 high/low 畫出紅色的規格警戒線。

警報互動：只要判斷 alerts 陣列長度大於 0，就代表有異常。請把 alerts[0].reason 的字串抓出來，顯示在紅色的警報彈窗上。

⚠️ 離線防雷提醒：邊緣端 VM 很可能無法連外網，請確保你的圖表套件與 CSS 都是實體下載到本地端，千萬不要依賴外部網路的 CDN 網址，以免斷網時 Dashboard 變成全白。