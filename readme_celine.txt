三、你 → 心靈：dashboard_status.json
怎麼產生

你不用另外寫程式。simulate_stream.py 跑起來就會即時產生真實的檔案：

powershell
cd bin
python simulate_stream.py --data-dir ..\data\Data --wafers 1 --delay 0.3 --dashboard dashboard_status.json

加 --delay 0.3 會讓它每 0.3 秒更新一次，模擬真實節拍，她可以直接對著這個檔案開發輪詢邏輯。跑 W01 有告警、跑 W04 全程安靜，兩種狀態都給她測。

完整格式
json
{
  "wafer_id": 1,
  "device_count": 44,
  "yield": 79.55,
  "monitor_test": "220_Main.Suite1#CP",
  "alerts": [
    {
      "is_anomaly": true,
      "anomaly_type": "Site to Site Unbalance",
      "action": "set_pause",
      "reason": "近 40 顆中有 1.75% 的測項在「site_range」上偏離正常基準超過 4σ（門檻 1.03%）。偏離最大：subflow1.Flow1_Suite295#CP(+35.2σ)、subflow1.Flow1_Suite26#CP(+33.2σ)",
      "score": 1.75,
      "threshold": 1.03,
      "top_items": [
        { "test": "6460_Main.subflow1.Flow1_Suite295#CP", "z": 35.2 },
        { "test": "1080_Main.subflow1.Flow1_Suite26#CP", "z": 33.2 }
      ]
    }
  ],
  "scores": {
    "std": 1.55,
    "slope": 1.19,
    "rstd_slope": 1.38,
    "site_range": 1.75
  },
  "thresholds": {
    "std": 1.03,
    "slope": 0.64,
    "rstd_slope": 0.73,
    "site_range": 1.03
  },
  "current_window": [
    { "site": 1, "value": 1.088 },
    { "site": 2, "value": 1.098 },
    { "site": 3, "value": 1.254 },
    { "site": 4, "value": 1.101 }
  ],
  "limits": { "high": 1.8, "low": 0.6 }
}
欄位說明
欄位	型別	說明
wafer_id	int / null	當前 wafer
device_count	int	本片已測顆數，0~80
yield	float / null	累計良率百分比
monitor_test	string	折線圖畫的是哪個測項
alerts	array	空陣列代表正常
scores	object / null	四個異常分數，單位 %
thresholds	object	四個門檻，固定不變
current_window	array	最近最多 40 筆，依測試順序，不是依 site 排序
limits	object	monitor_test 的規格上下限
她可以怎麼用

scores 對 thresholds 是最有畫面的 —— 四根長條圖配四條門檻紅線，超標的變紅。比單純的折線圖更能表達「系統正在監控什麼」。

current_window 配 limits 畫折線 + 上下規格紅線，這是她原本就在做的。可以用 site 欄位上色，四個 site 四種顏色，site unbalance 發生時會看到某一條明顯偏離。

top_items 可以做成展開式的細節，點告警就看到是哪幾個測項、偏離幾 sigma。

五個必須提醒她的點

第一，alerts 只有「最近一次評估」的結果，不是累積的歷史。 這是最容易踩的。下一次評估如果正常，alerts 會變回空陣列，之前的告警就消失了。她如果要做告警清單，必須自己在前端累積 —— 比對 device_count 有沒有變、alerts 非空就 push 進自己的陣列。

第二，on_wafer_start 會清空 alerts。 換片時整個歸零，這是預期行為。

第三，前 40 顆 scores 是 null。 視窗還沒滿。她的圖表要能處理 null，不要直接 .toFixed() 炸掉。

第四，limits 的值可能是 null。 某些測項沒有有效規格（上下限相同）就會是 null。

第五，直接用 fetch() 讀本地檔案會被 CORS 擋。 如果她的 dashboard 是 file:// 開的 HTML，瀏覽器不讓它 fetch 同目錄的 json。解法是用最簡單的靜態伺服器：

powershell
cd bin
python -m http.server 8000

然後瀏覽器開 http://localhost:8000/dashboard.html，fetch('dashboard_status.json') 就能work。這個 server 只服務靜態檔案，不算「後端 API」，跟你們原本的 B 計畫設計沒有衝突。