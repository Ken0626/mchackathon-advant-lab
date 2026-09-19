# Dashboard 介面說明（給心靈）

前後端唯一的介面是 `dashboard_status.json`。
你只需要定時讀這個檔案，不需要呼叫任何 API、不需要 sklearn、不需要碰模型。

---

## 一、怎麼拿到測試檔案

我附了兩個範例，可以直接拿去開發：

| 檔案 | 內容 |
|---|---|
| `sample_normal.json` | 正常狀態，`alerts` 是空陣列 |
| `sample_alert.json` | W01 第 40 顆的告警瞬間，三則告警同時觸發 |

要看即時更新的版本，在我的專案裡跑：

```
cd bin
python simulate_stream.py --data-dir ..\data\Data --wafers 1 --delay 0.3
```

每 0.3 秒覆寫一次 `dashboard_status.json`，模擬真實節拍。
跑 `--wafers 1` 會有告警，`--wafers 4` 全程安靜，兩種都測一下。

---

## 二、完整格式

```json
{
  "wafer_id": 1,
  "device_count": 40,
  "yield": 82.5,
  "monitor_test": "220_Main.Suite1#CP",

  "alerts": [
    {
      "is_anomaly": true,
      "anomaly_type": "Site to Site Unbalance",
      "action": "set_pause",
      "reason": "近 40 顆中有 1.75% 的測項在「site_range」上偏離正常基準超過 4σ（門檻 1.03%）。偏離最大：...",
      "score": 1.75,
      "threshold": 1.03,
      "top_items": [
        { "test": "6460_Main.subflow1.Flow1_Suite295#CP", "z": 35.2 }
      ]
    }
  ],

  "scores":     { "std": 1.55, "slope": 0.36, "rstd_slope": 1.35, "site_range": 1.75 },
  "thresholds": { "std": 1.03, "slope": 0.64, "rstd_slope": 0.73, "site_range": 1.03 },

  "current_window": [
    { "site": 1, "value": 1.088 },
    { "site": 2, "value": 1.098 }
  ],
  "limits": { "high": 1.8, "low": 0.6 },

  "temperature": {
    "unit": "C",
    "latest": {
      "sensor1": { "1": 29.16, "2": 28.46, "3": 28.28, "4": 28.98 }
    },
    "history": {
      "sensor1": [
        { "n": 76, "1": 28.81, "2": 29.43, "3": 28.83, "4": 29.21 },
        { "n": 80, "1": 29.16, "2": 28.46, "3": 28.28, "4": 28.98 }
      ]
    },
    "mae": { "1": 0.021, "2": 0.077, "3": 0.149, "4": 0.112, "5": 0.103, "6": 0.130 }
  }
}
```

---

## 三、欄位說明

### 基本狀態

| 欄位 | 型別 | 說明 |
|---|---|---|
| `wafer_id` | int / null | 當前 wafer 編號 |
| `device_count` | int | 本片已測顆數，0~80 |
| `yield` | float / null | 累計良率（%） |
| `monitor_test` | string | 折線圖畫的是哪個測項 |

### 異常偵測

`alerts` 是陣列，空的代表正常。`anomaly_type` 只會是這六種：

```
Measure Value Shift / Variation
Mean Trend Up/Down
Stdev Trend Up/Down
Site to Site Unbalance
Low Yield
ML Composite Anomaly
```

`action` 只會是 `set_message`（警告）或 `set_pause`（停機）。
兩則以上同時觸發時全部升級為 `set_pause`。

`scores` 和 `thresholds` 的四個 key 一一對應，單位都是百分比。
`scores` 超過同名的 `thresholds` 就是該項觸發。

### 溫度預測（場景二）

`latest` 是每個 sensor 最新一次的四個 site 預測值，單位攝氏，範圍約 25~35 °C。
**注意 site 編號是字串 key**（`"1"` 不是 `1`），因為 JSON 規格不允許數字當 key。

`history` 保留最近 40 個 touchdown，`n` 是當時的 `device_count`，可以拿來當 x 軸畫走勢。

`mae` 是每個 sensor 的模型平均誤差（攝氏），固定不變，可以顯示成信心區間或誤差說明。

---

## 四、畫面建議

**四根長條圖 + 門檻紅線**：`scores` 對 `thresholds`，超標變紅。
這是最能表達「系統正在監控什麼」的視覺，比單純折線圖有看頭。

**折線圖 + 規格紅線**：`current_window` 的 value 對 `limits` 的 high / low。
用 `site` 欄位上色，四個 site 四種顏色 —— Site unbalance 發生時會看到某一條明顯偏離。

**溫度走勢**：`temperature.history.sensor1` 畫四條線（四個 site）。
六個 sensor 可以做成下拉選單或分頁。

**告警細節**：點 alert 展開 `top_items`，顯示是哪幾個測項、偏離幾個 sigma。

---

## 五、五個一定要注意的地方

**1. `alerts` 只有「最近一次評估」的結果，不是累積歷史。**
下一次評估如果正常，`alerts` 會變回空陣列，之前的告警就消失了。
要做告警清單的話，必須在前端自己累積：比對 `device_count` 有沒有變、
`alerts` 非空就 push 進自己的陣列。

**2. 前 40 顆 `scores` 是 `null`。**
滑動視窗還沒滿，系統不判斷。圖表要能處理 null，不要直接 `.toFixed()`。

**3. 換 wafer 時全部歸零。**
`alerts`、`scores`、`temperature` 都會清空，`device_count` 回到 0。
這是預期行為，不是資料遺失。

**4. `limits` 的值可能是 null。**
某些測項沒有有效規格（上下限相同）就會是 null。

**5. `file://` 開的 HTML 不能 fetch 本地 json（CORS 會擋）。**
用最簡單的靜態伺服器：

```
cd bin
python -m http.server 8000
```

瀏覽器開 `http://localhost:8000/你的頁面.html`，
`fetch('dashboard_status.json')` 就能運作。
這個 server 只服務靜態檔案，不算後端 API，跟原本的設計沒有衝突。

---

## 六、輪詢寫法範例

```javascript
let lastCount = -1;
const alertLog = [];

async function poll() {
  try {
    const res = await fetch('dashboard_status.json?t=' + Date.now());
    const s = await res.json();

    if (s.device_count !== lastCount) {
      lastCount = s.device_count;
      if (s.alerts.length > 0) {
        alertLog.push({ at: s.device_count, items: s.alerts });
      }
      render(s, alertLog);
    }
  } catch (e) {
    // 檔案正在被覆寫時偶爾會讀失敗，忽略即可，下一輪會成功
  }
}
setInterval(poll, 500);
```

網址後面加 `?t=` + 時間戳是為了避開瀏覽器快取，不然可能一直讀到同一份舊的。

寫檔端用的是原子覆寫（`os.replace`），所以不會讀到寫到一半的殘缺 JSON，
但檔案切換的瞬間偶爾會讀失敗，try/catch 包起來就好。