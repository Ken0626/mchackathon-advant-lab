一、正賢 → 你：他呼叫你的函式時傳什麼

他要呼叫兩個函式，你要接的東西如下。

on_wafer_start(wafer_id)
參數	型別	說明
wafer_id	int 或 None	當前 wafer 編號，純粹給 dashboard 顯示用

對應 ONEAPI 的 PRODUCTION_WAFERSTART 事件。沒呼叫的話滑動窗口和累計良率不會歸零，上一片的資料會污染下一片。

on_device_end(site, results, passed)

對應 PRODUCTION_TESTEND，一顆 die 測完呼叫一次。

參數	型別	說明
site	int	1~4
results	dict 或 pandas Series	這顆 die 的所有測項結果
passed	bool	這顆有沒有通過

results 的樣子：

python
{
    "220_Main.Suite1#CP": 1.102,
    "240_Main.Suite1_0#DS0": 1.083,
    "260_Main.Suite2#IO1": 1.121,
    ...
    "100_Main.sensor1#CP": 28.95,
    ...
}

鍵必須是 <test number>_<test suite name>#<pin name>，值是 float。約 3036 筆。

你這邊怎麼處理

on_device_end() 的第一件事就是 _to_vector()，把 dict 依照訓練時存下的 feature_cols 順序攤成一個長度 3036 的 numpy array。找不到的鍵補 0，多出來的鍵忽略。這段你不用改。

要跟正賢確認的三件事

第一，欄位名稱的組裝方式。 這是整個介接唯一會靜默失敗的地方。組錯的話 _to_vector() 找不到鍵就補 0，不報錯，但所有特徵歸零 —— 症狀是「完全偵測不到」或「全部異常」。

請他在 oneapi_monitor.py 第 48 行的 _column_name() 裡暫時加一行 print，跑幾顆 die，跟這個比對：

powershell
python -c "import joblib; print(joblib.load('models/anomaly_baseline.pkl')['feature_cols'][:5])"

字串要完全一樣，含 # 和大小寫。對上之後再拿掉 print。

第二，passed 的判定依據。 我在 oneapi_monitor.py 裡暫定 getSoftBin() == 1，但要確認 SmarTest 這邊 SBin 1 是不是就代表通過。訓練資料用的是 PF == 0，兩者要一致，否則良率規則會全錯。

第三，會不會漏測項。 如果某些測項在某些條件下沒執行（例如前面 fail 就跳過後面），那顆 die 的 results 會缺一大塊，被補成 0 之後統計量會爆掉。請他確認並回報實際會拿到幾個鍵。

二、你 → 正賢：你交付給他的東西

你交付的不是資料，是一個可以 import 的模組 + 模型檔。

交付清單
bin/
├── anomaly_detector.py          你的主程式
├── window_features.py           特徵計算（必須跟訓練端同一份）
└── models/
    ├── anomaly_baseline.pkl     2.4 MB
    ├── anomaly_iforest.pkl
    ├── temperature_predictors.pkl  11.7 MB
    └── global_limits.json

外加 requirements.txt 要有 scikit-learn、joblib、pandas、numpy。工作坊簡報第 19 頁列的基礎 image 套件清單裡沒有 sklearn，要他在 dockerfile 自己加。

你回傳給他的資料格式

on_device_end() 回傳一個 list，沒有異常時是空的 []。有異常時每個元素長這樣：

python
{
    "is_anomaly": True,
    "anomaly_type": "Site to Site Unbalance",
    "action": "set_pause",
    "reason": "近 40 顆中有 1.75% 的測項在「site_range」上偏離正常基準超過 4σ（門檻 1.03%）。偏離最大：subflow1.Flow1_Suite295#CP(+35.2σ)、...",
    "score": 1.75,
    "threshold": 1.03,
    "top_items": [
        {"test": "1080_Main.subflow1.Flow1_Suite26#CP", "z": 33.2},
        ...
    ]
}

anomaly_type 只會是這六種之一：

Measure Value Shift / Variation
Mean Trend Up/Down
Stdev Trend Up/Down
Site to Site Unbalance
Low Yield
ML Composite Anomaly

action 只會是 set_message 或 set_pause。兩項以上同時觸發時，全部升級成 set_pause。

predict_temperature(sensor_num)

輸入 1~6 的 int，回傳字串：

"site[1]: 29.16 site[2]: 28.46 site[3]: 28.28 site[4]: 28.98"

單位是攝氏，範圍約 25~35 °C。如果他看到 0.18 這種數字，代表模型檔是舊的。

注意事項

不是 thread-safe。 滑動窗口是共用狀態，ONEAPI 如果多執行緒呼叫 consumeData 會race。簡報第 16 頁說「ONEAPI 會佔用執行緒直到函式結束」，聽起來是單執行緒，但請他確認。真的是多執行緒就要在 on_device_end 外面包一個 threading.Lock。

前 40 顆一定回傳空 list。 視窗未滿不判斷，這是設計。他不要以為是壞掉。

之後每 4 顆才真正評估一次。 中間那 3 顆也是回傳空 list。所以告警只會出現在第 40、44、48… 顆。

例外要吞掉。 consumeData 丟例外會卡住 ONEAPI 執行緒，我在 oneapi_monitor.py 裡已經包了 try/except 並 logging，請他不要拿掉。

predict_temperature 需要各 site 至少測過一顆。 沒有的話會回 site[N]: NA。

效能：平均 12 ms/顆，評估的那幾顆約 110 ms。如果他的測試節拍比這快很多，要調大 window_features.py 的 EVAL_EVERY。