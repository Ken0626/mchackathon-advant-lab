import pandas as pd
import numpy as np

def load_and_simulate(csv_path):
    # 1. 讀取 Meta Data (High / Low Limits)
    df_meta = pd.read_csv(csv_path, nrows=3)
    target_param = "220_Main.Suite1#CP"
    high_limit = float(df_meta.iloc[1][target_param])
    low_limit = float(df_meta.iloc[2][target_param])
    
    print(f"[{target_param}] 規格範圍: {low_limit} ~ {high_limit}")
    print("-" * 50)

    # 2. 讀取實際機台數據
    df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
    
    sliding_window = []
    window_size = 16
    baseline_std = None # 用來儲存歷史平穩期的標準差
    
    # 3. 模擬即時資料流
    for index, row in df_data.iterrows():
        current_data = {
            "site": int(row["Site"]),
            "value": float(row[target_param])
        }
        sliding_window.append(current_data)
        
        if len(sliding_window) > window_size:
            sliding_window.pop(0) 
            
        # 當窗口滿載時，開始執行檢測
        if len(sliding_window) == window_size:
            # 修正 1: 將 list 轉換為 DataFrame，讓後續函式可以使用 Pandas 語法
            df_window = pd.DataFrame(sliding_window)
            
            # 修正 4: 初始化基準標準差 (以前 12 筆資料的標準差為基準)
            if baseline_std is None:
                baseline_std = df_window['value'].std()
                # 避免初始標準差為 0 導致後續除以 0 或過度敏感
                if baseline_std == 0:
                    baseline_std = 0.01 
            
            # 修正 2 & 3: 依序呼叫四個檢測函式，並承接回傳結果
            decisions = [
                check_site_unbalance(df_window, target_param),
                check_trend(df_window, threshold=0.05),
                check_std_trend(df_window, baseline_std, std_multiplier_threshold=2.0),
                check_value_shift(df_window, shift_threshold=0.2)
            ]
            
            # 過濾出有觸發異常的決策
            triggered_alerts = [d for d in decisions if d["is_anomaly"]]
            
            # 如果這個輪次有任何警報，就印出專屬的分隔標題
            if triggered_alerts:
                print(f"========== 窗口評估觸發 (讀入第 {index + 1} 筆數據, 最新為 Site {current_data['site']}) ==========")
                for decision in triggered_alerts:
                    print(f"🚨 [異常攔截] 類型: {decision['anomaly_type']}")
                    print(f"   建議動作: {decision['action']}")
                    print(f"   詳細原因: {decision['reason']}")
                print("-" * 60 + "\n") # 底部加上分隔線並換行

import numpy as np
import pandas as pd

def create_decision(is_anomaly, anomaly_type="Normal", action="none", reason=""):
    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "action": action,         # 給 ONEAPI 用的，例如 "set_message", "set_pause"
        "reason": reason          # 具體原因說明
    }

# 1. 實作異常特徵：Site 2 Site Unbalance (各 Site 數值不平衡)
def check_site_unbalance(df_window, param_name, std_multiplier=1.5):
    """
    比較各 Site 的平均值，若偏離整體平均超過一定倍數的標準差，即判定異常。
    """
    # 計算這個窗口內，各個 Site 的平均值
    site_means = df_window.groupby('site')['value'].mean()
    overall_mean = df_window['value'].mean()
    overall_std = df_window['value'].std()
    
    # 檢查是否有特定 Site 的平均值，偏離整體平均值超過設定的標準差倍數
    for site, mean_val in site_means.items():
        # 避免標準差為 0 導致誤判（例如數據完全一樣時）
        if overall_std > 0 and abs(mean_val - overall_mean) > (std_multiplier * overall_std):
            return create_decision(
                is_anomaly=True,
                anomaly_type="Site 2 Site Unbalance",
                action="set_message", # 觸發機台警報[cite: 1]
                reason=f"異常參數: {param_name}, 異常 Site: {site}, 該 Site 均值: {mean_val:.3f}, 整體均值: {overall_mean:.3f}"
            )
            
    # 若全部 Site 都在正常範圍內，回傳正常狀態
    return create_decision(is_anomaly=False)

# 2. 實作異常特徵：Trend up or down (升級版：按 Site 獨立計算)
def check_trend(df_window, threshold=0.1):
    # 依照 Site 分群，逐一檢查每個測試座的趨勢
    for site, group in df_window.groupby('site'):
        # 單一 Site 至少需要 3 筆資料才能畫出有意義的斜率
        if len(group) < 3:
            continue
            
        y = group['value'].values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        
        if abs(slope) > threshold:
            direction = "up" if slope > 0 else "down"
            return create_decision(
                is_anomaly=True,
                anomaly_type=f"Trend {direction}",
                action="set_message",
                reason=f"Site {site} 出現顯著 {direction} 趨勢，斜率達 {slope:.3f}"
            )
    return create_decision(is_anomaly=False)


# 3. 實作異常特徵：Standard deviation trend change (升級版：按 Site 獨立計算)
def check_std_trend(df_window, baseline_std, std_multiplier_threshold=2.0):
    for site, group in df_window.groupby('site'):
        current_std = group['value'].std()
        
        # 避免資料筆數太少導致標準差為 NaN
        if pd.isna(current_std):
            continue
            
        if current_std > (baseline_std * std_multiplier_threshold):
            return create_decision(
                is_anomaly=True,
                anomaly_type="Standard Deviation Trend Change",
                action="set_message",
                reason=f"Site {site} 標準差異常放大：當前 {current_std:.3f}，基準 {baseline_std:.3f}"
            )
    return create_decision(is_anomaly=False)


# 4. 實作異常特徵：Measure value shifts (升級版：按 Site 獨立計算)
def check_value_shift(df_window, shift_threshold=0.2):
    for site, group in df_window.groupby('site'):
        if len(group) < 4:
            continue
            
        mid_point = len(group) // 2
        first_half_mean = group['value'].iloc[:mid_point].mean()
        second_half_mean = group['value'].iloc[mid_point:].mean()
        
        diff = second_half_mean - first_half_mean
        
        if abs(diff) > shift_threshold:
            direction = "向上" if diff > 0 else "向下"
            return create_decision(
                is_anomaly=True,
                anomaly_type="Measure Value Shift",
                action="set_pause", 
                reason=f"Site {site} 發生 {direction} 斷崖式偏移，落差達 {abs(diff):.3f}"
            )
    return create_decision(is_anomaly=False)


if __name__ == "__main__":
    # 請確保上一回合的測試資料存放在 data/example.csv
    load_and_simulate("data/example.csv")