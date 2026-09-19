import pandas as pd
import numpy as np

def load_and_simulate(csv_path):
    # 1. 讀取 Meta Data (High / Low Limits)
    df_meta = pd.read_csv(csv_path, nrows=3)
    # 取出 220_Main.Suite1#CP 的上下限
    target_param = "220_Main.Suite1#CP"
    high_limit = float(df_meta.iloc[1][target_param])
    low_limit = float(df_meta.iloc[2][target_param])
    
    print(f"[{target_param}] 規格範圍: {low_limit} ~ {high_limit}")

    # 2. 讀取實際機台數據 (跳過第 1 到 3 列，保留第 0 列標頭)
    df_data = pd.read_csv(csv_path, skiprows=[1, 2, 3])
    
    sliding_window = []
    window_size = 12 # 假設每累積 12 筆資料 (約 3 個輪次) 檢查一次
    
    # 3. 模擬即時資料流
    for index, row in df_data.iterrows():
        current_data = {
            "site": int(row["Site"]),
            "value": float(row[target_param])
        }
        sliding_window.append(current_data)
        
        if len(sliding_window) > window_size:
            sliding_window.pop(0) # 剔除最舊的資料，維持窗口大小
            
        if len(sliding_window) == window_size:
            check_site_unbalance(sliding_window, target_param)

def check_site_unbalance(window, param_name):
    # 將窗口內的資料轉換為 DataFrame 方便分組計算
    df_window = pd.DataFrame(window)
    
    # 計算這個窗口內，各個 Site 的平均值
    site_means = df_window.groupby('site')['value'].mean()
    overall_mean = df_window['value'].mean()
    overall_std = df_window['value'].std()
    
    # 檢查是否有特定 Site 的平均值，偏離整體平均值超過 1.5 倍標準差
    for site, mean_val in site_means.items():
        if abs(mean_val - overall_mean) > (1.5 * overall_std):
            print(f"⚠️ [警報] Site 2 Site Unbalance 觸發！")
            print(f"異常參數: {param_name}, 異常 Site: {site}")
            print(f"該 Site 平均值: {mean_val:.3f}, 整體平均值: {overall_mean:.3f}\n")
            # 實務上這裡會 return 一個 JSON 格式的決策給正賢的 ONEAPI 模組

if __name__ == "__main__":
    # 請確保上一回合的測試資料存放在 data/example.csv
    load_and_simulate("data/example.csv")