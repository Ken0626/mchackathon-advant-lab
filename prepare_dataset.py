import os
import pandas as pd

# 根據 TrainDataInfo.txt 建立標籤字典
# label: 0 代表 Normal, 1 代表 Anomaly
wafer_info = {
    1: {"label": 1, "type": "Site unbalance"},
    2: {"label": 0, "type": "Normal"},
    3: {"label": 1, "type": "Low yield"},
    4: {"label": 0, "type": "Normal"},
    5: {"label": 0, "type": "Normal"},
    6: {"label": 0, "type": "Normal"},
    7: {"label": 0, "type": "Normal"},
    8: {"label": 0, "type": "Normal"},
    9: {"label": 1, "type": "Low yield"},
    10: {"label": 0, "type": "Normal"},
    11: {"label": 0, "type": "Normal"},
    12: {"label": 0, "type": "Normal"},
    13: {"label": 0, "type": "Normal"},
    14: {"label": 1, "type": "Mean Trend Up"},
    15: {"label": 0, "type": "Normal"},
    16: {"label": 0, "type": "Normal"},
    17: {"label": 0, "type": "Normal"},
    18: {"label": 1, "type": "Mean Trend Down"},
    19: {"label": 0, "type": "Normal"},
    20: {"label": 0, "type": "Normal"},
    21: {"label": 0, "type": "Normal"},
    22: {"label": 0, "type": "Normal"},
    23: {"label": 1, "type": "Stdev Trend Up"},
    24: {"label": 0, "type": "Normal"},
    25: {"label": 1, "type": "Stdev Trend Down"}
}

def prepare_data(base_dir="data/Data", output_file="data/training_data_cleaned.csv"):
    all_data = []
    
    print("⏳ 開始整併與清洗 Wafer 資料...\n")
    for i in range(1, 26):
        file_name = f"A12345_W{i:02d}_RawResult.csv"
        csv_path = os.path.join(base_dir, file_name)
        
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到檔案: {csv_path}，跳過。")
            continue
            
        try:
            # 跳過第 1~3 列 (Test Num, High Limit, Low Limit) 以正確對齊 DataFrame
            df = pd.read_csv(csv_path, skiprows=[1, 2, 3])
        except Exception as e:
            print(f"❌ 讀取 {file_name} 發生錯誤: {e}")
            continue
            
        # 1. 防呆：清除 Site 為 NaN 的無效髒數據
        df = df.dropna(subset=['Site'])
        
        # 2. 新增標籤與識別特徵
        df['Wafer_ID'] = i
        df['Label'] = wafer_info[i]["label"]
        df['Anomaly_Type'] = wafer_info[i]["type"]
        
        all_data.append(df)
        print(f"✅ 成功處理 {file_name:25s} | 狀態: {wafer_info[i]['type']}")
        
    if not all_data:
        print("\n❌ 沒有讀取到任何資料，請確認 base_dir 路徑是否正確！")
        return
        
    # 合併所有 DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 輸出成乾淨的 CSV
    final_df.to_csv(output_file, index=False)
    print(f"\n🎉 資料整併完成！共 {len(final_df)} 筆數據，已儲存至 {output_file}")
    
    # 顯示資料分佈狀態
    print("\n=== 資料集標籤分佈 ===")
    print(final_df['Anomaly_Type'].value_counts())
    
    print("\n=== 正常 (0) vs 異常 (1) 比例 ===")
    print(final_df['Label'].value_counts(normalize=True).round(3) * 100)

if __name__ == "__main__":
    # 若你的資料夾結構不同，請修改 base_dir 的路徑
    prepare_data(base_dir="data/Data", output_file="data/training_data_cleaned.csv")