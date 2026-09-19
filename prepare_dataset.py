import os
import pandas as pd
import numpy as np
import json

wafer_info = {
    1: {"label": 1, "type": "Site unbalance"}, 2: {"label": 0, "type": "Normal"},
    3: {"label": 1, "type": "Low yield"}, 4: {"label": 0, "type": "Normal"},
    5: {"label": 0, "type": "Normal"}, 6: {"label": 0, "type": "Normal"},
    7: {"label": 0, "type": "Normal"}, 8: {"label": 0, "type": "Normal"},
    9: {"label": 1, "type": "Low yield"}, 10: {"label": 0, "type": "Normal"},
    11: {"label": 0, "type": "Normal"}, 12: {"label": 0, "type": "Normal"},
    13: {"label": 0, "type": "Normal"}, 14: {"label": 1, "type": "Mean Trend Up"},
    15: {"label": 0, "type": "Normal"}, 16: {"label": 0, "type": "Normal"},
    17: {"label": 0, "type": "Normal"}, 18: {"label": 1, "type": "Mean Trend Down"},
    19: {"label": 0, "type": "Normal"}, 20: {"label": 0, "type": "Normal"},
    21: {"label": 0, "type": "Normal"}, 22: {"label": 0, "type": "Normal"},
    23: {"label": 1, "type": "Stdev Trend Up"}, 24: {"label": 0, "type": "Normal"},
    25: {"label": 1, "type": "Stdev Trend Down"}
}

def prepare_data(base_dir="data/Data", output_file="data/training_data_cleaned.csv"):
    all_data = []
    global_limits = {}
    
    print("⏳ 開始提取物理規格 (Limits) 並清洗 Wafer 資料...\n")
    
    for i in range(1, 26):
        file_name = f"A12345_W{i:02d}_RawResult.csv"
        csv_path = os.path.join(base_dir, file_name)
        
        if not os.path.exists(csv_path):
            continue
            
        # 1. 讀取前 5 列以捕捉 Limit 資訊 (不略過任何行)
        df_head = pd.read_csv(csv_path, nrows=5, header=0, low_memory=False)
        
        # 找出 High Limit 與 Low Limit 所在的 row (第一欄 PID 為指標)
        high_row = df_head[df_head.iloc[:, 0] == 'High Limit']
        low_row = df_head[df_head.iloc[:, 0] == 'Low Limit']
        
        # 讀取實際測試數據 (略過 meta rows)
        # 注意：我們現在手動過濾掉字串列，確保只留下數值資料
        df_data = pd.read_csv(csv_path, header=0, low_memory=False)
        df_data = df_data[pd.to_numeric(df_data['Site'], errors='coerce').notna()].copy()
        
        # 萃取所有帶有 '_Main.' 的測試欄位
        feature_cols = [col for col in df_data.columns if '_Main.' in col]
        
        # 2. 只有在處理第一片 Wafer 時，建立全域 Limit 字典
        if not global_limits and not high_row.empty and not low_row.empty:
            print("🔍 正在建立特徵標準化基準 (Physical Limits)...")
            for col in feature_cols:
                try:
                    high_val = float(high_row.iloc[0][col])
                    low_val = float(low_row.iloc[0][col])
                    # 避免上下限相同導致分母為 0
                    if pd.notna(high_val) and pd.notna(low_val) and high_val != low_val:
                        global_limits[col] = {"high": high_val, "low": low_val}
                except (ValueError, TypeError):
                    continue
                    
            # 將 Limits 存成 JSON，一箭雙鵰：
            # (1) 給心靈的前端畫紅線用
            # (2) 給 Edge 端 anomaly_detector 做即時 Normalization 用
            os.makedirs("models", exist_ok=True)
            with open("models/global_limits.json", "w", encoding="utf-8") as f:
                json.dump(global_limits, f, indent=2)
            print(f"✅ 成功擷取 {len(global_limits)} 個測試項目的上下限，已儲存至 models/global_limits.json")

        # 3. 完美特徵標準化 (Min-Max Scaling based on Physical Limits)
        # 將原始資料轉為 float
        df_data[feature_cols] = df_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        for col, limits in global_limits.items():
            if col in df_data.columns:
                h, l = limits['high'], limits['low']
                # 正規化公式：(Value - Low) / (High - Low)
                df_data[col] = (df_data[col] - l) / (h - l)
        
        # 4. 新增 ML 標籤
        df_data['Wafer_ID'] = i
        df_data['Label'] = wafer_info[i]["label"]
        df_data['Anomaly_Type'] = wafer_info[i]["type"]
        
        all_data.append(df_data)
        print(f"✅ 成功處理與標準化 {file_name:25s} | 狀態: {wafer_info[i]['type']}")

    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"\n🎉 資料整併與物理標準化完成！共 {len(final_df)} 筆數據，已儲存至 {output_file}")

if __name__ == "__main__":
    prepare_data()