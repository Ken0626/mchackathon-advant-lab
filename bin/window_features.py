"""
window_features.py
==================
訓練端與 Edge 推論端「唯一」的特徵計算來源。

設計原則：訓練時算特徵的程式碼，和機台上算特徵的程式碼，必須是同一份。
上一版之所以 80/80 全紅，就是因為這兩邊不一致（訓練吃正規化值、推論吃原始值）。
把它抽成一個模組，物理上杜絕這件事再發生。

只依賴 numpy / pandas，Python 3.10 可用。
"""

import numpy as np
import pandas as pd

# ---- 視窗參數（訓練與推論共用，改這裡兩邊一起改）----
WINDOW_SIZE = 40   # 視窗大小：40 顆 die = 4 個 site 各 10 個輪次
EVAL_EVERY = 4    # 每 4 顆 die（= 一個完整 touchdown）評估一次
ROLL = 10   # 計算「標準差趨勢」時的內部滾動視窗
SITES = (1, 2, 3, 4)

# 四個統計量的名稱，順序等同 window_stats() 回傳的第 0 軸
STAT_NAMES = ("std", "slope", "rstd_slope", "site_range")

# 每個統計量對應到的異常型態（產生告警文字用）
STAT_TO_ANOMALY = {
    "std": "Measure Value Shift / Variation",
    "slope": "Mean Trend Up/Down",
    "rstd_slope": "Stdev Trend Up/Down",
    "site_range": "Site to Site Unbalance",
}


def window_stats(values: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """
    把一個視窗的原始量測值壓成 4 個「每測項一個數字」的統計量。

    values : (N, M) 視窗內 N 顆 die × M 個測項的原始物理值
    sites  : (N,)   每顆 die 的 site 編號
    return : (4, M) 依序為 std / slope / rstd_slope / site_range
    """
    values = np.asarray(values, dtype=np.float64)
    sites = np.asarray(sites)
    n = values.shape[0]

    # 1) std：視窗內離散程度。抓「量測值跳動變大」
    std = values.std(axis=0)

    # 2) slope：對 die 順序做一次線性迴歸的斜率 × 視窗長度
    #    = 整個視窗從頭到尾的漂移量。抓 Mean Trend Up / Down
    idx = np.arange(n, dtype=np.float64)
    idx_c = idx - idx.mean()
    denom = (idx_c ** 2).sum()
    slope = (idx_c[:, None] * values).sum(axis=0) / denom * n

    # 3) rstd_slope：滾動標準差本身的斜率。抓 Stdev Trend Up / Down
    roll = pd.DataFrame(values).rolling(ROLL).std().dropna().values
    if len(roll) >= 2:
        j = np.arange(len(roll), dtype=np.float64)
        j_c = j - j.mean()
        rstd_slope = (j_c[:, None] * roll).sum(axis=0) / (j_c ** 2).sum() * len(roll)
    else:
        rstd_slope = np.zeros_like(std)

    # 4) site_range：四個 site 各自平均值的極差。抓 Site to Site Unbalance
    site_means = []
    for s in SITES:
        m = sites == s
        site_means.append(values[m].mean(axis=0) if m.sum() >= 2 else values.mean(axis=0))
    site_means = np.stack(site_means)
    site_range = site_means.max(axis=0) - site_means.min(axis=0)

    return np.stack([std, slope, rstd_slope, site_range])


def anomaly_scores(stats: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                   z_cut: float = 4.0) -> np.ndarray:
    """
    把 (4, M) 的統計量壓成 4 個純量分數。

    分數定義 = 「這個統計量上，有百分之幾的測項偏離正常基準超過 z_cut 個標準差」。

    為什麼用「極端測項的比例」而不是「平均偏移量」：
    實測發現異常是注入在一小撮測項上（幾十到上百個），不是全部 3036 個一起漂。
    取平均會被 2900 個正常測項稀釋掉，訊號完全消失；數極端值的比例才抓得到。
    """
    z = np.abs((stats - mu) / sd)
    return (z > z_cut).mean(axis=1) * 100.0


def top_offenders(stats: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                  stat_idx: int, columns, k: int = 3):
    """回傳該統計量上偏離最嚴重的前 k 個測項名稱與 z 值，用來寫人看得懂的告警理由。"""
    z = (stats[stat_idx] - mu[stat_idx]) / sd[stat_idx]
    order = np.argsort(-np.abs(z))[:k]
    return [(columns[i], float(z[i])) for i in order]
