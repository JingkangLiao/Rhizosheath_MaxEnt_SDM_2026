# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as crs_transform

# ============== 路径配置 ==============
CSV_IN = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\Rhizosheath_occurrence_CroRemoval_RepliRemoval.csv"
RASTER_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021"
OUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\f.PermutationImportanceFigure_withSoilDatabase"
OUT_CSV = os.path.join(OUT_DIR, "points_with_6vars.csv")

# 变量与文件名映射（列名 -> 文件名）
VAR_FILES = {
    "BIO01": "BIO01.tif",            # 年均温 (°C)
    "BIO11": "BIO11.tif",            # 最冷季度均温 (°C)
    "BIO04": "BIO04.tif",            # 温度季节性 (std * 100, 无量纲)
    "AWC":   "LAYERS_AWC.tif",       # 土壤有效水容量 (mm; HWSD2)
    "TN":    "LAYERS_TATAL_N.tif",   # 土壤总氮 (单位依数据而定, 常见 g/kg)
    "BIO17": "BIO17.tif"             # 最干季度降水 (mm)
}

# ============== 小工具 ==============
def _iter_windows(src, block=1024):
    """为任意栅格生成分块窗口（避免一次性读入整幅）。"""
    for y in range(0, src.height, block):
        h = min(block, src.height - y)
        for x in range(0, src.width, block):
            w = min(block, src.width - x)
            yield Window(x, y, w, h)

def masked_min_max(src):
    """
    分块读取，计算非空(非nodata/非NaN/非±1e30)像元的全局 min/max。
    返回 (min_val, max_val, count_valid)；若无有效像元，返回 (np.nan, np.nan, 0)。
    """
    gmin = np.inf
    gmax = -np.inf
    n_valid = 0
    nd = src.nodata

    for win in _iter_windows(src, block=1024):
        arr = src.read(1, window=win, masked=False).astype(np.float32, copy=False)

        # 构建有效掩膜：非 NaN、非 ±inf、非极端哨兵、且不等于 nodata
        mask = np.isfinite(arr)
        mask &= (np.abs(arr) < 1e30)
        if nd is not None and np.isfinite(nd):
            mask &= (arr != nd)

        if not np.any(mask):
            continue

        block_min = float(np.min(arr[mask]))
        block_max = float(np.max(arr[mask]))
        if block_min < gmin: gmin = block_min
        if block_max > gmax: gmax = block_max
        n_valid += int(mask.sum())

    if n_valid == 0:
        return np.nan, np.nan, 0
    return float(gmin), float(gmax), n_valid

def find_lat_lon_cols(columns):
    """在 CSV 列名中寻找 latitude / longitude（大小写/别名兼容）。"""
    cols = [c.lower() for c in columns]
    lat_candidates = ["latitude", "lat", "y", "y_lat", "ycoord"]
    lon_candidates = ["longitude", "lon", "long", "x", "x_lon", "xcoord"]
    lat_col = next((columns[i] for i,c in enumerate(cols) if c in lat_candidates), None)
    lon_col = next((columns[i] for i,c in enumerate(cols) if c in lon_candidates), None)
    if lat_col is None or lon_col is None:
        raise ValueError("CSV 中未找到经纬度列，请确保包含 'latitude' 和 'longitude'（或常见别名）。")
    return lat_col, lon_col

def sample_raster_at_points(tif_path, lons, lats):
    """
    在 tif_path 中按经纬度采样像元值；自动处理 CRS 差异与 nodata/NaN。
    返回 np.ndarray，长度与点数一致（缺失填 np.nan）。
    """
    with rasterio.open(tif_path) as src:
        # 坐标系转换（CSV 假定为 EPSG:4326）
        if src.crs is None or getattr(src.crs, "is_geographic", False) or \
           (hasattr(src.crs, "to_string") and src.crs.to_string() in ("EPSG:4326", "WGS84", "OGC:CRS84")):
            xs, ys = lons, lats
        else:
            xs, ys = crs_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())

        # 采样
        vals = []
        nd = src.nodata
        for v in src.sample(list(zip(xs, ys))):
            val = float(v[0]) if np.size(v) else np.nan
            if (not np.isfinite(val)) or (np.abs(val) >= 1e30) or (nd is not None and np.isfinite(nd) and val == nd):
                vals.append(np.nan)
            else:
                vals.append(val)
        return np.array(vals, dtype=np.float32)

def ci95_mean(arr):
    """
    计算样本均值的 95% 置信区间（忽略 NaN）。
    n>=2 时返回 (mean, lower, upper, n)；否则返回 (nan, nan, nan, n)。
    优先用 t 分布临界值（若 SciPy 可用），否则用 1.96/2.0 近似。
    """
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (np.nan, np.nan, np.nan, int(n))
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    # t 临界值
    try:
        from scipy.stats import t
        tcrit = float(t.ppf(0.975, df=n-1))
    except Exception:
        # n>=30 用 1.96；小样本保守用 2.0
        tcrit = 1.96 if n >= 30 else 2.0
    lower = mean - tcrit * se
    upper = mean + tcrit * se
    return (mean, lower, upper, int(n))

# ============== 主流程 ==============
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 打印每个栅格的非空范围
    print("\n=== 栅格非空值范围检查（按块统计） ===")
    raster_paths = {}
    for var, fname in VAR_FILES.items():
        tif_path = os.path.join(RASTER_DIR, fname)
        if not os.path.exists(tif_path):
            print(f"❌ 缺失：{var} -> {tif_path}")
            continue
        raster_paths[var] = tif_path
        with rasterio.open(tif_path) as src:
            vmin, vmax, nvalid = masked_min_max(src)
            crs_str = src.crs.to_string() if src.crs else "None"
            print(f"- {var:<5} ({fname}): CRS={crs_str} | 有效像元={nvalid:,}")
            print(f"  值域: min={vmin}, max={vmax}")

    if not raster_paths:
        print("\n❌ 没有可用的 TIF，流程结束。")
        return

    # 2) 读取点位 CSV 并提取像元值
    df = pd.read_csv(CSV_IN)
    lat_col, lon_col = find_lat_lon_cols(df.columns)
    lats = df[lat_col].astype(float).to_numpy()
    lons = df[lon_col].astype(float).to_numpy()

    print(f"\n=== 点位提取：共 {len(df)} 个点 ===")
    for var, tif_path in raster_paths.items():
        vals = sample_raster_at_points(tif_path, lons, lats)
        df[var] = vals
        n_nan = int(np.isnan(vals).sum())
        print(f"- {var:<5} 提取完成：空值 {n_nan} 个")

    # 3) 对每个变量输出 CI95（仅基于提取到的样本）
    print("\n=== 变量样本的 95% 置信区间（基于点位样本的均值） ===")
    for var in VAR_FILES.keys():
        if var not in df.columns:
            continue
        mean, lo, hi, n = ci95_mean(df[var].values)
        print(f"- {var:<5}  n={n:>5} | mean={mean:.6f} | CI95=({lo:.6f}, {hi:.6f})")

    # 4) 导出合并 CSV
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存带环境变量的新CSV：{OUT_CSV}")

if __name__ == "__main__":
    main()
