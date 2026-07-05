import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as crs_transform

# ------------------- 路径配置 -------------------
CSV_PATH = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\Rhizosheath_occurrence_CroRemoval_RepliRemoval.csv"
RASTER_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021"
OUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\i.TwoFactorsFigure"
OUT_PATH = os.path.join(OUT_DIR, "2factor.csv")

# ------------------- 工具函数 -------------------
def find_raster(base_dir, preferred_names, fallback_patterns):
    """
    在目录中优先按 exact name 查找，其次按通配模式查找（不区分大小写）。
    返回找到的首个文件路径；找不到则抛异常。
    """
    # 先 exact
    for name in preferred_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    # 再 pattern（大小写不敏感）
    candidates = []
    for pat in fallback_patterns:
        for p in glob.glob(os.path.join(base_dir, pat)):
            candidates.append(p)
        # 再加一遍大小写不敏感搜索
        for p in glob.glob(os.path.join(base_dir, pat.replace(".tif", ".TIF"))):
            candidates.append(p)
    if candidates:
        # 选最短/最匹配的一个
        candidates.sort(key=lambda x: (len(os.path.basename(x)), x))
        return candidates[0]
    raise FileNotFoundError(f"未找到匹配的栅格：首选={preferred_names}，备选模式={fallback_patterns}")

def sample_raster_at_points(raster_path, lons_deg, lats_deg):
    """
    给定经纬度（WGS84 度），从 raster 中提取像元值。
    - 自动坐标转换到栅格 CRS
    - 越界/NoData -> np.nan
    返回：np.ndarray(float32)
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        # 将 WGS84 坐标转换到栅格 CRS（若本身就是 WGS84 则等价）
        src_crs = "EPSG:4326"
        dst_crs = src.crs if src.crs is not None else "EPSG:4326"
        xs, ys = crs_transform(src_crs, dst_crs, lons_deg.tolist(), lats_deg.tolist())

        # 先做边界过滤，越界点直接置 NaN，避免 sample 出错
        b = src.bounds
        inb = (np.array(xs) >= b.left) & (np.array(xs) <= b.right) & \
              (np.array(ys) >= b.bottom) & (np.array(ys) <= b.top)

        vals = np.full(len(xs), np.nan, dtype=np.float32)
        if inb.any():
            coords_in = [(float(x), float(y)) for x, y, k in zip(xs, ys, inb) if k]
            # 使用 sample 批量抽样
            # 注意：sample 返回形如 [[v], [v], ...]
            samples = list(src.sample(coords_in))
            arr = np.array(samples, dtype=np.float32).reshape(-1)
            # NoData -> NaN
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            # 回填到对应位置
            vals[inb] = arr

        # 清理异常值
        vals = np.where(np.isfinite(vals), vals, np.nan).astype(np.float32)
        return vals

# ------------------- 主流程 -------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读 CSV（容错列名）
    df = pd.read_csv(CSV_PATH)
    lower_cols = {c.lower(): c for c in df.columns}
    if 'latitude' not in lower_cols or 'longitude' not in lower_cols:
        raise ValueError("CSV 中需要包含列 'latitude' 与 'longitude'（不区分大小写）。")
    lat_col = lower_cols['latitude']
    lon_col = lower_cols['longitude']

    lats = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
    lons = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
    valid_pts = np.isfinite(lats) & np.isfinite(lons)
    if not valid_pts.any():
        raise ValueError("CSV 中经纬度列没有有效数值。")

    # 2) 找到 BIO01 与 LAYERS_TATAL_N 栅格
    #    - BIO01: 优先 BIO01.tif；备选如 bioclim01*.tif / *BIO01*.tif
    bio01_path = find_raster(
        RASTER_DIR,
        preferred_names=["BIO01.tif"],
        fallback_patterns=["*BIO01*.tif", "*bioclim01*.tif", "*_bioclim01_*.tif"]
    )
    #    - TN: 优先 LAYERS_TATAL_N.tif；兼容 TOTAL/TOTAL_N/Total_N 等大小写
    tn_path = find_raster(
        RASTER_DIR,
        preferred_names=["LAYERS_TATAL_N.tif", "LAYERS_TOTAL_N.tif"],
        fallback_patterns=["*TATAL*N*.tif", "*TOTAL*N*.tif", "*Total*N*.tif", "*_N*.tif"]
    )

    print(f"BIO01 (AMT) 栅格: {bio01_path}")
    print(f"TN 栅格        : {tn_path}")

    # 3) 抽样（只对有效点）
    AMT = np.full(len(df), np.nan, dtype=np.float32)
    TN  = np.full(len(df), np.nan, dtype=np.float32)

    AMT[valid_pts] = sample_raster_at_points(bio01_path, lons[valid_pts], lats[valid_pts])
    TN[valid_pts]  = sample_raster_at_points(tn_path,    lons[valid_pts], lats[valid_pts])

    # 4) 合并进 DataFrame（列名：AMT, TN）
    df['AMT'] = AMT
    df['TN']  = TN

    # 小结信息
    n_total = len(df)
    n_amt = int(np.isfinite(AMT).sum())
    n_tn  = int(np.isfinite(TN).sum())
    print(f"AMT 成功提取: {n_amt}/{n_total}")
    print(f"TN  成功提取: {n_tn}/{n_total}")

    # 5) 保存
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到: {OUT_PATH}")

if __name__ == "__main__":
    main()
