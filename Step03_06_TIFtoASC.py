import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# --- 环境优化（GDAL 多线程 & 缓存）---
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "2048")          # MB
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")   # C 端多线程

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# --- 配置 ---
INPUT_DIR    = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version06_260128"
ALIGNED_DIR  = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version06_260128\final"
OUTPUT_DIR   = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version06_260128\ASC"

REFERENCE_FILE   = os.path.join(INPUT_DIR, "BIO01.tif")  # 参考几何
VEG_FILE_NAME    = "GLC_FCS1000_Global_2023_aligned.tif"                   # 植被文件（分类）
VEG_SPECIAL_NODATA = {0, 1, 210}                                           # 植被额外空值
NODATA_VALUE     = -9999
BLOCK_ROWS       = 512                       # 转 ASC 时每块行数（适中更快）
MAX_WORKERS_IO   = max(2, (os.cpu_count() or 8) // 2)   # 并发文件数（别把 SSD 压爆）

# ---------- 工具 ----------
def _safe_src_nodata(src):
    """稳健获取源 nodata；若缺失且为 float32，兜底为常见哨兵 -3.4028235e+38。"""
    nd = src.nodata
    if nd is not None:
        return nd
    dt = np.dtype(src.dtypes[0])
    if dt.kind == 'f' and dt.itemsize == 4:
        return -3.4028235e+38
    return None

def _read_reference_geometry(ref_path):
    with rasterio.open(ref_path) as ref:
        return ref.transform, ref.width, ref.height, ref.crs

# ---------- 第 1 阶段：并行重采样/对齐 ----------
def reproject_one(src_path, ref_geometry, out_dir):
    """将 src 对齐到参考几何；返回(成功与否, 文件名, 错误信息或'OK')"""
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(src_path))
        if os.path.exists(out_path):
            return True, src_path, "skip"

        ref_transform, ref_w, ref_h, ref_crs = ref_geometry
        with rasterio.open(src_path) as src:
            is_veg = (os.path.basename(src_path) == VEG_FILE_NAME)
            src_nd  = _safe_src_nodata(src)
            resamp  = Resampling.nearest if is_veg else Resampling.bilinear
            out_dt  = 'int32' if is_veg else 'float32'

            profile = {
                'driver': 'GTiff',
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_w,
                'height': ref_h,
                'count': 1,
                'dtype': out_dt,
                'tiled': True,
                'compress': 'lzw',
                'BIGTIFF': 'IF_SAFER',
                'nodata': NODATA_VALUE
            }
            num_threads = max(1, (os.cpu_count() or 8) - 1)

            with rasterio.open(out_path, 'w', **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nd,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    dst_nodata=NODATA_VALUE,
                    resampling=resamp,
                    init_dest_nodata=True,
                    num_threads=num_threads  # 必须是整数
                )
        return True, src_path, "OK"
    except Exception as e:
        return False, src_path, str(e)

def reproject_all(input_dir, ref_path, out_dir):
    ref_geom = _read_reference_geometry(ref_path)
    files = glob.glob(os.path.join(input_dir, "*.tif"))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as ex:
        futs = {ex.submit(reproject_one, fp, ref_geom, out_dir): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="对齐重采样(并行)"):
            results.append(fut.result())
    # 日志汇总
    ok = sum(1 for r in results if r[0] and r[2] != "skip")
    skip = sum(1 for r in results if r[0] and r[2] == "skip")
    fail = [r for r in results if not r[0]]
    print(f"对齐完成：成功 {ok}，跳过 {skip}，失败 {len(fail)}")
    if fail:
        for _, fp, msg in fail[:8]:
            print(f"  失败: {os.path.basename(fp)} -> {msg}")
    return len(fail) == 0

# ---------- 第 2 阶段：并行 TIF → ASC ----------
def tif_to_asc_one(tif_path, out_dir):
    """单文件转 ASC（块写 + np.savetxt），返回(成功与否, 文件名, 错误或'OK')"""
    try:
        os.makedirs(out_dir, exist_ok=True)
        asc_path = os.path.join(out_dir, os.path.splitext(os.path.basename(tif_path))[0] + ".asc")
        if os.path.exists(asc_path):
            return True, tif_path, "skip"

        is_veg = (os.path.basename(tif_path) == VEG_FILE_NAME)

        with rasterio.open(tif_path) as src, open(asc_path, "w", encoding="utf-8", newline="\n") as f:
            H, W = src.height, src.width
            T = src.transform
            # 头部从 transform 精确推导
            xllcorner = float(T.c)
            yllcorner = float(T.f + H * T.e)     # e 为负，向下到左下角
            cellsize  = float(abs(T.a))

            # 写头部
            f.write(f"ncols         {W}\n")
            f.write(f"nrows         {H}\n")
            f.write(f"xllcorner     {xllcorner:.10f}\n")
            f.write(f"yllcorner     {yllcorner:.10f}\n")
            f.write(f"cellsize      {cellsize:.10f}\n")
            f.write(f"NODATA_value  {int(NODATA_VALUE)}\n")

            # 行块写
            n_blocks = (H + BLOCK_ROWS - 1) // BLOCK_ROWS
            fmt = "%d" if is_veg else "%.6f"      # 分类整数；连续保留 6 位小数

            for bi in range(n_blocks):
                start = bi * BLOCK_ROWS
                rows  = min(BLOCK_ROWS, H - start)
                window = Window(0, start, W, rows)
                # 掩膜读取（掩膜/NaN 统一处理）
                arrm = src.read(1, window=window, masked=True)

                # 植被：额外把 {0,1,210} 视为空值
                if is_veg:
                    data = arrm.filled(NODATA_VALUE)
                    if VEG_SPECIAL_NODATA:
                        data = np.where(np.isin(data, list(VEG_SPECIAL_NODATA)), NODATA_VALUE, data)
                    data = data.astype(np.int32, copy=False)
                else:
                    data = arrm.filled(NODATA_VALUE).astype(np.float32, copy=False)

                # 清理异常值
                bad = ~np.isfinite(data) | (np.abs(data).astype("float64") > 1e38)
                if bad.any():
                    data[bad] = NODATA_VALUE

                # 直接一次写出整块（极大减少 Python 循环）
                np.savetxt(f, data, fmt=fmt, delimiter=" ")

        return True, tif_path, "OK"
    except Exception as e:
        return False, tif_path, str(e)

def tif_to_asc_all(aligned_dir, out_dir):
    files = glob.glob(os.path.join(aligned_dir, "*.tif"))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IO) as ex:
        futs = {ex.submit(tif_to_asc_one, fp, out_dir): fp for fp in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="TIF→ASC(并行)"):
            results.append(fut.result())
    ok = sum(1 for r in results if r[0] and r[2] != "skip")
    skip = sum(1 for r in results if r[0] and r[2] == "skip")
    fail = [r for r in results if not r[0]]
    print(f"ASC 转换完成：成功 {ok}，跳过 {skip}，失败 {len(fail)}")
    if fail:
        for _, fp, msg in fail[:8]:
            print(f"  失败: {os.path.basename(fp)} -> {msg}")
    return len(fail) == 0

# ---------- 主流程 ----------
if __name__ == "__main__":
    # 1) 并行重采样/对齐到参考
    ok1 = reproject_all(INPUT_DIR, REFERENCE_FILE, ALIGNED_DIR)

    # 2) 并行 TIF→ASC（对齐后的目录）
    ok2 = tif_to_asc_all(ALIGNED_DIR, OUTPUT_DIR)

    # 3) 简要提示
    if ok1 and ok2:
        print("\n全部完成 ✅")
    else:
        print("\n部分失败，请上面日志排查 ❗")
