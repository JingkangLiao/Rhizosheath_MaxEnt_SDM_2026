# -*- coding: utf-8 -*-
# ====== 环境变量（必须在 import 之前）======
import os
os.environ["GDAL_CACHEMAX"] = "2048"                 # 2GB GDAL 缓存
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"  # 加快目录扫描
os.environ["NUMBA_THREADING_LAYER"] = "omp"          # 让 Numba 用 OMP 层并行（Windows 更稳）

import json
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from numba import njit, prange, set_num_threads
import psutil

# 忽略特定警告
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ============================== 用户配置区域 ==============================
# 输入文件夹（重采样后的月度数据，float32，NaN 表示无效）
INPUT_FOLDER = r"L:\monthly_resample2.5minto1000m"

# 输出文件夹（年度 Bioclim 变量，float32，NaN 表示无效）
OUTPUT_FOLDER = r"L:\Bioclim_1km_peryear"

# 有效年份 JSON
VALID_YEARS_JSON = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\2000-2021\version02\valid_years_output\valid_years.json"

# 年份范围
START_YEAR = 2006
END_YEAR   = 2010

# 初始块大小（按需会被动态放大）
BLOCK_SIZE = 1024

# 跨年并行进程数（8核建议 2；想更猛可 3，但易受 I/O 限制）
YEAR_CONCURRENCY = 2
# =======================================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bioclim_calculation_fast.log"), logging.StreamHandler()]
)
logger = logging.getLogger()


# ========================= Numba 计算内核（并行） =========================
@njit(parallel=True, fastmath=False)
def _bioclim_kernel(tmin, tmax, prec, out):
    """
    tmin, tmax, prec: (12, H, W) float32
    out: (19, H, W) float32，计算结果写入其中
    注意：遇到 NaN 会自然传播（与 numpy 行为一致）
    """
    H = tmin.shape[1]
    W = tmin.shape[2]

    for r in prange(H):
        for c in range(W):
            # 取出该像素12个月序列
            tmin_v = np.empty(12, dtype=np.float32)
            tmax_v = np.empty(12, dtype=np.float32)
            prec_v = np.empty(12, dtype=np.float32)
            for m in range(12):
                tmin_v[m] = tmin[m, r, c]
                tmax_v[m] = tmax[m, r, c]
                prec_v[m] = prec[m, r, c]

            # tavg
            tavg_v = np.empty(12, dtype=np.float32)
            for m in range(12):
                tavg_v[m] = (tmin_v[m] + tmax_v[m]) * 0.5

            # ==== BIO1 年平均温度 ====
            bio1 = 0.0
            for m in range(12):
                bio1 += tavg_v[m]
            bio1 /= 12.0

            # ==== BIO2 平均日较差 ====
            diurnal_mean = 0.0
            for m in range(12):
                diurnal_mean += (tmax_v[m] - tmin_v[m])
            diurnal_mean /= 12.0

            # ==== BIO4 温度季节性（样本标准差 ×100）====
            mean_tavg = bio1
            var_tavg = 0.0
            for m in range(12):
                d = tavg_v[m] - mean_tavg
                var_tavg += d * d
            var_tavg /= 11.0  # ddof=1
            std_tavg = np.sqrt(var_tavg)
            bio4 = std_tavg * 100.0

            # ==== BIO5 / BIO6 / BIO7 ====
            bio5 = tmax_v[0]
            bio6 = tmin_v[0]
            for m in range(1, 12):
                if tmax_v[m] > bio5:
                    bio5 = tmax_v[m]
                if tmin_v[m] < bio6:
                    bio6 = tmin_v[m]
            bio7 = bio5 - bio6

            # ==== BIO3 等温性 ====
            if bio7 != 0.0:
                bio3 = (diurnal_mean / bio7) * 100.0
            else:
                bio3 = 0.0

            # ==== BIO12 / 13 / 14 / 15 ====
            sum_prec = 0.0
            max_prec = prec_v[0]
            min_prec = prec_v[0]
            for m in range(12):
                p = prec_v[m]
                sum_prec += p
                if p > max_prec:
                    max_prec = p
                if p < min_prec:
                    min_prec = p
            bio12 = sum_prec
            bio13 = max_prec
            bio14 = min_prec

            mean_prec = sum_prec / 12.0
            if mean_prec != 0.0:
                var_prec = 0.0
                for m in range(12):
                    d = prec_v[m] - mean_prec
                    var_prec += d * d
                var_prec /= 11.0
                std_prec = np.sqrt(var_prec)
                bio15 = (std_prec / mean_prec) * 100.0
            else:
                bio15 = 0.0

            # ==== 滑动季度 ====
            quarter_tavg = np.empty(12, dtype=np.float32)
            quarter_prec = np.empty(12, dtype=np.float32)
            for s in range(12):
                quarter_tavg[s] = (tavg_v[s] + tavg_v[(s + 1) % 12] + tavg_v[(s + 2) % 12]) / 3.0
                quarter_prec[s] = (prec_v[s] + prec_v[(s + 1) % 12] + prec_v[(s + 2) % 12])

            wet_idx = 0
            dry_idx = 0
            warm_idx = 0
            cold_idx = 0
            for s in range(1, 12):
                if quarter_prec[s] > quarter_prec[wet_idx]:
                    wet_idx = s
                if quarter_prec[s] < quarter_prec[dry_idx]:
                    dry_idx = s
                if quarter_tavg[s] > quarter_tavg[warm_idx]:
                    warm_idx = s
                if quarter_tavg[s] < quarter_tavg[cold_idx]:
                    cold_idx = s

            bio8  = quarter_tavg[wet_idx]
            bio9  = quarter_tavg[dry_idx]
            bio10 = quarter_tavg[warm_idx]
            bio11 = quarter_tavg[cold_idx]

            bio16 = quarter_prec[wet_idx]
            bio17 = quarter_prec[dry_idx]
            bio18 = quarter_prec[warm_idx]
            bio19 = quarter_prec[cold_idx]

            # 写出
            out[0,  r, c] = bio1
            out[1,  r, c] = diurnal_mean
            out[2,  r, c] = bio3
            out[3,  r, c] = bio4
            out[4,  r, c] = bio5
            out[5,  r, c] = bio6
            out[6,  r, c] = bio7
            out[7,  r, c] = bio8
            out[8,  r, c] = bio9
            out[9,  r, c] = bio10
            out[10, r, c] = bio11
            out[11, r, c] = bio12
            out[12, r, c] = bio13
            out[13, r, c] = bio14
            out[14, r, c] = bio15
            out[15, r, c] = bio16
            out[16, r, c] = bio17
            out[17, r, c] = bio18
            out[18, r, c] = bio19
# ======================================================================


def calculate_bioclim_vars(tmin, tmax, prec):
    """
    Python 包装器：分配 out，调用 Numba 内核，返回 list[ndarray]
    """
    H, W = tmin.shape[1], tmin.shape[2]
    out = np.empty((19, H, W), dtype=np.float32)
    # 确保连续内存，便于 Numba
    tmin = np.ascontiguousarray(tmin, dtype=np.float32)
    tmax = np.ascontiguousarray(tmax, dtype=np.float32)
    prec = np.ascontiguousarray(prec, dtype=np.float32)
    _bioclim_kernel(tmin, tmax, prec, out)
    return [out[i, :, :] for i in range(19)]


def process_year(year, tmin_files, tmax_files, prec_files, output_folder,
                 block_size=2048, threads_per_proc=None):
    """
    单年处理（可在子进程里跑）
    """
    # 每个进程内设置 Numba 线程数，避免线程爆炸
    if threads_per_proc is not None:
        try:
            set_num_threads(int(threads_per_proc))
            logger.info(f"[{year}] Numba threads per proc = {threads_per_proc}")
        except Exception as e:
            logger.warning(f"[{year}] set_num_threads failed: {e}")

    year_output_folder = os.path.join(output_folder, year)
    os.makedirs(year_output_folder, exist_ok=True)

    try:
        # 元数据模板
        with rasterio.open(tmin_files[0]) as src:
            meta = src.meta.copy()
            height, width = src.shape
            # 追求速度：不压缩写出；使用较大瓦片（SSD 上最快）
            meta.update(count=1, dtype='float32', compress='lzw', tiled=True, nodata=np.nan)

        # 创建 19 个输出文件
        output_files = []
        for i in range(1, 20):
            bio_idx = f"{i:02d}"
            output_path = os.path.join(year_output_folder, f"wc2.1_1km_bioclim{bio_idx}_{year}.tif")
            dst = rasterio.open(output_path, 'w', **meta)
            output_files.append(dst)

        # 打开该年的 12×3 输入
        tmin_srcs = [rasterio.open(fp) for fp in tmin_files]
        tmax_srcs = [rasterio.open(fp) for fp in tmax_files]
        prec_srcs = [rasterio.open(fp) for fp in prec_files]

        # 计算分块数
        ncols = (width + block_size - 1) // block_size
        nrows = (height + block_size - 1) // block_size
        logger.info(f"[{year}] size={width}x{height}, block={block_size}, tiles={ncols}x{nrows}")

        # 分块处理
        for row in tqdm(range(nrows), desc=f"行块 {year}", position=0):
            for col in range(ncols):
                yoff = row * block_size
                xoff = col * block_size
                win_h = min(block_size, height - yoff)
                win_w = min(block_size, width - xoff)
                window = Window(xoff, yoff, win_w, win_h)

                # 读入 12 个月数据
                tmin_arr = np.empty((12, win_h, win_w), dtype=np.float32)
                tmax_arr = np.empty((12, win_h, win_w), dtype=np.float32)
                prec_arr = np.empty((12, win_h, win_w), dtype=np.float32)
                for m in range(12):
                    tmin_arr[m] = tmin_srcs[m].read(1, window=window, out_dtype='float32')
                    tmax_arr[m] = tmax_srcs[m].read(1, window=window, out_dtype='float32')
                    prec_arr[m] = prec_srcs[m].read(1, window=window, out_dtype='float32')

                # 计算 19 个 bioclim
                bioclims = calculate_bioclim_vars(tmin_arr, tmax_arr, prec_arr)

                # 写出
                for idx, arr in enumerate(bioclims):
                    output_files[idx].write(arr, 1, window=window)

        # 关闭
        for dst in output_files:
            dst.close()
        for ds in tmin_srcs + tmax_srcs + prec_srcs:
            ds.close()

        logger.info(f"[{year}] 完成，输出到: {year_output_folder}")
        return True, year

    except Exception as e:
        logger.exception(f"[{year}] 处理出错: {e}")
        try:
            for dst in output_files:
                dst.close()
        except Exception:
            pass
        try:
            for ds in tmin_srcs + tmax_srcs + prec_srcs:
                ds.close()
        except Exception:
            pass
        return False, year


def main():
    # 动态放大 BLOCK_SIZE（看可用内存）
    mem = psutil.virtual_memory().available / (1024**3)
    global BLOCK_SIZE
    if mem > 40:
        BLOCK_SIZE = max(BLOCK_SIZE, 4096)
    elif mem > 20:
        BLOCK_SIZE = max(BLOCK_SIZE, 2048)
    else:
        BLOCK_SIZE = max(BLOCK_SIZE, 1024)

    logger.info("==============================")
    logger.info("开始计算年度 Bioclim 变量（Turbo）")
    logger.info(f"输入:  {INPUT_FOLDER}")
    logger.info(f"输出:  {OUTPUT_FOLDER}")
    logger.info(f"年份:  {START_YEAR}-{END_YEAR}")
    logger.info(f"块大小: {BLOCK_SIZE}")
    logger.info(f"年并行进程: {YEAR_CONCURRENCY}")
    logger.info("==============================")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 加载有效年份列表，并把路径重定向到 INPUT_FOLDER
    if not os.path.exists(VALID_YEARS_JSON):
        logger.error(f"有效年份 JSON 不存在: {VALID_YEARS_JSON}")
        return

    with open(VALID_YEARS_JSON, "r", encoding="utf-8") as f:
        all_valid_years = json.load(f)

    valid_years = {}
    for year, data in all_valid_years.items():
        yi = int(year)
        if START_YEAR <= yi <= END_YEAR:
            valid_years[year] = {
                'tmin': [os.path.join(INPUT_FOLDER, os.path.basename(f)) for f in data['tmin']],
                'tmax': [os.path.join(INPUT_FOLDER, os.path.basename(f)) for f in data['tmax']],
                'prec': [os.path.join(INPUT_FOLDER, os.path.basename(f)) for f in data['prec']],
            }

    if not valid_years:
        logger.error("指定年份范围内没有可用数据。")
        return

    years_sorted = sorted(valid_years.keys())
    logger.info(f"将处理 {len(years_sorted)} 个年份: {', '.join(years_sorted)}")

    # 线程配额：把 CPU 合理分给每个进程
    total_cpu = os.cpu_count() or 8
    threads_per_proc = max(1, total_cpu // max(1, YEAR_CONCURRENCY))

    success, failure = 0, 0
    # 跨年并行
    with ProcessPoolExecutor(max_workers=YEAR_CONCURRENCY) as ex:
        futures = []
        for year in years_sorted:
            files = valid_years[year]
            futures.append(
                ex.submit(
                    process_year,
                    year,
                    files['tmin'], files['tmax'], files['prec'],
                    OUTPUT_FOLDER,
                    BLOCK_SIZE,
                    threads_per_proc
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="年份进度"):
            ok, y = fut.result()
            if ok:
                success += 1
            else:
                failure += 1

    logger.info("=" * 60)
    logger.info("年度 Bioclim 计算完成")
    logger.info(f"成功年份: {success}，失败年份: {failure}")
    logger.info(f"结果目录: {OUTPUT_FOLDER}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("处理过程中发生致命错误")
