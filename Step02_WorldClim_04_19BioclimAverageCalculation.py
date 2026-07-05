import os
import time
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from numba import njit, prange

# ============================== 用户配置区域 ==============================
INPUT_FOLDER = r"L:\Bioclim_1km_peryear"
OUTPUT_FOLDER = r"L:\Bioclim_1km_avg"
START_YEAR = 2000
END_YEAR   = 2021

# 如果你担心内存，把 4096 改成 2048 会更稳
BLOCK_SIZE = 2048
# ============================== 配置结束 ==============================

# GDAL I/O 优化（可选）
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "2048")  # MB

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bioclim_average_optimized.log"),
              logging.StreamHandler()]
)
logger = logging.getLogger()

def get_raster_metadata():
    sample_file = os.path.join(INPUT_FOLDER, "2000", "wc2.1_1km_bioclim01_2000.tif")
    with rasterio.open(sample_file) as src:
        return {
            'driver': 'GTiff',
            'height': src.height,
            'width': src.width,
            'transform': src.transform,
            'crs': src.crs,
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'tiled': True,
            'BIGTIFF': 'IF_SAFER',
            'NUM_THREADS': 'ALL_CPUS',
            # 不手动设置 nodata，这里延续逐年文件的处理方式（NaN）
        }

@njit(parallel=True)
def nanmean_stack(stack):
    N, H, W = stack.shape
    out = np.empty((H, W), dtype=np.float32)
    for r in prange(H):
        for c in range(W):
            s = 0.0
            cnt = 0
            for n in range(N):
                v = stack[n, r, c]
                if not np.isnan(v):
                    s += v
                    cnt += 1
            if cnt > 0:
                out[r, c] = s / cnt
            else:
                out[r, c] = np.nan
    return out

def calculate_average_parallel(bio_idx, years, metadata, output_path):
    height, width = metadata['height'], metadata['width']

    ncols = (width  + BLOCK_SIZE - 1) // BLOCK_SIZE
    nrows = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = nrows * ncols

    logger.info(f"开始计算 BIO{bio_idx} 的多年平均, 总块数: {total_blocks}")

    year_datasets = []
    try:
        for y in years:
            fp = os.path.join(INPUT_FOLDER, str(y), f"wc2.1_1km_bioclim{bio_idx}_{y}.tif")
            if not os.path.exists(fp):
                logger.warning(f"缺文件: {fp}，将跳过该年")
                year_datasets.append(None)
            else:
                year_datasets.append(rasterio.open(fp))

        with rasterio.open(output_path, 'w', **metadata) as dst:
            pbar = tqdm(total=total_blocks, desc=f"BIO{bio_idx} 块进度")
            for row in range(nrows):
                for col in range(ncols):
                    yoff = row * BLOCK_SIZE
                    xoff = col * BLOCK_SIZE
                    win_h = min(BLOCK_SIZE, height - yoff)
                    win_w = min(BLOCK_SIZE, width - xoff)
                    window = Window(xoff, yoff, win_w, win_h)

                    stack = np.empty((len(years), win_h, win_w), dtype=np.float32)
                    stack[:] = np.nan

                    yi = 0
                    for ds in year_datasets:
                        if ds is None:
                            yi += 1
                            continue
                        arr = ds.read(1, window=window, masked=True).astype(np.float32)
                        if hasattr(arr, "mask"):
                            stack[yi, :, :] = arr.filled(np.nan)
                        else:
                            stack[yi, :, :] = arr
                        yi += 1

                    mean_block = nanmean_stack(stack)
                    dst.write(mean_block, 1, window=window)
                    pbar.update(1)
            pbar.close()

    finally:
        for ds in year_datasets:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

    logger.info(f"完成 BIO{bio_idx} 的多年平均: {output_path}")
    return True

def main():
    logger.info("==============================")
    logger.info("开始优化计算Bioclim变量多年平均")
    logger.info(f"输入文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info(f"年份范围: {START_YEAR}-{END_YEAR}")
    logger.info(f"块大小: {BLOCK_SIZE}")
    logger.info("==============================")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    years = list(range(START_YEAR, END_YEAR + 1))
    logger.info(f"将处理 {len(years)} 个年份: {years}")

    metadata = get_raster_metadata()
    logger.info(f"栅格尺寸: {metadata['width']}x{metadata['height']}")

    success_count = 0
    failure_count = 0
    start_time = time.time()

    for bio_idx in range(1, 20):
        bio_idx_str = f"{bio_idx:02d}"
        logger.info(f"开始处理 BIO{bio_idx_str} ({bio_idx}/19)")
        output_path = os.path.join(OUTPUT_FOLDER, f"wc2.1_1km_bioclim{bio_idx_str}_avg.tif")
        try:
            ok = calculate_average_parallel(bio_idx_str, years, metadata, output_path)
            if ok:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.error(f"处理 BIO{bio_idx_str} 时出错: {e}")
            failure_count += 1

    total_time = time.time() - start_time
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)

    logger.info("=" * 60)
    logger.info("Bioclim 多年平均计算完成！")
    logger.info(f"总耗时: {int(h)}小时 {int(m)}分钟 {int(s)}秒")
    logger.info(f"成功处理变量: {success_count}/19, 失败: {failure_count}/19")
    logger.info(f"结果保存在: {OUTPUT_FOLDER}")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("处理过程中发生致命错误")
