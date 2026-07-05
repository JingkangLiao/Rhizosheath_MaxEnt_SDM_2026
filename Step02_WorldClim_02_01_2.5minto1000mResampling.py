import os
os.environ["GDAL_CACHEMAX"] = "2048"                 # GDAL 缓存 ~2GB
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"  # 加快目录扫描

import json
import logging
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import shutil

# ============================== 用户配置区域 ==============================
# 年份范围
START_YEAR = 2000
END_YEAR   = 2010

# 输入/输出
INPUT_FOLDER  = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\2000-2021\monthly"
OUTPUT_FOLDER = r"L:\monthly_resample2.5minto1000m"
VALID_YEARS_JSON = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\2000-2021\version02\valid_years_output\valid_years.json"

# 目标分辨率（单位：度）
# 0.00833333 度 ≈ 30 arc-sec（~1 km at equator）
TARGET_RESOLUTION = (0.00833333, 0.00833333)

# 自动根据可用内存调整批次大小（只是任务分组用，不是并行线程数）
def auto_batch_size(base=10):
    mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    if mem_gb < 8:
        return max(2, base // 2)
    elif mem_gb > 32:
        return base * 2
    else:
        return base

BATCH_SIZE = auto_batch_size()
# ============================== 配置结束 ==============================


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resample_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def check_disk_space(folder, required_gb=50):
    """
    检查输出盘剩余空间，单位GB
    """
    try:
        usage = shutil.disk_usage(folder)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < required_gb:
            return False, free_gb
        return True, free_gb
    except Exception as e:
        logger.error(f"检查磁盘空间失败: {e}")
        # 如果出错，返回“假定足够”，避免程序直接崩
        return True, 0.0


def resample_file(input_path, output_folder, target_resolution):
    """
    重采样单个 GeoTIFF 到指定分辨率 (target_resolution)，并写出为 float32
    - 保留投影和范围
    - nodata -> NaN
    - 输出不写 nodata 标签到元数据里（实际像元内保留 NaN）
    """
    try:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)

        # 如果已经处理过同名输出，就直接跳过
        if os.path.exists(output_path):
            return True, input_path, "已存在，跳过"

        with rasterio.open(input_path) as src:
            # 根据目标分辨率计算目标变换、大小
            transform, width, height = calculate_default_transform(
                src.crs,
                src.crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=target_resolution
            )

            # 保证是整数，防止后面报 “width/height must be integers”
            width = int(round(width))
            height = int(round(height))

            # 选择插值方式：
            # - 若把分辨率变粗(像素变大)，更合理的是用平均 (Resampling.average)
            # - 若把分辨率变细(像素变小)，用双线性 (Resampling.bilinear)
            src_xres, src_yres = src.res  # 原始分辨率
            tgt_xres, tgt_yres = target_resolution
            if (tgt_xres > src_xres) or (tgt_yres > src_yres):
                resampling_method = Resampling.average
            else:
                resampling_method = Resampling.bilinear

            # 源 nodata，如果缺的话我们自己给个不会出现在真实值范围内的占位
            src_nodata = src.nodata if src.nodata is not None else -9999.0

            # 准备输出元数据
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'width': width,
                'height': height,
                'transform': transform,
                'crs': src.crs,
                'dtype': 'float32',
                'count': 1,
                'tiled': True,
                'compress': 'lzw',       # 如果想更快写盘，可以改成 'NONE'
                'BIGTIFF': 'IF_SAFER',
            })

            # 不在元数据里声明 nodata（避免 GDAL 自动把 NaN 变成极端值）
            if 'nodata' in meta:
                meta.pop('nodata', None)

            # 写目标文件
            with rasterio.open(output_path, 'w', **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    dst_nodata=np.nan,         # 输出内部用 NaN 表示无效
                    resampling=resampling_method,
                    init_dest_nodata=True       # 先把目标填成 NaN
                )

        return True, input_path, "成功"

    except Exception as e:
        return False, input_path, str(e)


def process_year(year_files, output_folder, target_resolution):
    """
    处理一个年份的所有月度文件（tmin/tmax/prec）
    - year_files 是 valid_years[year] 那个 dict，比如包含 ['tmin', 'tmax', 'prec']
    """
    logger = logging.getLogger()
    logger.info(f"开始处理年份: {year_files['year']}")

    # 这一年的所有文件路径
    all_files = year_files['tmin'] + year_files['tmax'] + year_files['prec']

    # 分批（只是为了更好地控制内存/日志）
    batches = [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    total_batches = len(batches)

    success_count = 0
    failure_count = 0
    skip_count    = 0

    for i, batch in enumerate(batches, start=1):
        # 每一批都检查一下磁盘空间
        has_space, free_gb = check_disk_space(output_folder, 10)  # 假设一批最多吃~10GB
        if not has_space:
            logger.error(f"磁盘空间不足! 需要至少10GB，当前可用: {free_gb:.2f}GB")
            logger.error(f"年份 {year_files['year']} 处理被迫中止")
            break

        logger.info(f"处理批次 {i}/{total_batches} ({len(batch)} 个文件)")

        # 用进程池并行这一批（尽量吃满 CPU 的一半，避免IO争用太夸张）
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            futures = {
                executor.submit(
                    resample_file,
                    fp,
                    output_folder,
                    target_resolution
                ): fp
                for fp in batch
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    success, _, message = future.result()
                    if success:
                        if "跳过" in message:
                            skip_count += 1
                        else:
                            success_count += 1
                        logger.info(f"[完成] {os.path.basename(file_path)}: {message}")
                    else:
                        failure_count += 1
                        logger.error(f"[失败] {os.path.basename(file_path)}: {message}")
                except Exception as e:
                    failure_count += 1
                    logger.error(f"[崩溃] {file_path}: {str(e)}")

    logger.info(
        f"年份 {year_files['year']} 完成: 成功 {success_count}, 失败 {failure_count}, 跳过 {skip_count}"
    )
    return success_count, failure_count, skip_count


def main():
    logger.info("==============================")
    logger.info(f"开始处理年份范围: {START_YEAR}-{END_YEAR}")
    logger.info(f"输入文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info(f"目标分辨率(度): {TARGET_RESOLUTION[0]} x {TARGET_RESOLUTION[1]}")
    logger.info(f"批次大小: {BATCH_SIZE} 文件/批")
    logger.info("==============================")

    # 确保输出目录存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 初始磁盘检查
    has_space, free_gb = check_disk_space(OUTPUT_FOLDER, 50)
    if not has_space:
        logger.error(f"磁盘空间不足! 需要至少50GB，当前可用: {free_gb:.2f}GB")
        logger.error("请清理磁盘空间后再运行此脚本")
        return
    logger.info(f"磁盘空间检查通过，可用空间约: {free_gb:.2f} GB")

    # 读取有效年份 JSON
    if not os.path.exists(VALID_YEARS_JSON):
        logger.error(f"有效年份JSON文件不存在: {VALID_YEARS_JSON}")
        return

    with open(VALID_YEARS_JSON, 'r', encoding='utf-8') as f:
        all_valid_years = json.load(f)

    # 选出指定年份的条目
    valid_years = {}
    for year, data in all_valid_years.items():
        year_int = int(year)
        if START_YEAR <= year_int <= END_YEAR:
            valid_years[year] = data
            valid_years[year]['year'] = year  # 方便日志用

    if not valid_years:
        logger.error(f"在有效年份数据中未找到 {START_YEAR}-{END_YEAR} 范围内的年份")
        return

    logger.info(f"找到 {len(valid_years)} 个有效年份: {', '.join(sorted(valid_years.keys()))}")

    total_success = 0
    total_failure = 0
    total_skip    = 0

    # 按年份顺序跑
    for year in sorted(valid_years.keys()):
        s, f, sk = process_year(valid_years[year], OUTPUT_FOLDER, TARGET_RESOLUTION)
        total_success += s
        total_failure += f
        total_skip    += sk

        logger.info("=" * 60)
        logger.info(f"{year} 年处理完成")
        logger.info(f"累计: 成功 {total_success}, 失败 {total_failure}, 跳过 {total_skip}")
        logger.info("=" * 60)

        # 每年后再做一次磁盘检查，防止后面直接把盘写满
        has_space, free_gb = check_disk_space(OUTPUT_FOLDER, 50)
        if not has_space:
            logger.error(f"磁盘空间不足! 需要至少50GB，当前可用: {free_gb:.2f}GB")
            logger.error("请清理磁盘空间后再继续处理下一年")
            break

    # 最终总结
    total_files = total_success + total_failure + total_skip
    logger.info("======== 最终报告 ========")
    logger.info(f"处理年份范围: {START_YEAR}-{END_YEAR}")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功重采样: {total_success}")
    logger.info(f"跳过(已存在): {total_skip}")
    logger.info(f"失败: {total_failure}")

    if total_failure == 0:
        logger.info("🎉 所有文件成功处理！")
    else:
        logger.warning(f"⚠ 有 {total_failure} 个文件处理失败，请查看日志了解详情")

    logger.info("=" * 60)
    logger.info("处理完成。若需调整空间分辨率，请修改 TARGET_RESOLUTION。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("处理过程中发生致命错误")
