import os
import logging

import numpy as np
import rasterio

# ========================
# 配置日志
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # ===== 输入路径 =====
    median_path = r"L:\FutureClim\Ensemble_SSP245\Rhizosheath_SSP245_median.asc"
    sd_path     = r"L:\FutureClim\Ensemble_SSP245\Rhizosheath_SSP245_sd.asc"

    # 阈值（P10）
    threshold = 0.2732

    # ===== 输出路径 =====
    output_dir = r"L:\FutureClim\Ensemble_SSP245"
    os.makedirs(output_dir, exist_ok=True)

    # 这里先写成 GeoTIFF，方便后续在 ArcGIS / QGIS 里直接用
    # 如果你一定要 ASCII，可以把后面的 driver 改成 "AAIGrid"，扩展名改成 .asc
    snr_out_path = os.path.join(
        output_dir,
        f"Rhizosheath_SSP245_SNR_thr_{threshold:.5f}.tif"
    )

    logger.info("====== 计算 SNR map (仅基于 GCM 间 SD) ======")
    logger.info(f"median: {median_path}")
    logger.info(f"sd    : {sd_path}")
    logger.info(f"阈值 T = {threshold}")
    logger.info(f"SNR 输出将保存到: {snr_out_path}")

    # ===== 打开栅格 =====
    if not os.path.exists(median_path):
        raise FileNotFoundError(f"找不到 median 栅格: {median_path}")
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"找不到 sd 栅格: {sd_path}")

    src_med = rasterio.open(median_path)
    src_sd  = rasterio.open(sd_path)

    # 尺寸 / 空间参考一致性检查
    if (src_med.width != src_sd.width) or (src_med.height != src_sd.height):
        raise ValueError("median 与 sd 的尺寸不一致")
    if src_med.transform != src_sd.transform:
        raise ValueError("median 与 sd 的 transform 不一致")
    if src_med.crs != src_sd.crs:
        raise ValueError("median 与 sd 的 CRS 不一致")

    height, width = src_med.height, src_med.width

    # nodata
    nod_med = src_med.nodata
    nod_sd  = src_sd.nodata
    if nod_med is None:
        nod_med = -9999.0
        logger.warning(f"median 未定义 nodata，默认使用 {nod_med}")
    if nod_sd is None:
        nod_sd = -9999.0
        logger.warning(f"sd 未定义 nodata，默认使用 {nod_sd}")

    # ===== 输出 profile =====
    profile_snr = src_med.profile.copy()
    profile_snr.update(
        dtype="float32",
        nodata=-9999.0,
        driver="GTiff"   # 如果想直接输出 ASCII 改成 "AAIGrid"，并把 snr_out_path 后缀改为 .asc
    )

    # 一些统计量（全图的 min / max SNR）
    global_min = np.inf
    global_max = -np.inf

    # 计算参数
    block_height = 128
    eps = 1e-6  # 防止除 0

    with rasterio.open(snr_out_path, "w", **profile_snr) as dst_snr:
        for row_start in range(0, height, block_height):
            row_stop = min(height, row_start + block_height)
            window = ((row_start, row_stop), (0, width))

            med_block = src_med.read(1, window=window).astype("float32")
            sd_block  = src_sd.read(1, window=window).astype("float32")

            # 处理 nodata -> nan
            med_block[med_block == nod_med] = np.nan
            sd_block[sd_block == nod_sd] = np.nan

            valid_mask = (~np.isnan(med_block)) & (~np.isnan(sd_block))

            # 初始化 SNR 块
            snr_block = np.full_like(med_block, np.nan, dtype="float32")

            if np.any(valid_mask):
                # 距离阈值
                delta_block = np.abs(med_block - threshold)

                # SNR = |μ - T| / (sd + eps)
                snr_block[valid_mask] = delta_block[valid_mask] / (
                    sd_block[valid_mask] + eps
                )

                # 更新全图统计
                block_valid_snr = snr_block[valid_mask]
                if block_valid_snr.size > 0:
                    bmin = float(np.nanmin(block_valid_snr))
                    bmax = float(np.nanmax(block_valid_snr))
                    global_min = min(global_min, bmin)
                    global_max = max(global_max, bmax)

            # 输出：nan -> nodata
            snr_block_out = snr_block.copy()
            snr_block_out[np.isnan(snr_block_out)] = profile_snr["nodata"]

            dst_snr.write(snr_block_out, 1, window=window)

            logger.info(f"已处理行 {row_start}–{row_stop}")

    src_med.close()
    src_sd.close()

    if global_min is np.inf:
        logger.warning("全图没有任何有效 SNR 像元")
    else:
        logger.info(f"SNR 全图范围: min={global_min:.3f}, max={global_max:.3f}")

    logger.info("SNR map 计算完成。")
    logger.info("注意：该 SNR 仅反映 GCM 间不确定性下，对阈值分类的稳健性。")


if __name__ == "__main__":
    main()
