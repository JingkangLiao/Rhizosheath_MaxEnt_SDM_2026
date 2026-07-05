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
    base_dir = (
        r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt"
        r"\c.LocationMaxEntFigure\Result_0128_Currently_AIinModel_Final"
    )

    # 输入文件
    avg_path    = os.path.join(base_dir, "Rhizosheath_avg.asc")
    std_path    = os.path.join(base_dir, "Rhizosheath_stddev.asc")

    # 输出文件
    cv_out_path = os.path.join(base_dir, "Rhizosheath_CV.asc")

    logger.info("====== 计算 CV map: stddev / avg ======")
    logger.info(f"平均值栅格: {avg_path}")
    logger.info(f"标准差栅格: {std_path}")
    logger.info(f"输出 CV 栅格: {cv_out_path}")

    if not os.path.exists(avg_path):
        raise FileNotFoundError(f"找不到 Rhizosheath_avg.asc: {avg_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"找不到 Rhizosheath_stddev.asc: {std_path}")

    # 打开两个栅格
    src_avg = rasterio.open(avg_path)
    src_std = rasterio.open(std_path)

    # 尺寸 / 空间参考检查
    if src_avg.width != src_std.width or src_avg.height != src_std.height:
        raise ValueError("avg 与 stddev 的尺寸不一致")
    if src_avg.transform != src_std.transform:
        raise ValueError("avg 与 stddev 的 transform 不一致")
    if src_avg.crs != src_std.crs:
        raise ValueError("avg 与 stddev 的 CRS 不一致")

    height, width = src_avg.height, src_avg.width

    # nodata
    nod_avg = src_avg.nodata
    nod_std = src_std.nodata
    if nod_avg is None:
        nod_avg = -9999.0
        logger.warning(f"avg 未定义 nodata，默认使用 {nod_avg}")
    if nod_std is None:
        nod_std = -9999.0
        logger.warning(f"std 未定义 nodata，默认使用 {nod_std}")

    # 输出 profile
    profile_cv = src_avg.profile.copy()
    profile_cv.update(
        driver="AAIGrid",   # 输出 ESRI ASCII .asc
        dtype="float32",
        nodata=-9999.0
    )

    # 如果 profile 中有不被 AAIGrid 接受的参数，可以安全删除
    for key in ["tiled", "compress", "interleave", "blockxsize", "blockysize"]:
        profile_cv.pop(key, None)

    # 统计 CV 的全图范围
    global_min = np.inf
    global_max = -np.inf

    block_height = 128
    eps = 1e-6  # 防止除 0

    with rasterio.open(cv_out_path, "w", **profile_cv) as dst_cv:
        for row_start in range(0, height, block_height):
            row_stop = min(height, row_start + block_height)
            window = ((row_start, row_stop), (0, width))

            avg_block = src_avg.read(1, window=window).astype("float32")
            std_block = src_std.read(1, window=window).astype("float32")

            # nodata -> nan
            avg_block[avg_block == nod_avg] = np.nan
            std_block[std_block == nod_std] = np.nan

            valid_mask = (~np.isnan(avg_block)) & (~np.isnan(std_block))

            cv_block = np.full_like(avg_block, np.nan, dtype="float32")

            if np.any(valid_mask):
                # 仅在 avg 不接近 0 的地方计算 CV
                safe_mask = valid_mask & (np.abs(avg_block) > eps)

                cv_block[safe_mask] = std_block[safe_mask] / avg_block[safe_mask]

                # 更新全图统计
                valid_cv_vals = cv_block[safe_mask]
                if valid_cv_vals.size > 0:
                    bmin = float(np.nanmin(valid_cv_vals))
                    bmax = float(np.nanmax(valid_cv_vals))
                    global_min = min(global_min, bmin)
                    global_max = max(global_max, bmax)

            # 输出：nan -> nodata
            cv_block_out = cv_block.copy()
            cv_block_out[np.isnan(cv_block_out)] = profile_cv["nodata"]

            dst_cv.write(cv_block_out, 1, window=window)
            logger.info(f"已处理行 {row_start}–{row_stop}")

    src_avg.close()
    src_std.close()

    if global_min is np.inf:
        logger.warning("全图没有任何有效 CV 像元。")
    else:
        logger.info(f"CV 全图范围: min={global_min:.4f}, max={global_max:.4f}")

    logger.info("CV map 计算完成。")
    logger.info("说明：CV = stddev / avg，未乘以 100。若需要百分比，可在代码中乘以 100。")


if __name__ == "__main__":
    main()
