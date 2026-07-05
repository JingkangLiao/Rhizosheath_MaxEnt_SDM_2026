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
    # ========= Step02: 当前 vs SSP245 中位数 → 0/1/2/3 变化图 =========

    # 1. 输入路径
    current_path = (
        r"L:\FutureClim\585_EC\Rhizosheath_avg.asc"
    )

    median_245_path = (
        r"L:\FutureClim\Ensemble_SSP245\Rhizosheath_SSP245_median.asc"
    )

    # 输出目录与文件
    output_dir = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\version_final_260406\Ensemble_SSP245"
    os.makedirs(output_dir, exist_ok=True)
    out_change_path = os.path.join(output_dir, "Change_SSP245_4class.asc")

    # 阈值（10th percentile training presence）
    threshold = 0.2732
    logger.info(f"使用阈值 T = {threshold} 进行分类")

    # 2. 打开两个栅格
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"找不到当前气候栅格: {current_path}")
    if not os.path.exists(median_245_path):
        raise FileNotFoundError(f"找不到 SSP245 中位栅格: {median_245_path}")

    src_cur = rasterio.open(current_path)
    src_fut = rasterio.open(median_245_path)

    # 检查尺寸与空间参考是否一致
    if src_cur.height != src_fut.height or src_cur.width != src_fut.width:
        raise ValueError(
            f"栅格尺寸不一致: current ({src_cur.height}, {src_cur.width}) "
            f" vs SSP245_median ({src_fut.height}, {src_fut.width})"
        )
    if src_cur.crs != src_fut.crs:
        raise ValueError(f"CRS 不一致: {src_cur.crs} vs {src_fut.crs}")
    if src_cur.transform != src_fut.transform:
        raise ValueError("transform 不一致，无法逐像元对比。")

    height = src_cur.height
    width = src_cur.width
    meta_out = src_cur.meta.copy()

    # nodata 处理
    nodata_cur = src_cur.nodata
    nodata_fut = src_fut.nodata

    # 如果没有定义 nodata，就统一设一个
    if nodata_cur is None:
        nodata_cur = -9999.0
        logger.warning(f"当前栅格未定义 nodata，默认使用 {nodata_cur}")
    if nodata_fut is None:
        nodata_fut = -9999.0
        logger.warning(f"SSP245 中位栅格未定义 nodata，默认使用 {nodata_fut}")

    nodata_out = -9999.0  # 输出的 nodata

    meta_out.update(
        dtype="float32",
        count=1,
        nodata=nodata_out
    )

    # 3. 创建输出栅格，并按块处理
    with rasterio.open(out_change_path, "w", **meta_out) as dst_change:

        block_height = 128  # 每次处理的行数

        for row_start in range(0, height, block_height):
            row_stop = min(height, row_start + block_height)
            window = ((row_start, row_stop), (0, width))

            # 读取当前与未来中位数的块
            cur_block = src_cur.read(1, window=window).astype("float32")
            fut_block = src_fut.read(1, window=window).astype("float32")

            # 识别 nodata → np.nan
            cur_block[cur_block == nodata_cur] = np.nan
            fut_block[fut_block == nodata_fut] = np.nan

            # 有效像元：两个时期都不是 nan
            valid_mask = (~np.isnan(cur_block)) & (~np.isnan(fut_block))

            # 当前 / 未来是否适宜
            cur_suit = valid_mask & (cur_block >= threshold)
            fut_suit = valid_mask & (fut_block >= threshold)

            # 初始化输出块为 nodata
            change_block = np.full(cur_block.shape, nodata_out, dtype="float32")

            # 四类：
            # 0: 都不适宜
            both_unsuit = valid_mask & (~cur_suit) & (~fut_suit)
            change_block[both_unsuit] = 0.0

            # 1: 仅当前适宜（loss）
            cur_only = valid_mask & cur_suit & (~fut_suit)
            change_block[cur_only] = 1.0

            # 2: 仅未来适宜（gain）
            fut_only = valid_mask & (~cur_suit) & fut_suit
            change_block[fut_only] = 2.0

            # 3: 都适宜（stable）
            both_suit = valid_mask & cur_suit & fut_suit
            change_block[both_suit] = 3.0

            # 写入输出
            dst_change.write(change_block, 1, window=window)

            logger.info(
                f"已处理行 {row_start}–{row_stop} / {height} "
                f"(有效像元比例 ~ {valid_mask.mean():.3f})"
            )

    # 4. 关闭输入栅格
    src_cur.close()
    src_fut.close()

    logger.info(
        "Step02 完成：已生成 0/1/2/3 四类变化图 -> "
        f"{out_change_path}\n"
        "分类含义: 0=都不适宜, 1=仅当前适宜, 2=仅未来适宜, 3=都适宜, nodata=任一时期缺值"
    )


if __name__ == "__main__":
    main()
