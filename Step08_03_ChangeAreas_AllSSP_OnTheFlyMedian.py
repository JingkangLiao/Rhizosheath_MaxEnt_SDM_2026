import os
import logging
import csv

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

# ========================
# 配置日志
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_change_stats_for_ssp_on_the_fly(
    ssp_name: str,
    src_cur: rasterio.io.DatasetReader,
    gcm_paths: dict,
    threshold: float,
    block_height: int = 128
):
    """
    对某一个 SSP：
    - 读取 current 与该 SSP 的 3 个 GCM 栅格（按块处理）
    - 每个块内先对 3 个 GCM 做中位数（nanmedian）
    - 再用同一阈值 threshold 做 0/1/2/3 分类（不写出栅格）
    - 返回四类像元数和比例

    分类定义：
      0: 都不适宜（current < T & future_median < T）
      1: 仅当前适宜（current >= T & future_median < T）
      2: 仅未来适宜（current < T & future_median >= T）
      3: 都适宜（current >= T & future_median >= T）
    """

    # 打开 3 个 GCM 栅格
    srcs = {}
    for gcm_name, path in gcm_paths.items():
        logger.info(f"[{ssp_name}] 打开 GCM: {gcm_name} -> {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{ssp_name}] 找不到 GCM 栅格: {path}")
        src = rasterio.open(path)
        srcs[gcm_name] = src

    # 与当前栅格检查尺寸 / 空间参考
    height = src_cur.height
    width = src_cur.width
    crs = src_cur.crs
    transform = src_cur.transform

    nodata_cur = src_cur.nodata
    if nodata_cur is None:
        nodata_cur = -9999.0
        logger.warning(f"[{ssp_name}] 当前栅格未定义 nodata，默认使用 {nodata_cur}")

    nodata_gcm = {}
    for gcm_name, src in srcs.items():
        if src.height != height or src.width != width:
            raise ValueError(
                f"[{ssp_name}] {gcm_name} 的尺寸与当前不一致: "
                f"({src.height}, {src.width}) != ({height}, {width})"
            )
        if src.crs != crs:
            raise ValueError(f"[{ssp_name}] {gcm_name} 的 CRS 不一致: {src.crs} != {crs}")
        if src.transform != transform:
            raise ValueError(f"[{ssp_name}] {gcm_name} 的 transform 不一致。")

        nod = src.nodata
        if nod is None:
            nod = -9999.0
            logger.warning(f"[{ssp_name}] {gcm_name} 未定义 nodata，默认使用 {nod}")
        nodata_gcm[gcm_name] = nod

    # 统计计数器
    n0 = n1 = n2 = n3 = 0
    N_total = 0

    logger.info(f"[{ssp_name}] 开始分块统计，阈值 T = {threshold}")

    for row_start in range(0, height, block_height):
        row_stop = min(height, row_start + block_height)
        window = ((row_start, row_stop), (0, width))

        # 当前块
        cur_block = src_cur.read(1, window=window).astype("float32")
        cur_block[cur_block == nodata_cur] = np.nan

        # 3 个 GCM 块
        gcm_block_list = []
        for gcm_name, src in srcs.items():
            try:
                arr = src.read(1, window=window).astype("float32")
            except RasterioIOError as e:
                logger.error(
                    f"[{ssp_name}] 读取 GCM 出错:\n"
                    f"  GCM = {gcm_name}\n"
                    f"  路径 = {src.name}\n"
                    f"  window 行范围 = {row_start}-{row_stop}\n"
                    f"  原始错误 = {e}"
                )
                # 直接抛出，让程序中止，这样你在控制台能看到完整路径
                raise

            nod = nodata_gcm[gcm_name]
            arr[arr == nod] = np.nan
            gcm_block_list.append(arr)

        # 堆叠 (GCM, rows_block, cols)
        stack = np.stack(gcm_block_list, axis=0)

        # 对 3 个 GCM 逐像元求中位数
        fut_median_block = np.nanmedian(stack, axis=0)

        # 有效像元：当前和未来中位数都不是 nan
        valid_mask = (~np.isnan(cur_block)) & (~np.isnan(fut_median_block))

        if not np.any(valid_mask):
            logger.info(f"[{ssp_name}] 行 {row_start}-{row_stop} 全为无效像元，跳过。")
            continue

        # 当前 / 未来是否适宜
        cur_suit = valid_mask & (cur_block >= threshold)
        fut_suit = valid_mask & (fut_median_block >= threshold)

        # 四类
        both_unsuit = valid_mask & (~cur_suit) & (~fut_suit)
        cur_only = valid_mask & cur_suit & (~fut_suit)
        fut_only = valid_mask & (~cur_suit) & fut_suit
        both_suit = valid_mask & cur_suit & fut_suit

        # 统计
        n0_block = int(np.sum(both_unsuit))
        n1_block = int(np.sum(cur_only))
        n2_block = int(np.sum(fut_only))
        n3_block = int(np.sum(both_suit))
        N_block = int(np.sum(valid_mask))

        n0 += n0_block
        n1 += n1_block
        n2 += n2_block
        n3 += n3_block
        N_total += N_block

        logger.info(
            f"[{ssp_name}] 行 {row_start}-{row_stop}: "
            f"有效={N_block}, 0={n0_block}, 1={n1_block}, 2={n2_block}, 3={n3_block}"
        )

    # 关闭 GCM 栅格
    for src in srcs.values():
        src.close()

    if N_total == 0:
        raise RuntimeError(f"[{ssp_name}] 整幅图没有有效像元，无法计算面积比例。")

    # 比例
    p0 = n0 / N_total
    p1 = n1 / N_total
    p2 = n2 / N_total
    p3 = n3 / N_total

    logger.info(
        f"[{ssp_name}] 总结: N_total={N_total}, "
        f"n0={n0}, n1={n1}, n2={n2}, n3={n3}"
    )
    logger.info(
        f"[{ssp_name}] 比例: p0={p0:.3f}, p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}"
    )

    return {
        "SSP": ssp_name,
        "N_total": N_total,
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def main():
    # ========= Step03: 三个 SSP 的四类面积比例（阈值 0.29445） =========

    # 当前气候栅格
    current_path = (
        r"L:\FutureClim\585_EC\Rhizosheath_avg.asc"
    )

    if not os.path.exists(current_path):
        raise FileNotFoundError(f"找不到当前气候栅格: {current_path}")

    src_cur = rasterio.open(current_path)

    # 三个 SSP 下 3 个 GCM 的路径
    ssp_gcm_paths = {
        "SSP126": {
            "EC-Earth3-Veg": r"L:\FutureClim\126_EC\Rhizosheath_EC-Earth3-Veg_avg.asc",
            "MPI-ESM1-2-HR": r"L:\FutureClim\126_MP\Rhizosheath_MPI-ESM1-2-HR_avg.asc",
            "MRI-ESM2-0": r"L:\FutureClim\126_MR\Rhizosheath_MRI-ESM2-0_avg.asc",
        },
        "SSP245": {
            "EC-Earth3-Veg": r"L:\FutureClim\245_EC\Rhizosheath_EC-Earth3-Veg_avg.asc",
            "MPI-ESM1-2-HR": r"L:\FutureClim\245_MP\Rhizosheath_MPI-ESM1-2-HR_avg.asc",
            "MRI-ESM2-0": r"L:\FutureClim\245_MR\Rhizosheath_MRI-ESM2-0_avg.asc",
        },
        "SSP585": {
            "EC-Earth3-Veg": r"L:\FutureClim\585_EC\Rhizosheath_EC-Earth3-Veg_avg.asc",
            "MPI-ESM1-2-HR": r"L:\FutureClim\585_MP\Rhizosheath_MPI-ESM1-2-HR_avg.asc",
            "MRI-ESM2-0": r"L:\FutureClim\585_MR\Rhizosheath_MRI-ESM2-0_avg.asc",
        },
    }

    # 阈值
    threshold = 0.2732

    # 输出目录与文件
    output_dir = r"I:\Version02\Ensemble_AllSSP"
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(
        output_dir,
        f"Change_Areas_AllSSP_threshold_{threshold:.5f}.csv"
    )

    logger.info("====== Step03: 统计 SSP126 / SSP245 / SSP585 的四类面积比例 ======")
    logger.info(f"使用阈值 T = {threshold}")
    logger.info(f"结果 CSV 将保存到: {out_csv}")

    results = []
    for ssp_name, gcm_paths in ssp_gcm_paths.items():
        logger.info(f"---- 处理 {ssp_name} ----")
        stats = compute_change_stats_for_ssp_on_the_fly(
            ssp_name=ssp_name,
            src_cur=src_cur,
            gcm_paths=gcm_paths,
            threshold=threshold,
            block_height=128,
        )
        results.append(stats)

    # 写出 CSV
    fieldnames = [
        "SSP",
        "N_total",
        "n0", "n1", "n2", "n3",
        "p0", "p1", "p2", "p3",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    src_cur.close()

    logger.info("Step03 完成：所有 SSP 的四类像元数与比例已写入 CSV。")
    logger.info("分类含义: 0=都不适宜, 1=仅当前适宜, 2=仅未来适宜, 3=都适宜")


if __name__ == "__main__":
    main()
