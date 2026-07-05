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
    # ========================
    # 输入路径设置
    # ========================
    # 当前气候路径（本步仍然暂时不用，只是预留给后续 Step02）
    current_path = (
        r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\Result_0128_Currently_AIinModel_Final"
    )

    # SSP245 下三个 GCM 的路径
    gcm_files_245 = {
        "EC-Earth3-Veg": r"L:\FutureClim\245_EC\Rhizosheath_EC-Earth3-Veg_avg.asc",
        "MPI-ESM1-2-HR": r"L:\FutureClim\245_MP\Rhizosheath_MPI-ESM1-2-HR_avg.asc",
        "MRI-ESM2-0": r"L:\FutureClim\245_MR\Rhizosheath_MRI-ESM2-0_avg.asc",
    }

    # 输出目录
    output_dir = r"L:\FutureClim\Ensemble_SSP245"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("====== Step01: 分块计算 SSP245 多模型中位数与不确定性 ======")

    # 1. 用第一个 GCM 作为模板，只读取 meta，不读整幅数据
    first_name, first_path = next(iter(gcm_files_245.items()))
    logger.info(f"以 {first_name} 作为模板: {first_path}")

    with rasterio.open(first_path) as src_first:
        meta_template = src_first.meta.copy()
        height = src_first.height
        width = src_first.width
        crs = src_first.crs
        transform = src_first.transform
        nodata = src_first.nodata

    if nodata is None:
        nodata = -9999.0
        logger.warning(f"模板栅格未定义 nodata，默认使用 {nodata}")

    logger.info(f"栅格尺寸: height={height}, width={width}")
    logger.info(f"CRS: {crs}")
    logger.info(f"transform: {transform}")
    logger.info(f"nodata: {nodata}")

    # 2. 打开所有 GCM 栅格，检查尺寸 / CRS / transform 一致性
    srcs = {}
    for name, path in gcm_files_245.items():
        logger.info(f"打开 GCM: {name} -> {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到栅格文件: {path}")

        src = rasterio.open(path)

        if src.height != height or src.width != width:
            raise ValueError(f"{name} 的尺寸与模板不一致: "
                             f"({src.height}, {src.width}) != ({height}, {width})")
        if src.crs != crs:
            raise ValueError(f"{name} 的 CRS 与模板不一致: {src.crs} != {crs}")
        if src.transform != transform:
            raise ValueError(f"{name} 的 transform 与模板不一致.")

        srcs[name] = src

    # 3. 创建输出栅格文件（中位数 / 标准差 / 范围）
    meta_out = meta_template.copy()
    meta_out.update(
        dtype="float32",
        count=1,
        nodata=nodata
    )

    out_median_path = os.path.join(output_dir, "Rhizosheath_SSP245_median.asc")
    out_sd_path = os.path.join(output_dir, "Rhizosheath_SSP245_sd.asc")
    out_range_path = os.path.join(output_dir, "Rhizosheath_SSP245_range.asc")

    with rasterio.open(out_median_path, "w", **meta_out) as dst_median, \
         rasterio.open(out_sd_path, "w", **meta_out) as dst_sd, \
         rasterio.open(out_range_path, "w", **meta_out) as dst_range:

        # 3.1 分块参数：每次处理 128 行，控制内存用量
        block_height = 128

        for row_start in range(0, height, block_height):
            row_stop = min(height, row_start + block_height)
            window = ((row_start, row_stop), (0, width))

            # 读取 3 个 GCM 在该 window 下的数据
            block_list = []
            for name, src in srcs.items():
                arr = src.read(1, window=window).astype("float32")
                # 将 nodata -> np.nan
                arr[arr == nodata] = np.nan
                block_list.append(arr)

            # 堆栈: (GCM, rows_block, cols)
            stack = np.stack(block_list, axis=0)

            # 3.2 计算中位数 / 标准差 / 范围
            median_block = np.nanmedian(stack, axis=0)
            sd_block = np.nanstd(stack, axis=0)
            max_block = np.nanmax(stack, axis=0)
            min_block = np.nanmin(stack, axis=0)
            range_block = max_block - min_block

            # 3.3 将 nan 替换为 nodata，并写入输出栅格
            median_block = np.where(np.isnan(median_block), nodata, median_block).astype("float32")
            sd_block = np.where(np.isnan(sd_block), nodata, sd_block).astype("float32")
            range_block = np.where(np.isnan(range_block), nodata, range_block).astype("float32")

            dst_median.write(median_block, 1, window=window)
            dst_sd.write(sd_block, 1, window=window)
            dst_range.write(range_block, 1, window=window)

            logger.info(f"已处理行 {row_start}–{row_stop} / {height}")

    # 4. 关闭所有输入栅格
    for src in srcs.values():
        src.close()

    logger.info("Step01 完成：SSP245 多模型中位数 / 不确定性栅格已生成。")


if __name__ == "__main__":
    main()
