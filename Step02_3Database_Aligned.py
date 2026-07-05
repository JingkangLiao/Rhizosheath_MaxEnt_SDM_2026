import os
import numpy  as np
import rasterio
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor
import threading

# 线程锁，用于安全打印
print_lock = threading.Lock()


def print_safe(message):
    """线程安全的打印函数"""
    with print_lock:
        print(message)


def align_glc_file(glc_path, ref_path, output_path):
    """处理GLC_FCS1000_Global_2023.tif文件，使用内存映射优化"""
    try:
        with rasterio.open(ref_path) as ref:
            ref_profile = ref.profile
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height

            with rasterio.open(glc_path) as src:
                # 检查数据类型
                if src.dtypes[0] == 'uint8':
                    nodata_value = 255
                else:
                    nodata_value = np.nan if src.nodata is None else src.nodata

                # 更新元数据
                dst_profile = src.profile.copy()
                dst_profile.update(
                    width=ref_width,
                    height=ref_height,
                    transform=ref_transform,
                    nodata=nodata_value
                )

                # 使用内存映射创建输出文件
                with rasterio.open(output_path, 'w', **dst_profile) as dst:
                    # 分块处理，减少内存占用
                    block_size = 1024
                    for i in range(0, ref_height, block_size):
                        rows = min(block_size, ref_height - i)

                        # 源数据对应行
                        src_rows = min(rows, src.height - i) if i < src.height else 0

                        if src_rows > 0:
                            # 读取源数据块
                            src_data = src.read(1, window=((i, i + src_rows), (0, ref_width)))

                            # 处理uint8类型
                            if src.dtypes[0] == 'uint8':
                                src_data[src_data == 255] = 0

                            # 创建目标块
                            if src_rows < rows:
                                # 需要填充底部
                                if src.dtypes[0] == 'uint8':
                                    dst_data = np.full((rows, ref_width), nodata_value, dtype=np.uint8)
                                else:
                                    dst_data = np.full((rows, ref_width), nodata_value, dtype=src.dtypes[0])
                                dst_data[:src_rows, :] = src_data
                            else:
                                dst_data = src_data
                        else:
                            # 完全超出源数据范围，填充nodata
                            if src.dtypes[0] == 'uint8':
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=np.uint8)
                            else:
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=src.dtypes[0])

                        # 写入目标块
                        dst.write(dst_data, 1, window=((i, i + rows), (0, ref_width)))

        print_safe(f"✓ Processed: {os.path.basename(glc_path)}")
        return True
    except Exception as e:
        print_safe(f"✗ Error processing {os.path.basename(glc_path)}: {str(e)}")
        return False


def align_soil_file(soil_path, ref_path, output_path):
    """处理土壤数据文件，使用内存映射优化"""
    try:
        with rasterio.open(ref_path) as ref:
            ref_profile = ref.profile
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height

            with rasterio.open(soil_path) as src:
                # 检查数据类型
                if src.dtypes[0] == 'uint8':
                    nodata_value = 255
                else:
                    nodata_value = np.nan if src.nodata is None else src.nodata

                # 更新元数据
                dst_profile = src.profile.copy()
                dst_profile.update(
                    width=ref_width,
                    height=ref_height,
                    transform=ref_transform,
                    nodata=nodata_value
                )

                # 使用内存映射创建输出文件
                with rasterio.open(output_path, 'w', **dst_profile) as dst:
                    # 分块处理
                    block_size = 1024
                    for i in range(0, ref_height, block_size):
                        rows = min(block_size, ref_height - i)

                        # 读取源数据块（如果有）
                        if i < src.height:
                            src_rows = min(rows, src.height - i)
                            src_data = src.read(1, window=((i, i + src_rows), (0, src.width)))

                            # 创建目标块
                            if src.dtypes[0] == 'uint8':
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=np.uint8)
                            else:
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=src.dtypes[0])

                            # 复制源数据到目标块
                            dst_data[:src_rows, :src.width] = src_data

                            # 处理右边界
                            if src.width < ref_width:
                                dst_data[:src_rows, src.width:] = src_data[:, -1].reshape(-1, 1)

                            # 处理下边界（如果需要）
                            if src_rows < rows:
                                dst_data[src_rows:, :] = dst_data[src_rows - 1, :]
                        else:
                            # 超出源数据范围，填充nodata
                            if src.dtypes[0] == 'uint8':
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=np.uint8)
                            else:
                                dst_data = np.full((rows, ref_width), nodata_value, dtype=src.dtypes[0])

                        # 写入目标块
                        dst.write(dst_data, 1, window=((i, i + rows), (0, ref_width)))

        print_safe(f"✓ Processed: {os.path.basename(soil_path)}")
        return True
    except Exception as e:
        print_safe(f"✗ Error processing {os.path.basename(soil_path)}: {str(e)}")
        return False


def main():
    # 参考文件路径
    ref_file = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\WorldClim\2000-2021\version02\Bioclim_1km_avg\wc2.1_1km_bioclim01_avg.tif"

    # GLC文件路径
    glc_folder = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Veg_GLC_FCS"
    glc_file = os.path.join(glc_folder, "GLC_FCS1000_Global_2023.tif")
    glc_output = os.path.join(glc_folder, "GLC_FCS1000_Global_2023_aligned.tif")

    # 土壤数据文件夹
    soil_folder = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Soil_HWSD2"

    # 确定线程数（根据内存和CPU情况调整）
    max_workers = min(8, os.cpu_count() or 1)  # 最多8个线程
    print(f"Using {max_workers} worker threads for parallel processing.")

    # 处理任务列表
    tasks = []

    # 添加GLC文件任务
    if os.path.exists(glc_file):
        tasks.append((align_glc_file, glc_file, ref_file, glc_output))
        print(f"Queued GLC file: {os.path.basename(glc_file)}")
    else:
        print(f"Warning: GLC file not found: {glc_file}")

    # 添加土壤数据文件任务
    soil_files = [f for f in os.listdir(soil_folder)
                  if f.endswith(".tif") and not f.endswith("_aligned.tif")]

    if soil_files:
        print(f"Queued {len(soil_files)} soil data files for processing.")
        for file in soil_files:
            soil_path = os.path.join(soil_folder, file)
            output_path = os.path.join(soil_folder, file.replace(".tif", "_aligned.tif"))
            tasks.append((align_soil_file, soil_path, ref_file, output_path))
    else:
        print(f"Warning: No soil files found in {soil_folder}")

    # 使用线程池并行处理任务
    if tasks:
        print(f"Starting parallel processing of {len(tasks)} files...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda args: args[0](*args[1:]), tasks))

        success_count = sum(results)
        print(f"\nProcessing completed: {success_count}/{len(tasks)} files processed successfully.")
    else:
        print("No files to process.")


if __name__ == "__main__":
    main()