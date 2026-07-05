# python 3.12

import os
import glob
import re
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.windows import Window
from tqdm import tqdm
import shutil
import math

# 配置参数
input_dir = r"D:\GISDataFrame\Data\World_veg_GLC_FCS10\Data\GLC_FCS10maps_2023"
resampled_dir = r"D:\GISDataFrame\Data\World_veg_GLC_FCS10\Data\GLC_FCS1000maps_2023"
output_dir = r"D:\GISDataFrame\Data\World_veg_GLC_FCS10\Data\Final_Mosaic"
reference_file = r"D:\GISDataFrame\Data\World_SoilDatabase\LAYERDATA\AVE\ave_alum_sat.tif"
standard_tiles_dir = os.path.join(output_dir, "Standard_Tiles")
final_output = os.path.join(output_dir, "GLC_FCS1000_Global_2023.tif")

# 创建必要的目录
os.makedirs(resampled_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(standard_tiles_dir, exist_ok=True)

# 从参考文件获取目标分辨率
with rasterio.open(reference_file) as ref:
    reference_resolution = (abs(ref.transform.a), abs(ref.transform.e))
    reference_crs = ref.crs
    reference_bounds = ref.bounds
    print(f"参考文件分辨率: X={reference_resolution[0]:.8f}度, Y={reference_resolution[1]:.8f}度")
    print(f"参考文件坐标系: {ref.crs}")
    print(f"参考文件范围: {ref.bounds}")

# 目标分辨率 (与参考文件相同)
TARGET_RESOLUTION = reference_resolution
DEGREE_TO_METERS = 111320  # 赤道处1度≈111,320米

# 瓦片大小 (5度×5度)
TILE_SIZE_DEGREES = 5


def calculate_resolution_info(src):
    """计算并返回分辨率信息"""
    # 原始分辨率（度）
    x_res_deg = src.transform.a
    y_res_deg = abs(src.transform.e)  # Y分辨率通常是负值，取绝对值

    # 地面分辨率（米）
    x_res_m = abs(x_res_deg) * DEGREE_TO_METERS
    y_res_m = abs(y_res_deg) * DEGREE_TO_METERS
    avg_res_m = (x_res_m + y_res_m) / 2

    return {
        "x_res_deg": x_res_deg,
        "y_res_deg": y_res_deg,
        "x_res_m": x_res_m,
        "y_res_m": y_res_m,
        "avg_res_m": avg_res_m
    }


def print_resolution_info(resolution_info, prefix=""):
    """打印分辨率信息"""
    print(f"{prefix}原始分辨率: X={resolution_info['x_res_deg']:.8f} 度, Y={resolution_info['y_res_deg']:.8f} 度")
    print(f"{prefix}地面分辨率: X={resolution_info['x_res_m']:.2f}米, Y={resolution_info['y_res_m']:.2f}米")
    print(f"{prefix}平均分辨率: {resolution_info['avg_res_m']:.2f}米")


def create_standard_tiles():
    """创建标准瓦片作为参考"""
    print("\n创建标准瓦片...")

    with rasterio.open(reference_file) as src:
        # 获取参考文件元数据
        ref_meta = src.meta.copy()

        # 计算瓦片数量
        west, south, east, north = reference_bounds
        num_cols = math.ceil((east - west) / TILE_SIZE_DEGREES)
        num_rows = math.ceil((north - south) / TILE_SIZE_DEGREES)

        print(f"全球瓦片网格: {num_cols}列 × {num_rows}行 ({TILE_SIZE_DEGREES}度×{TILE_SIZE_DEGREES}度)")

        # 创建空瓦片模板
        tile_width = int(TILE_SIZE_DEGREES / TARGET_RESOLUTION[0])
        tile_height = int(TILE_SIZE_DEGREES / TARGET_RESOLUTION[1])

        # 更新元数据
        tile_meta = ref_meta.copy()
        tile_meta.update({
            'width': tile_width,
            'height': tile_height,
            'dtype': 'uint8',
            'nodata': 0
        })

        # 创建所有标准瓦片
        for col in range(num_cols):
            for row in range(num_rows):
                # 计算瓦片边界
                left = west + col * TILE_SIZE_DEGREES
                top = north - row * TILE_SIZE_DEGREES
                right = left + TILE_SIZE_DEGREES
                bottom = top - TILE_SIZE_DEGREES

                # 创建瓦片变换
                transform = rasterio.transform.from_origin(left, top, TARGET_RESOLUTION[0], TARGET_RESOLUTION[1])

                # 设置文件名
                ew = "E" if left >= 0 else "W"
                ns = "N" if top >= 0 else "S"
                filename = f"GLC_STD_{ew}{abs(int(left)):03d}{ns}{abs(int(top)):02d}.tif"
                tile_path = os.path.join(standard_tiles_dir, filename)

                # 创建空瓦片
                with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                    dst.transform = transform
                    # 创建空数组并写入
                    empty_data = np.zeros((tile_height, tile_width), dtype=np.uint8)
                    dst.write(empty_data, 1)

        print(f"已创建 {num_cols * num_rows} 个标准瓦片在: {standard_tiles_dir}")

    return glob.glob(os.path.join(standard_tiles_dir, "*.tif"))


def resample_tif(input_path, output_path, pbar=None):
    """将TIFF文件重采样到目标分辨率，使用众数统计方法"""
    with rasterio.open(input_path) as src:
        # 打印原始分辨率信息
        orig_res = calculate_resolution_info(src)
        if pbar:
            pbar.write(f"\n文件: {os.path.basename(input_path)}")
            pbar.write(f"[原始] 分辨率: X={orig_res['x_res_deg']:.8f}度, Y={orig_res['y_res_deg']:.8f}度")
            pbar.write(f"[原始] 地面分辨率: X={orig_res['x_res_m']:.2f}米, Y={orig_res['y_res_m']:.2f}米")
        else:
            print(f"\n文件: {os.path.basename(input_path)}")
            print(f"[原始] 分辨率: X={orig_res['x_res_deg']:.8f}度, Y={orig_res['y_res_deg']:.8f}度")
            print(f"[原始] 地面分辨率: X={orig_res['x_res_m']:.2f}米, Y={orig_res['y_res_m']:.2f}米")

        # 使用源文件的边界信息计算目标变换
        transform, width, height = calculate_default_transform(
            src.crs,
            reference_crs,  # 目标坐标系
            src.width,
            src.height,
            *src.bounds,  # 提供边界信息 (left, bottom, right, top)
            resolution=TARGET_RESOLUTION  # 目标分辨率
        )

        # 设置输出元数据
        dst_meta = src.meta.copy()
        dst_meta.update({
            'transform': transform,
            'width': width,
            'height': height,
            'crs': reference_crs,  # 目标坐标系
            'dtype': 'uint8',  # 分类数据通常使用uint8
            'nodata': 0,  # 设置合适的nodata值
            'compress': 'lzw'  # 使用LZW压缩
        })

        # 创建目标文件
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            # 重采样处理
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=reference_crs,  # 目标坐标系
                resampling=Resampling.mode  # 使用众数统计方法
            )

        # 打印重采样后的分辨率信息
        with rasterio.open(output_path) as dst:
            new_res = calculate_resolution_info(dst)
            if pbar:
                pbar.write(f"[重采样后] 分辨率: X={new_res['x_res_deg']:.8f}度, Y={new_res['y_res_deg']:.8f}度")
                pbar.write(f"[重采样后] 地面分辨率: X={new_res['x_res_m']:.2f}米, Y={new_res['y_res_m']:.2f}米")
                pbar.write(f"[重采样后] 平均分辨率: {new_res['avg_res_m']:.2f}米")
            else:
                print(f"[重采样后] 分辨率: X={new_res['x_res_deg']:.8f}度, Y={new_res['y_res_deg']:.8f}度")
                print(f"[重采样后] 地面分辨率: X={new_res['x_res_m']:.2f}米, Y={new_res['y_res_m']:.2f}米")
                print(f"[重采样后] 平均分辨率: {new_res['avg_res_m']:.2f}米")

    return output_path


def parse_coordinates_from_filename(filename):
    """从文件名中解析经纬度坐标"""
    # 示例文件名: GLC_FCS10_2023_E000N05.tif
    match = re.search(r'([EW])(\d{3})([NS])(\d{2})', filename)
    if match:
        ew = match.group(1)
        ew_val = int(match.group(2))
        ns = match.group(3)
        ns_val = int(match.group(4))
        return ew, ew_val, ns, ns_val
    return None, None, None, None


def group_files_by_region(files):
    """按地理区域分组文件（北半球和南半球）"""
    north_files = []
    south_files = []

    for file in files:
        filename = os.path.basename(file)
        ew, ew_val, ns, ns_val = parse_coordinates_from_filename(filename)

        if ns == 'N':
            north_files.append(file)
        elif ns == 'S':
            south_files.append(file)

    return north_files, south_files


def create_mosaic(files, output_path, description):
    """创建镶嵌图并打印分辨率信息"""
    print(f"\n开始创建{description}镶嵌图...")

    # 打开所有文件
    src_files = []
    for file in tqdm(files, desc="打开文件"):
        src = rasterio.open(file)
        src_files.append(src)

    # 合并文件
    print("合并文件中...")
    mosaic, out_trans = merge(src_files)

    # 获取输出元数据
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": reference_crs,  # 使用参考文件的坐标系
        "compress": "lzw",
        "bigtiff": "YES"  # 处理大文件
    })

    # 写入输出文件
    print("写入镶嵌图...")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # 关闭所有源文件
    for src in src_files:
        src.close()

    # 打印镶嵌图分辨率信息
    with rasterio.open(output_path) as mosaic_src:
        mosaic_res = calculate_resolution_info(mosaic_src)
        print(f"\n{description}镶嵌图分辨率:")
        print_resolution_info(mosaic_res)

    print(f"{description}镶嵌图已保存至: {output_path}")
    return output_path


def main():
    print("=" * 80)
    print("LandCover Mosaic Master - 土地覆盖镶嵌处理工具")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"重采样输出目录: {resampled_dir}")
    print(f"最终输出目录: {output_dir}")
    print(f"参考文件: {reference_file}")
    print(f"目标分辨率: X={TARGET_RESOLUTION[0]:.8f}度, Y={TARGET_RESOLUTION[1]:.8f}度")

    # 步骤1: 创建标准瓦片
    print("\n" + "=" * 80)
    print("步骤1: 创建标准瓦片")
    print("=" * 80)
    standard_tiles = create_standard_tiles()

    # 步骤2: 重采样所有文件
    print("\n" + "=" * 80)
    print("步骤2: 重采样文件到目标分辨率")
    print("=" * 80)

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"找到 {len(tif_files)} 个TIFF文件")

    # 处理每个文件
    for input_path in tqdm(tif_files, desc="重采样文件", total=len(tif_files)):
        filename = os.path.basename(input_path)
        output_path = os.path.join(resampled_dir, filename)

        # 如果文件已存在，跳过重采样
        if os.path.exists(output_path):
            print(f"\n文件已存在，跳过: {filename}")
            continue

        try:
            # 执行重采样
            resample_tif(input_path, output_path)
        except Exception as e:
            print(f"\n错误: 无法处理文件 {filename} - {str(e)}")
            continue

    print(f"\n所有文件已重采样并保存至: {resampled_dir}")

    # 步骤3: 准备拼接
    print("\n" + "=" * 80)
    print("步骤3: 准备拼接 (Mosaicking)")
    print("=" * 80)

    resampled_files = glob.glob(os.path.join(resampled_dir, "*.tif"))
    print(f"找到 {len(resampled_files)} 个重采样后的文件")

    # 按区域分组文件
    north_files, south_files = group_files_by_region(resampled_files)
    print(f"北半球文件数量: {len(north_files)}")
    print(f"南半球文件数量: {len(south_files)}")

    # 步骤4: 创建镶嵌图
    print("\n" + "=" * 80)
    print("步骤4: 创建全球镶嵌图")
    print("=" * 80)

    # 创建北半球镶嵌图
    north_mosaic = os.path.join(output_dir, "GLC_FCS1000_North_2023.tif")
    if len(north_files) > 0:
        create_mosaic(north_files, north_mosaic, "北半球")
    else:
        print("未找到北半球文件，跳过北半球镶嵌图创建")

    # 创建南半球镶嵌图
    south_mosaic = os.path.join(output_dir, "GLC_FCS1000_South_2023.tif")
    if len(south_files) > 0:
        create_mosaic(south_files, south_mosaic, "南半球")
    else:
        print("未找到南半球文件，跳过南半球镶嵌图创建")

    # 步骤5: 合并南北半球
    print("\n" + "=" * 80)
    print("步骤5: 合并南北半球创建全球镶嵌图 (Global Mosaicking)")
    print("=" * 80)

    # 收集所有镶嵌图
    mosaic_files = []
    if os.path.exists(north_mosaic):
        mosaic_files.append(north_mosaic)
    if os.path.exists(south_mosaic):
        mosaic_files.append(south_mosaic)

    if len(mosaic_files) == 0:
        print("错误: 没有可用的镶嵌图进行合并")
        return

    # 创建最终全球镶嵌图
    create_mosaic(mosaic_files, final_output, "全球")

    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"最终全球镶嵌图已保存至: {final_output}")

    # 检查最终文件分辨率
    with rasterio.open(final_output) as final:
        final_res = calculate_resolution_info(final)
        print("\n最终镶嵌图分辨率:")
        print_resolution_info(final_res)
        print(f"文件大小: {os.path.getsize(final_output) / (1024 * 1024 * 1024):.2f} GB")

    # 检查与参考文件的一致性
    with rasterio.open(reference_file) as ref:
        ref_res = calculate_resolution_info(ref)
        print("\n参考文件分辨率:")
        print_resolution_info(ref_res)

        # 比较分辨率
        res_match = (abs(final_res['x_res_deg'] - ref_res['x_res_deg']) < 1e-6 and
                     abs(final_res['y_res_deg'] - ref_res['y_res_deg']) < 1e-6)

        print(f"\n分辨率一致性检查: {'通过' if res_match else '失败'}")
        if not res_match:
            print("警告: 最终文件分辨率与参考文件不一致!")


if __name__ == "__main__":
    main()