import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import shutil
from rasterio.crs import CRS

# ===================== 参数设置 =====================
# 输入文件路径
INPUT_FILE = [
    r"L:\Pre-Resample_Maxent\BIO01.asc",
    r"L:\Pre-Resample_Maxent\BIO04.asc",
    r"L:\Pre-Resample_Maxent\BIO07.asc",
    r"L:\Pre-Resample_Maxent\BIO11.asc"
]

# 输出目录
OUTPUT_DIR = r"L:\Resample_Maxent"

# 目标分辨率（度）
TARGET_RESOLUTION = 0.05  # 约5km分辨率（可调整）

# 目标文件大小（MB）
TARGET_SIZE_MB = 900  # 900MB ≈ 0.9GB，留出缓冲空间

# 默认坐标系（WGS84）
DEFAULT_CRS = CRS.from_epsg(4326)


# ===================== 主函数 =====================
def resample_raster(input_path, output_dir, target_resolution, max_size_mb):
    """重采样栅格文件并控制文件大小"""
    print(f"🔍 开始处理文件: {os.path.basename(input_path)}")
    print(f"📏 目标分辨率: {target_resolution}度")
    print(f"💾 目标文件大小: < {max_size_mb}MB")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))

    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入文件不存在 - {input_path}")
        return False

    try:
        # 获取原始文件信息
        with rasterio.open(input_path) as src:
            # 获取或设置CRS
            src_crs = src.crs
            if src_crs is None:
                src_crs = DEFAULT_CRS
                print(f"⚠️ 原始文件缺少CRS，使用默认值: {DEFAULT_CRS}")

            # 计算当前分辨率
            x_res = src.transform.a
            y_res = abs(src.transform.e)
            current_res = min(x_res, y_res)

            print(f"📐 原始分辨率: {current_res:.6f}度")
            print(f"📏 原始尺寸: {src.width}列 × {src.height}行")

            # 计算原始文件大小
            original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            print(f"💾 原始文件大小: {original_size:.2f}MB")

            # 如果分辨率已经小于目标分辨率且文件大小合适，则直接复制
            if current_res <= target_resolution and original_size <= max_size_mb:
                print("✅ 文件已符合要求，直接复制")
                shutil.copy2(input_path, output_path)
                return True

            # 计算目标尺寸
            scale_factor = current_res / target_resolution
            target_width = int(src.width * scale_factor)
            target_height = int(src.height * scale_factor)

            print(f"📏 目标尺寸: {target_width}列 × {target_height}行")

            # 创建目标profile
            dst_profile = src.profile.copy()
            dst_transform = rasterio.Affine(
                target_resolution, 0, src.bounds.left,
                0, -target_resolution, src.bounds.top
            )
            dst_profile.update({
                'width': target_width,
                'height': target_height,
                'transform': dst_transform,
                'crs': src_crs  # 确保设置CRS
            })

            # 读取数据并重采样
            print("🔄 正在进行重采样...")
            data = src.read(1)
            resampled = np.zeros((target_height, target_width), dtype=data.dtype)

            # 执行重采样，明确指定CRS
            reproject(
                source=data,
                destination=resampled,
                src_transform=src.transform,
                src_crs=src_crs,  # 明确指定源CRS
                dst_transform=dst_transform,
                dst_crs=src_crs,  # 明确指定目标CRS
                resampling=Resampling.bilinear
            )

        # 保存重采样结果
        print("💾 保存重采样结果...")
        with rasterio.open(output_path, 'w', **dst_profile) as dst:
            dst.write(resampled, 1)

        # 检查文件大小
        final_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ 重采样完成! 文件大小: {final_size:.2f}MB")

        # 如果文件仍然太大，进行二次优化
        if final_size > max_size_mb:
            print(f"⚠️ 文件大小超过目标({final_size:.2f}MB > {max_size_mb}MB)，进行二次优化")
            return compress_raster(output_path, max_size_mb)

        return True

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compress_raster(file_path, max_size_mb):
    """压缩栅格文件到目标大小"""
    print(f"📦 开始压缩文件: {os.path.basename(file_path)}")

    try:
        # 读取文件
        with rasterio.open(file_path) as src:
            data = src.read(1)
            profile = src.profile.copy()

        # 原始文件大小
        original_size = os.path.getsize(file_path) / (1024 * 1024)

        # 尝试不同的压缩级别
        for compression in ['DEFLATE', 'LZW', 'PACKBITS']:
            print(f"🔧 尝试压缩方法: {compression}")

            # 更新profile
            profile.update({
                'driver': 'GTiff',
                'compress': compression,
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256
            })

            # 创建临时文件
            temp_path = file_path.replace('.asc', f'_temp_{compression}.tif')

            # 保存压缩文件
            with rasterio.open(temp_path, 'w', **profile) as dst:
                dst.write(data, 1)

            # 检查大小
            compressed_size = os.path.getsize(temp_path) / (1024 * 1024)
            print(f"  压缩后大小: {compressed_size:.2f}MB")

            # 如果满足要求，替换原文件
            if compressed_size <= max_size_mb:
                print(f"✅ 压缩成功! 使用 {compression} 压缩")
                os.remove(file_path)
                shutil.move(temp_path, file_path)
                return True

        # 如果所有压缩方法都不行，尝试降低分辨率
        print("⚠️ 所有压缩方法均未达到目标大小，尝试进一步降低分辨率")
        return resample_raster(file_path, os.path.dirname(file_path),
                               TARGET_RESOLUTION * 1.5, max_size_mb)

    except Exception as e:
        print(f"❌ 压缩失败: {str(e)}")
        return False


# ===================== 执行 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("栅格文件重采样与压缩工具")
    print("=" * 50)

    # 处理多个输入文件
    success_count = 0
    total_files = len(INPUT_FILE)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 在 __main__ 部分
    with ProcessPoolExecutor() as executor:
        futures = {}
        for input_file in INPUT_FILE:
            future = executor.submit(
                resample_raster,
                input_file,
                OUTPUT_DIR,
                TARGET_RESOLUTION,
                TARGET_SIZE_MB
            )
            futures[future] = input_file

        success_count = 0
        for future in as_completed(futures):
            input_file = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"✅ {os.path.basename(input_file)} 处理成功")
                else:
                    print(f"❌ {os.path.basename(input_file)} 处理失败")
            except Exception as e:
                print(f"❌ {os.path.basename(input_file)} 处理异常: {str(e)}")

        if success:
            success_count += 1
            output_file = os.path.join(OUTPUT_DIR, os.path.basename(input_file))
            print(f"✅ 处理成功! 输出文件: {output_file}")
        else:
            print(f"❌ 处理失败: {input_file}")

    print(f"\n🎉 处理完成! 成功: {success_count}/{total_files} 个文件")

    # 如果有失败的文件，列出它们
    if success_count < total_files:
        print("\n⚠️ 以下文件处理失败:")
        for input_file in INPUT_FILE:
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_file))
            if not os.path.exists(output_path):
                print(f"  - {input_file}")

    print("\n程序结束")