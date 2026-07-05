# python 3.12
"这段代码遍历指定目录下的所有 TIFF 栅格文件，分析每个文件的坐标系类型（地理 / 投影）、原始分辨率（适配度 / 米单位）并将地理坐标系的分辨率转换为米级地面分辨率，计算平均地面分辨率等关键参数，最终生成包含文件详情、分辨率统计及分布的文本格式分辨率分析报告并保存至指定路径。"

import os
import glob
import rasterio
from tqdm import tqdm
import numpy as np

# 配置参数
input_dir = r"G:\World_SoilDatabase\LAYERD1_TIF"
output_report = os.path.join(input_dir, "resolution_details.txt")


def calculate_meter_resolution(transform, crs):
    """
    计算以米为单位的分辨率
    处理地理坐标系（度）和投影坐标系（米）两种情况
    """
    # 获取分辨率值
    x_res = transform.a
    y_res = abs(transform.e)  # Y分辨率通常是负值，取绝对值

    # 如果是地理坐标系（单位是度）
    if crs.is_geographic:
        # 1度 ≈ 111,320米（赤道处）
        x_res_m = abs(x_res) * 111320
        y_res_m = abs(y_res) * 111320
        unit = "度"
    else:
        # 投影坐标系，单位已经是米
        x_res_m = abs(x_res)
        y_res_m = abs(y_res)
        unit = "米"

    return x_res, y_res, x_res_m, y_res_m, unit


def analyze_resolution(tif_path):
    """分析单个TIFF文件的分辨率"""
    with rasterio.open(tif_path) as src:
        # 计算分辨率（以米为单位）
        x_res, y_res, x_res_m, y_res_m, unit = calculate_meter_resolution(src.transform, src.crs)

        # 计算平均分辨率
        avg_res_m = (x_res_m + y_res_m) / 2

        # 获取坐标系信息
        crs_type = "地理坐标系" if src.crs.is_geographic else "投影坐标系"

        return {
            "filename": os.path.basename(tif_path),
            "x_resolution": x_res,
            "y_resolution": y_res,
            "unit": unit,
            "x_resolution_m": x_res_m,
            "y_resolution_m": y_res_m,
            "avg_resolution_m": avg_res_m,
            "crs": str(src.crs),
            "crs_type": crs_type,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds
        }


def generate_report(results):
    """生成详细分辨率报告"""
    with open(output_report, "w", encoding="utf-8") as f:
        # 写入报告头部
        f.write("遥感影像分辨率详细报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"分析目录: {input_dir}\n")
        f.write(f"文件总数: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        # 写入详细结果
        f.write("文件分辨率详情:\n")
        f.write("=" * 80 + "\n")
        for result in results:
            f.write(f"文件名: {result['filename']}\n")
            f.write(f"坐标系类型: {result['crs_type']} ({result['crs']})\n")
            f.write(f"原始X方向分辨率: {result['x_resolution']:.8f} {result['unit']}\n")
            f.write(f"原始Y方向分辨率: {result['y_resolution']:.8f} {result['unit']}\n")
            f.write(f"X方向地面分辨率: {result['x_resolution_m']:.2f} 米\n")
            f.write(f"Y方向地面分辨率: {result['y_resolution_m']:.2f} 米\n")
            f.write(f"平均地面分辨率: {result['avg_resolution_m']:.2f} 米\n")
            f.write(f"影像尺寸: {result['width']}×{result['height']} 像素\n")
            f.write(f"地理范围: \n")
            f.write(f"  左下: ({result['bounds'].left:.4f}, {result['bounds'].bottom:.4f})\n")
            f.write(f"  右上: ({result['bounds'].right:.4f}, {result['bounds'].top:.4f})\n")
            f.write("-" * 80 + "\n")

        # 添加总结
        f.write("\n" + "=" * 80 + "\n")
        f.write("分析总结:\n")
        f.write("=" * 80 + "\n")

        # 计算分辨率统计
        min_res = min(r['avg_resolution_m'] for r in results)
        max_res = max(r['avg_resolution_m'] for r in results)
        avg_res = np.mean([r['avg_resolution_m'] for r in results])

        f.write(f"最小平均分辨率: {min_res:.2f} 米\n")
        f.write(f"最大平均分辨率: {max_res:.2f} 米\n")
        f.write(f"平均分辨率: {avg_res:.2f} 米\n")
        f.write("\n")

        # 分辨率分布统计
        f.write("分辨率分布:\n")
        resolutions = sorted([r['avg_resolution_m'] for r in results])
        for res in resolutions:
            f.write(f"  - {res:.2f} 米\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告已保存至: " + output_report + "\n")
        f.write("=" * 80 + "\n")


def main():
    print("遥感影像分辨率详细分析工具")
    print("=" * 80)
    print(f"分析目录: {input_dir}")

    # 获取所有TIFF文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"找到 {len(tif_files)} 个TIFF文件")

    # 分析每个文件
    results = []
    for tif_path in tqdm(tif_files, desc="分析文件分辨率"):
        try:
            result = analyze_resolution(tif_path)
            results.append(result)
        except Exception as e:
            print(f"\n错误: 无法分析文件 {os.path.basename(tif_path)} - {str(e)}")

    # 生成报告
    generate_report(results)

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print(f"详细报告已保存至: {output_report}")

    # 在控制台显示简要结果
    print("\n文件分辨率摘要:")
    print("=" * 80)
    for result in results:
        print(f"{result['filename']}:")
        print(f"  坐标系: {result['crs_type']}")
        print(
            f"  原始分辨率: X={result['x_resolution']:.8f} {result['unit']}, Y={result['y_resolution']:.8f} {result['unit']}")
        print(f"  地面分辨率: X={result['x_resolution_m']:.2f}米, Y={result['y_resolution_m']:.2f}米")
        print(f"  平均分辨率: {result['avg_resolution_m']:.2f}米")
        print("-" * 80)


if __name__ == "__main__":
    main()