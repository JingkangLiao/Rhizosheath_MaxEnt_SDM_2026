import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===================== 参数设置 =====================
INPUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version03_0721\ASC"
OUTPUT_CSV = r"D:\ASC_Files_Validation_Report.csv"
EXPECTED_NODATA = -9999  # 期望的空值表示


# ===================== 主函数 =====================
def analyze_asc_files(directory):
    """分析ASC文件属性"""
    # 获取所有ASC文件
    asc_files = [f for f in os.listdir(directory) if f.lower().endswith('.asc')]

    if not asc_files:
        print(f"❌ 在目录中未找到任何ASC文件: {directory}")
        return None

    print(f"🔍 找到 {len(asc_files)} 个ASC文件进行分析...")

    # 存储结果
    results = []
    dimensions = {}  # 存储尺寸统计
    nodata_stats = {}  # 存储空值统计

    # 分析每个文件
    for filename in tqdm(asc_files, desc="分析文件"):
        file_path = os.path.join(directory, filename)
        result = {
            "File": filename,
            "Path": file_path,
            "Nodata_Value": None,
            "Nodata_Percent": None,
            "Rows": None,
            "Columns": None,
            "Resolution": None,
            "Size_MB": None,
            "Is_Nodata_Correct": False,
            "Error": None
        }

        try:
            # 获取文件大小
            result["Size_MB"] = os.path.getsize(file_path) / (1024 * 1024)

            # 读取文件
            with rasterio.open(file_path) as src:
                # 获取基本信息
                result["Rows"] = src.height
                result["Columns"] = src.width
                result["Resolution"] = src.res

                # 获取nodata值
                result["Nodata_Value"] = src.nodata
                result["Is_Nodata_Correct"] = (result["Nodata_Value"] == EXPECTED_NODATA)

                # 读取数据
                data = src.read(1)

                # 计算nodata占比
                if src.nodata is not None:
                    nodata_mask = (data == src.nodata)
                    nodata_count = np.sum(nodata_mask)
                    total_pixels = data.size
                    result["Nodata_Percent"] = (nodata_count / total_pixels) * 100

                    # 更新全局统计
                    nodata_stats[filename] = result["Nodata_Percent"]
                else:
                    result["Nodata_Percent"] = 0
                    result["Error"] = "未设置Nodata值"

                # 更新尺寸统计
                dim_key = f"{src.width}x{src.height}"
                if dim_key not in dimensions:
                    dimensions[dim_key] = []
                dimensions[dim_key].append(filename)

        except Exception as e:
            result["Error"] = str(e)

        results.append(result)

    return results, dimensions, nodata_stats


def generate_report(results, dimensions, nodata_stats):
    """生成分析报告"""
    # 创建DataFrame
    df = pd.DataFrame(results)

    # 保存到CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 分析报告已保存到: {OUTPUT_CSV}")

    # 打印汇总信息
    print("\n" + "=" * 50)
    print("ASC文件分析汇总报告")
    print("=" * 50)

    # 1. 尺寸一致性分析
    print("\n📏 尺寸一致性分析:")
    if len(dimensions) == 1:
        dim_key = list(dimensions.keys())[0]
        print(f"✅ 所有文件尺寸一致: {dim_key}")
    else:
        print(f"⚠️ 发现 {len(dimensions)} 种不同尺寸:")
        for dim, files in dimensions.items():
            print(f"  - 尺寸 {dim}: {len(files)} 个文件")
            if len(files) < 5:  # 如果文件少，列出文件名
                print(f"    文件列表: {', '.join(files)}")

    # 2. Nodata值分析
    correct_nodata = df["Is_Nodata_Correct"].sum()
    total_files = len(df)
    print(f"\n🔢 Nodata值分析:")
    print(
        f"  - 使用{EXPECTED_NODATA}作为空值的文件: {correct_nodata}/{total_files} ({correct_nodata / total_files * 100:.1f}%)")

    # 3. Nodata占比分析
    if nodata_stats:
        avg_nodata = sum(nodata_stats.values()) / len(nodata_stats)
        min_nodata = min(nodata_stats.values())
        max_nodata = max(nodata_stats.values())

        print(f"\n📊 Nodata占比分析:")
        print(f"  - 平均空值占比: {avg_nodata:.2f}%")
        print(f"  - 最小空值占比: {min_nodata:.2f}%")
        print(f"  - 最大空值占比: {max_nodata:.2f}%")

        # 找出空值占比异常的文件
        high_nodata_files = [f for f, p in nodata_stats.items() if p > 50]
        if high_nodata_files:
            print(f"⚠️ 以下文件空值占比超过50%:")
            for f in high_nodata_files:
                print(f"  - {f}: {nodata_stats[f]:.2f}%")

    # 4. 错误文件分析
    error_files = df[df["Error"].notnull()]
    if not error_files.empty:
        print("\n❌ 错误文件分析:")
        for _, row in error_files.iterrows():
            print(f"  - {row['File']}: {row['Error']}")

    # 5. 文件大小分析
    avg_size = df["Size_MB"].mean()
    min_size = df["Size_MB"].min()
    max_size = df["Size_MB"].max()

    print(f"\n💾 文件大小分析:")
    print(f"  - 平均大小: {avg_size:.2f}MB")
    print(f"  - 最小大小: {min_size:.2f}MB")
    print(f"  - 最大大小: {max_size:.2f}MB")

    return df


# ===================== 执行 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("ASC文件分析工具")
    print("=" * 50)
    print(f"分析目录: {INPUT_DIR}")
    print(f"期望的空值表示: {EXPECTED_NODATA}")

    # 分析文件
    results, dimensions, nodata_stats = analyze_asc_files(INPUT_DIR)

    if results:
        # 生成报告
        df = generate_report(results, dimensions, nodata_stats)

        # 保存详细报告
        detailed_report = os.path.join(os.path.dirname(OUTPUT_CSV), "ASC_Files_Detailed_Report.xlsx")
        df.to_excel(detailed_report, index=False)
        print(f"\n📝 详细报告已保存到: {detailed_report}")

    print("\n分析完成!")