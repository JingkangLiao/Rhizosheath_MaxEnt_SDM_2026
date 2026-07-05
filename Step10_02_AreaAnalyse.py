import rasterio
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# ===================== 参数设置 =====================
# 当前气候适宜性文件
current_asc = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\GCM_CurrentlyClim_avg.asc"

# 未来气候适宜性文件列表
future_ascs = [
    {
        "path": r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\Rhizosheath_2040-2060_SSP245Prediction_Suitablity.asc",
        "scenario": "SSP245"
    },
    {
        "path": r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\Rhizosheath_2040-2060_SSP126Prediction_Suitablity.asc",
        "scenario": "SSP126"
    },
    {
        "path": r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\Rhizosheath_2040-2060_SSP585Prediction_Suitablity.asc",
        "scenario": "SSP585"
    }
]

# 输出统计文件
output_csv = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\Suitability_Change_Statistics.csv"


# ===================== 函数定义 =====================
def calculate_change_statistics(current_path, future_path, scenario_name):
    """计算当前与未来情景的变化统计"""
    print(f"\n📊 计算变化统计: {scenario_name}")

    # 读取当前和未来栅格数据
    with rasterio.open(current_path) as src_current, rasterio.open(future_path) as src_future:
        current = src_current.read(1)
        future = src_future.read(1)
        nodata = src_current.nodata

        # 获取像元面积信息（假设为平方千米）
        transform = src_current.transform
        cell_area = abs(transform[0] * transform[4])  # 计算单个像元面积

        # 检查尺寸一致性
        if current.shape != future.shape:
            raise ValueError("❌ 输入栅格的尺寸不一致，请检查。")

    # 创建有效像元掩膜
    valid_mask = (current != nodata) & (future != nodata)

    # 初始化结果数组
    result = np.full_like(current, nodata, dtype=np.int32)

    # 分类变化
    result[valid_mask & (current == 1) & (future == 1)] = 2  # 稳定区
    result[valid_mask & (current == 1) & (future == 0)] = 1  # 退缩区
    result[valid_mask & (current == 0) & (future == 1)] = 3  # 扩张区
    result[valid_mask & (current == 0) & (future == 0)] = 0  # 持续不适宜区

    # 计算各类像元数量
    count_total = np.count_nonzero(valid_mask)
    count_unsuitable = np.count_nonzero(result == 0)
    count_decrease = np.count_nonzero(result == 1)
    count_stable = np.count_nonzero(result == 2)
    count_increase = np.count_nonzero(result == 3)

    # 验证统计一致性
    total_counted = count_unsuitable + count_decrease + count_stable + count_increase
    if total_counted != count_total:
        print(f"⚠️ 警告: 统计总数不一致 ({total_counted} vs {count_total})")

    # 计算面积比例（百分比）
    def pct(n):
        return n / count_total * 100 if count_total > 0 else np.nan

    # 计算各类面积（平方千米）
    def area(n):
        return n * cell_area

    # 准备统计结果
    stats = {
        "Scenario": scenario_name,
        "TotalArea_km2": area(count_total),
        "UnsuitableArea_km2": area(count_unsuitable),
        "DecreaseArea_km2": area(count_decrease),
        "StableArea_km2": area(count_stable),
        "IncreaseArea_km2": area(count_increase),
        "UnsuitablePercent": pct(count_unsuitable),
        "DecreasePercent": pct(count_decrease),
        "StablePercent": pct(count_stable),
        "IncreasePercent": pct(count_increase),
        "TotalPixels": count_total,
        "UnsuitablePixels": count_unsuitable,
        "DecreasePixels": count_decrease,
        "StablePixels": count_stable,
        "IncreasePixels": count_increase
    }

    print(f"✅ 完成统计: {scenario_name}")
    return stats


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 检查当前文件是否存在
    if not os.path.exists(current_asc):
        raise FileNotFoundError(f"❌ 当前气候文件不存在: {current_asc}")

    # 准备存储所有统计结果
    all_stats = []

    # 处理每个未来情景
    for future_info in tqdm(future_ascs, desc="处理情景"):
        scenario_path = future_info["path"]
        scenario_name = future_info["scenario"]

        # 检查文件是否存在
        if not os.path.exists(scenario_path):
            print(f"⚠️ 跳过: 文件不存在 - {scenario_path}")
            continue

        # 计算统计
        try:
            stats = calculate_change_statistics(current_asc, scenario_path, scenario_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ 处理失败 {scenario_name}: {str(e)}")

    # 转换为DataFrame并保存
    if all_stats:
        df = pd.DataFrame(all_stats)

        # 设置列顺序
        columns_order = [
            "Scenario", "TotalArea_km2", "TotalPixels",
            "UnsuitableArea_km2", "UnsuitablePixels", "UnsuitablePercent",
            "DecreaseArea_km2", "DecreasePixels", "DecreasePercent",
            "StableArea_km2", "StablePixels", "StablePercent",
            "IncreaseArea_km2", "IncreasePixels", "IncreasePercent"
        ]
        df = df[columns_order]

        # 保存到CSV
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ 所有统计结果已保存到: {output_csv}")

        # 打印摘要
        print("\n📈 变化统计摘要:")
        print(df[["Scenario", "UnsuitablePercent", "DecreasePercent",
                  "StablePercent", "IncreasePercent"]])
    else:
        print("\n⚠️ 未生成任何统计结果，请检查输入文件")