import rasterio
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from rasterio.sample import sample_gen
from rasterio.transform import from_bounds
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# 文件夹路径（基于之前的 ENV_FOLDERS）
ENV_FOLDERS = [
    r"L:\Bioclim_1km_avg",
    r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Veg_GLC_FCS\version02_251021",
    r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\Soil_HWSD2\version02_251021"
]

# 分开计算：气候组 (WorldClim) 和 土壤组 (仅 Soil_HWSD2，排除植被 GLC_FCS)
CLIM_FOLDERS = [ENV_FOLDERS[0]]  # WorldClim (气候变量)
SOIL_FOLDERS = [ENV_FOLDERS[2]]  # 仅 Soil_HWSD2 (土壤变量，排除 Veg_GLC_FCS)

OUTPUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 10000  # 目标有效采样点数
INITIAL_SAMPLE_MULTIPLIER = 2  # 初始采样点数倍数（以补偿NaN，初始20000）
GLOBAL_BOUNDS = (-180, -90, 180, 90)  # 全球大致范围 (left, bottom, right, top)


def get_raster_files(folders, group_name):
    """收集指定组的所有 .tif 文件路径"""
    all_files = []
    for folder in folders:
        files = glob.glob(os.path.join(folder, "*.tif"))
        all_files.extend(files)
        print(f"从 {group_name} 的 {folder} 找到 {len(files)} 个文件")
    print(f"\n{group_name} 总共找到 {len(all_files)} 个栅格文件")
    return all_files


def sample_environmental_data(files, target_size, bounds, group_name):
    """从栅格文件中随机采样环境值，确保至少 target_size 个有效值"""
    print(f"\n开始 {group_name} 采样，目标 {target_size} 个有效点...")

    # 初始采样点数（倍数补偿NaN）
    initial_size = target_size * INITIAL_SAMPLE_MULTIPLIER
    np.random.seed(42)  # 固定种子以确保可重复

    data_dict = {}
    valid_samples_per_file = {}

    # 先为每个文件采样足够有效点
    for file_path in files:
        var_name = os.path.basename(file_path).replace('.tif', '')  # 变量名从文件名提取
        valid_samples = []

        while len(valid_samples) < target_size:
            # 生成新批次随机坐标点
            lons = np.random.uniform(bounds[0], bounds[2], initial_size)
            lats = np.random.uniform(bounds[1], bounds[3], initial_size)
            coords = list(zip(lons, lats))

            try:
                with rasterio.open(file_path) as src:
                    samples = [s[0] for s in sample_gen(src, coords) if not np.isnan(s[0])]
                    valid_samples.extend(samples)
                    print(f"  - {var_name}: 当前有效值 {len(valid_samples)}, 本批次新增 {len(samples)}")
            except Exception as e:
                print(f"  - {var_name}: 采样错误 - {e}")
                continue

        # 截取到目标大小
        valid_samples = valid_samples[:target_size]
        data_dict[var_name] = valid_samples
        valid_samples_per_file[var_name] = len(valid_samples)
        print(f"  - {var_name}: 最终采样 {len(valid_samples)} 个有效值")

    # 构建 DataFrame（所有文件使用相同索引）
    min_samples = min(len(v) for v in data_dict.values()) if data_dict else 0
    if min_samples == 0:
        raise ValueError(f"{group_name} 无法采样有效数据，请检查栅格文件")

    df = pd.DataFrame({k: v[:min_samples] for k, v in data_dict.items()})
    print(f"\n{group_name} 采样完成: {df.shape[1]} 个变量, {df.shape[0]} 个点")
    return df


def calculate_pearson_correlation(df):
    """计算 Pearson 相关系数矩阵"""
    corr_matrix = df.corr(method='pearson')
    return corr_matrix


def calculate_vif(df):
    """计算每个变量的 VIF"""
    df_with_const = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i + 1) for i in range(len(df.columns))]
    return vif_data


def plot_heatmap(matrix, title, filename_prefix, output_dir):
    """绘制热力图并保存"""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(matrix, dtype=bool))  # 上三角掩码，避免对称重复
    sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0, mask=mask, fmt='.2f', square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_r_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{filename_prefix} 相关系数热力图已保存: {os.path.join(output_dir, f'{filename_prefix}_r_heatmap.png')}")


def plot_vif_bar(vif_df, title, filename_prefix, output_dir):
    """绘制 VIF 条形图并保存"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=vif_df, x='Variable', y='VIF', palette='viridis')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_vif_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{filename_prefix} VIF 图已保存: {os.path.join(output_dir, f'{filename_prefix}_vif_heatmap.png')}")


def process_group(folders, group_name, target_size, bounds, output_dir):
    """处理一个变量组（气候或土壤）"""
    print(f"\n=== 处理 {group_name} 组 ===")

    # 步骤 1: 收集文件
    raster_files = get_raster_files(folders, group_name)

    # 步骤 2: 采样数据（改进：重新采样直到有效）
    env_df = sample_environmental_data(raster_files, target_size, bounds, group_name)

    # 步骤 3: 计算 Pearson r 矩阵
    corr_matrix = calculate_pearson_correlation(env_df)
    corr_csv_path = os.path.join(output_dir, f"{group_name}_r_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)
    print(f"{group_name} Pearson r 矩阵已保存: {corr_csv_path}")

    # 步骤 4: 计算 VIF
    vif_df = calculate_vif(env_df)
    vif_csv_path = os.path.join(output_dir, f"{group_name}_vif_values.csv")
    vif_df.to_csv(vif_csv_path, index=False)
    print(f"{group_name} VIF 值已保存: {vif_csv_path}")

    # 步骤 5: 生成热力图
    plot_heatmap(corr_matrix, f'{group_name} Pearson Correlation Matrix (|r|)', group_name, output_dir)

    # 步骤 6: 生成 VIF 图
    plot_vif_bar(vif_df, f'{group_name} Variance Inflation Factor (VIF)', group_name, output_dir)

    # 步骤 7: 输出摘要
    print(f"\n{group_name} 摘要:")
    print(f"变量数量: {len(env_df.columns)}")
    print(f"高 VIF 变量 (>5): {len(vif_df[vif_df['VIF'] > 5])} 个")
    print(f"高 VIF 变量 (>10): {len(vif_df[vif_df['VIF'] > 10])} 个")
    print("-" * 50)


if __name__ == "__main__":
    print("=== Step03_01: Pearson Correlation & VIF Matrix Calculation (Separate Groups, Improved Sampling) ===")

    # 处理气候组
    process_group(CLIM_FOLDERS, "clim", SAMPLE_SIZE, GLOBAL_BOUNDS, OUTPUT_DIR)

    # 处理土壤组（仅Soil_HWSD2，排除GLC_FCS）
    process_group(SOIL_FOLDERS, "soil", SAMPLE_SIZE, GLOBAL_BOUNDS, OUTPUT_DIR)

    print("所有组计算完成！请检查输出目录。")