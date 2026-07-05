import pandas as pd
import numpy as np
import networkx as nx
import os
import json

# --- 配置参数 ---
INPUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version04_1021"
OUTPUT_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure"
CORR_MATRIX_FILES = {
    "clim": os.path.join(INPUT_DIR, "clim_r_matrix.csv"),
    "soil": os.path.join(INPUT_DIR, "soil_r_matrix.csv")
}
CORRELATION_THRESHOLD = 0.85  # 严格阈值 |r| > 0.85


def calculate_group_avg_correlation(sub_matrix):
    """计算组内平均相关系数（上三角绝对值）"""
    upper_tri = sub_matrix.where(np.triu(np.ones(sub_matrix.shape), k=1) == True)
    avg_r = np.nanmean(np.abs(upper_tri.values))
    return avg_r


def group_correlated_variables(corr_matrix_path, group_name, corr_threshold):
    """基于相关系数对变量进行分组，并提供详细反馈"""
    print(f"\n=== 分组 {group_name} 组 (|r| > {corr_threshold}) ===")

    # 检查输入文件
    if not os.path.exists(corr_matrix_path):
        print(f"错误: 相关系数矩阵文件不存在 - {corr_matrix_path}")
        return None, None, None

    print(f"读取矩阵: {os.path.basename(corr_matrix_path)}")

    try:
        # 读取相关系数矩阵
        corr_matrix = pd.read_csv(corr_matrix_path, index_col=0)
        corr_matrix.index = corr_matrix.index.map(str)  # 确保索引为字符串
        corr_matrix.columns = corr_matrix.columns.map(str)  # 确保列名为字符串
        num_vars = corr_matrix.shape[0]
        print(f"矩阵维度: {num_vars} x {num_vars} 变量")

        # --- 提取高相关变量对 ---
        upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_upper = corr_matrix.where(upper_tri_mask)
        highly_correlated_pairs = []
        num_high_pairs = 0
        for col in corr_upper.columns:
            for row, r_value in corr_upper[col].items():
                if pd.notna(r_value) and abs(r_value) > corr_threshold:
                    highly_correlated_pairs.append((row, col))
                    num_high_pairs += 1

        print(f"高相关变量对数量: {num_high_pairs} 对")

        # --- 图论分组 ---
        G = nx.Graph()
        G.add_edges_from(highly_correlated_pairs)
        groups = list(nx.connected_components(G))

        # --- 输出分组结果 ---
        group_data = []
        total_group_vars = 0
        print(f"\n发现 {len(groups)} 组高相关变量 (|r| > {corr_threshold}):\n")
        for i, group in enumerate(groups):
            group = sorted(group)  # 按字母顺序排序
            sub_matrix = corr_matrix.loc[group, group]
            avg_r = calculate_group_avg_correlation(sub_matrix)
            group_size = len(group)
            total_group_vars += group_size

            print(f"--- 组 {i + 1} ({group_size} 个变量, 平均 |r| = {avg_r:.3f}) ---")
            for j, var in enumerate(group):
                print(f"  {j + 1}. {var}")

            # 输出组内相关系数矩阵（仅显示上三角，避免冗长）
            print("  组内相关系数矩阵 (上三角):")
            upper_sub = sub_matrix.where(np.triu(np.ones(sub_matrix.shape), k=1) == True)
            print(upper_sub.to_string(float_format="%.3f"))
            print("-" * 80 + "\n")

            # 反馈建议
            if group_size > 1:
                suggestion = f"  建议: 从此组选择 1 个代表性变量（生态重要性高者）。例如，温度相关组选 bio1 (年均温度)；降水组选 bio12 (年降水量)；土壤化学组选 LAYERS_PH_WATER (pH)。"
            else:
                suggestion = "  建议: 此变量独立，可直接保留。"
            print(suggestion)
            print()

            # 保存分组信息
            group_data.append({
                "Group": i + 1,
                "Variables": group,
                "Size": group_size,
                "Avg_Abs_R": round(avg_r, 3),
                "SubMatrix": sub_matrix.to_dict()
            })

        # --- 识别可保留变量 ---
        all_vars = set(corr_matrix.columns)
        grouped_vars = {var for group in groups for var in group}
        safe_vars = sorted(all_vars - grouped_vars)

        print(f"--- 可直接保留的变量 ({group_name}, 无高相关性 |r| <= {corr_threshold}) ---")
        if safe_vars:
            print(f"  数量: {len(safe_vars)} 个")
            for j, var in enumerate(safe_vars):
                print(f"  {j + 1}. {var}")
            print(
                "  建议: 这些变量与其他变量相关性低，可直接用于模型。优先保留生态关键变量，如 bio12 (年降水) 或 LAYERS_PH_WATER (pH)。")
        else:
            print("  数量: 0 个")
            print("  建议: 所有变量均有高相关性，需从分组中选择。")

        # --- 整体反馈 ---
        print(f"\n{group_name} 组整体反馈:")
        print(f"  - 总变量: {num_vars}")
        print(f"  - 高相关组变量覆盖: {total_group_vars} 个 ({total_group_vars / num_vars * 100:.1f}%)")
        print(f"  - 可保留变量: {len(safe_vars)} 个")
        print(f"  - 建议总变量数: 5-8 个 (从高相关组选1个/组 + 可保留变量)")
        print(
            f"  - 生态提示: 根鞘分布受温度 (bio1), 降水 (bio12), 土壤pH (LAYERS_PH_WATER), 水分 (LAYERS_AWC) 影响大。优先这些。")
        print("-" * 80)

        # --- 保存分组结果 ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 保存为CSV (包含组信息和平均r)
        group_list = []
        for i, group in enumerate(groups):
            for var in group:
                group_list.append({
                    "Group": i + 1,
                    "Variable": var,
                    "Group_Size": len(group),
                    "Group_Avg_Abs_R": round(calculate_group_avg_correlation(corr_matrix.loc[group, group]), 3)
                })
        for var in safe_vars:
            group_list.append({
                "Group": "Safe",
                "Variable": var,
                "Group_Size": 1,
                "Group_Avg_Abs_R": 0.0
            })
        group_df = pd.DataFrame(group_list)
        group_csv_path = os.path.join(OUTPUT_DIR, f"{group_name}_variable_groups.csv")
        group_df.to_csv(group_csv_path, index=False)
        print(f"{group_name} 分组结果已保存: {group_csv_path}")

        # 保存为JSON（包含子矩阵）
        json_path = os.path.join(OUTPUT_DIR, f"{group_name}_variable_groups.json")
        with open(json_path, 'w') as f:
            json.dump(group_data, f, indent=4)
        print(f"{group_name} 分组结果（含子矩阵）已保存: {json_path}")

        return group_data, safe_vars, corr_matrix

    except Exception as e:
        print(f"处理 {group_name} 出错: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    print("=== Step03_02_01: Grouping Correlated Variables (Clim and Soil, Detailed Feedback) ===")

    # 处理 clim 和 soil 组
    all_results = {}
    for group_name, corr_matrix_path in CORR_MATRIX_FILES.items():
        group_data, safe_vars, corr_matrix = group_correlated_variables(corr_matrix_path, group_name,
                                                                        CORRELATION_THRESHOLD)
        all_results[group_name] = {
            "group_data": group_data,
            "safe_vars": safe_vars,
            "corr_matrix": corr_matrix
        }

    print("\n=== 所有组处理完成！ ===")
    print("请根据反馈从每组选择1个代表变量，总数控制在15-20个。")