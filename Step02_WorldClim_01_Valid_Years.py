import os
import re
import glob
import json
import logging
from collections import defaultdict


def scan_and_validate(input_folder):
    """
    步骤1: 扫描输入文件夹，验证数据完整性，并生成文件列表

    参数:
        input_folder (str): 包含月度气候数据的文件夹路径

    返回:
        tuple: (valid_years_dict, report)
        valid_years_dict: 包含完整数据的年份字典
        report: 验证过程的详细报告
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    logger.info(f"开始扫描文件夹: {input_folder}")

    # 获取所有TIFF文件
    files = glob.glob(os.path.join(input_folder, "*.tif"))
    if not files:
        logger.error("未找到任何TIFF文件！请检查输入路径")
        return None, "错误：未找到任何TIFF文件"

    logger.info(f"找到 {len(files)} 个TIFF文件")

    # 文件名模式
    pattern = r"wc2\.1_cruts4\.09_2\.5m_(tmin|tmax|prec)_(\d{4})-(\d{2})\.tif"

    # 按年份和变量类型分组
    grouped = defaultdict(lambda: defaultdict(list))
    years = set()

    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)

        if match:
            var_type = match.group(1)  # tmin, tmax 或 prec
            year = match.group(2)
            month = match.group(3)
            years.add(year)

            grouped[year][var_type].append((month, file_path))

    # 检查是否找到有效文件
    if not grouped:
        logger.error("未找到匹配的文件！请检查文件名格式是否符合预期")
        return None, "错误：未找到匹配的文件"

    # 按月份排序每个年份的每个变量
    for year, vars_dict in grouped.items():
        for var_type in ['tmin', 'tmax', 'prec']:
            if var_type in vars_dict:
                # 按月份排序
                vars_dict[var_type] = sorted(vars_dict[var_type], key=lambda x: x[0])

    # 验证数据完整性
    valid_years = {}
    incomplete_years = []

    # 预期的月份集合
    expected_months = [f"{i:02d}" for i in range(1, 13)]

    for year in sorted(years):
        year_data = grouped[year]
        is_complete = True
        missing_data = []

        # 检查三种变量是否都存在
        for var_type in ['tmin', 'tmax', 'prec']:
            if var_type not in year_data:
                is_complete = False
                missing_data.append(f"缺少{var_type}变量")
                continue

            # 检查月份是否完整
            existing_months = {month for month, _ in year_data[var_type]}
            missing_months = set(expected_months) - existing_months

            if missing_months:
                is_complete = False
                missing_data.append(f"{var_type}缺少月份: {', '.join(sorted(missing_months))}")

        if is_complete:
            valid_years[year] = {
                'tmin': [path for _, path in year_data['tmin']],
                'tmax': [path for _, path in year_data['tmax']],
                'prec': [path for _, path in year_data['prec']]
            }
        else:
            incomplete_years.append((year, missing_data))

    # 生成报告
    report_lines = [
        f"扫描报告 - 输入文件夹: {input_folder}",
        f"扫描时间: {logging.Formatter().formatTime(logging.makeLogRecord({}))}",
        f"",
        f"总文件数: {len(files)}",
        f"找到的年份: {len(years)}个 ({', '.join(sorted(years))})",
        f"",
        f"完整年份: {len(valid_years)}个",
        f"不完整年份: {len(incomplete_years)}个",
        f""
    ]

    # 添加完整年份详情
    report_lines.append("完整年份详情:")
    for year in sorted(valid_years.keys()):
        report_lines.append(f"  {year}:")
        report_lines.append(f"    tmin: {len(valid_years[year]['tmin'])}个文件")
        report_lines.append(f"    tmax: {len(valid_years[year]['tmax'])}个文件")
        report_lines.append(f"    prec: {len(valid_years[year]['prec'])}个文件")

    # 添加不完整年份详情
    if incomplete_years:
        report_lines.append("")
        report_lines.append("不完整年份详情:")
        for year, issues in incomplete_years:
            report_lines.append(f"  {year}:")
            for issue in issues:
                report_lines.append(f"    - {issue}")

    report = "\n".join(report_lines)

    logger.info(f"扫描完成！完整年份: {len(valid_years)}, 不完整年份: {len(incomplete_years)}")
    logger.info(f"完整年份列表: {', '.join(sorted(valid_years.keys()))}")

    return valid_years, report


def save_results(valid_years, report, output_dir):
    """保存扫描和验证结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON文件
    json_path = os.path.join(output_dir, "valid_years.json")
    with open(json_path, 'w') as f:
        json.dump(valid_years, f, indent=2)

    # 保存报告
    report_path = os.path.join(output_dir, "scan_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    return json_path, report_path


if __name__ == "__main__":
    # 配置输入文件夹 - 使用您的实际路径
    input_folder = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\WorldClim\2000-2021\version02\monthly"

    # 配置输出文件夹 - 使用您指定的路径
    output_dir = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\BioclimaticVariables\2000-2021\version02\valid_years_output"

    # 执行扫描和验证
    valid_years, report = scan_and_validate(input_folder)

    if valid_years is not None:
        # 保存结果
        json_path, report_path = save_results(valid_years, report, output_dir)

        print("\n" + "=" * 80)
        print("扫描和验证完成！结果已保存：")
        print(f"- 有效年份JSON文件: {json_path}")
        print(f"- 详细报告: {report_path}")
        print("=" * 80)

        # 在控制台打印报告摘要
        print("\n报告摘要:\n")
        print(report.split("完整年份详情:")[0])
    else:
        print("扫描失败，请检查错误信息。")