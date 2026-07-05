import rasterio
import numpy as np
import os
import sys

# ===================== 参数设置 =====================
input_files = [
    {
        "path": r"I:\FIO-ESM-2-0\Rhizosheath_avg.asc",
        "threshold": 0.4184,
        "output": r"I:\GCM_CurrentlyClim_avg.asc"
    },
    {
        "path": r"I:\FIO-ESM-2-0\Rhizosheath_FIO-ESM-2-0_avg.asc",
        "threshold": 0.4184,
        "output": r"I:\Rhizosheath_2040-2060_SSP245Prediction_Suitablity.asc"
    }
]


# ===================== 增强版分类函数 =====================
def classify_raster(input_path, threshold, output_path):
    print(f"\n{'=' * 50}")
    print(f"📂 处理: {os.path.basename(input_path)}")
    print(f"🔑 阈值: {threshold}")
    print(f"💾 输出: {os.path.basename(output_path)}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入文件不存在 - {input_path}")
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"📁 创建输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 无法创建输出目录: {e}")
            return False

    try:
        with rasterio.open(input_path) as src:
            # 获取详细的栅格信息
            print(f"📊 栅格信息: {src.width}×{src.height} 像元 | 分辨率: {src.res}")
            print(f"📐 范围: ({src.bounds.left}, {src.bounds.bottom}) 到 ({src.bounds.right}, {src.bounds.top})")

            data = src.read(1)
            meta = src.meta.copy()
            nodata = meta.get('nodata', -9999)

            print(f"🔢 NoData值: {nodata} | 数据类型: {data.dtype}")
            print(f"📈 值范围: {np.nanmin(data)} - {np.nanmax(data)}")

            # 创建结果数组
            classified = np.full_like(data, nodata, dtype=np.float32)

            # 处理nodata值
            valid_mask = data != nodata
            if np.any(valid_mask):
                print(f"✅ 有效像元: {np.count_nonzero(valid_mask)}")

                # 应用阈值
                classified[valid_mask] = np.where(data[valid_mask] > threshold, 1, 0)

                # 统计结果
                suitable_count = np.count_nonzero(classified[valid_mask] == 1)
                unsuitable_count = np.count_nonzero(classified[valid_mask] == 0)
                print(f"🌱 适宜区域: {suitable_count}像元 ({suitable_count / np.count_nonzero(valid_mask) * 100:.2f}%)")
                print(f"🚫 不适宜区域: {unsuitable_count}像元")
            else:
                print("⚠️ 警告: 没有有效像元!")

            # 更新元数据
            meta.update({
                "dtype": "float32",
                "driver": "AAIGrid",
                "nodata": nodata
            })

            # 保存结果
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(classified.astype(np.float32), 1)
                print(f"💾 已保存: {output_path}")

            return True

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ===================== 主执行部分 =====================
if __name__ == "__main__":
    success_count = 0

    for idx, f in enumerate(input_files):
        print(f"\n🔹 处理文件 {idx + 1}/{len(input_files)}")
        if classify_raster(f["path"], f["threshold"], f["output"]):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"🎉 处理完成! 成功: {success_count}/{len(input_files)}")
    if success_count < len(input_files):
        print("⚠️ 部分文件处理失败，请检查错误信息")

    # 添加暂停以便查看结果
    if sys.platform.startswith('win'):
        os.system("pause")