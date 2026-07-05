import rasterio
import numpy as np
import os
import sys

# ===================== 参数设置 =====================
input_files = [
    {
        "path": r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\UKESM1-0-LL_Results\Rhizosheath_avg.asc",
        "threshold": 0.344,
        "output": r"I:\GCM_CurrentlyClim_avg.asc"
    },
    {
        "path": r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\UKESM1-0-LL_Results\Rhizosheath_UKESM1-0-LL_avg.asc",
        "threshold": 0.344,
        "output": r"I:\Rhizosheath_2040-2060_SSP245Prediction_Suitablity.asc"
    },
    {
        "path": r"L:\126\Rhizosheath_126-UKESM1-0-LL_avg.asc",
        "threshold": 0.344,
        "output": r"I:\Rhizosheath_2040-2060_SSP126Prediction_Suitablity.asc"
    },
    {
        "path": r"L:\585\Rhizosheath_585-UKESM1-0-LL_avg.asc",
        "threshold": 0.344,
        "output": r"I:\Rhizosheath_2040-2060_SSP585Prediction_Suitablity.asc"
    }
]


# ===================== 增强版分类函数 =====================
def classify_raster(input_path, threshold, output_path):
    print(f"\n{'=' * 50}")
    print(f"📂 处理: {os.path.basename(input_path)}")
    print(f"🔑 阈值: {threshold}")
    print(f"💾 输出: {os.path.basename(output_path)}")

    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入文件不存在 - {input_path}")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with rasterio.open(input_path) as src:
            raw = src.read(1).astype(np.float32, copy=False)
            src_nodata = src.nodata

            # —— 统一屏蔽无效值：nodata / NaN / ±Inf -> np.nan
            data = raw
            if src_nodata is not None and np.isfinite(src_nodata):
                data = np.where(raw == src_nodata, np.nan, raw)
            data = np.where(np.isfinite(data), data, np.nan)

            # 打印有效范围（不受 nodata 污染）
            if np.isnan(data).all():
                vmin = vmax = np.nan
            else:
                vmin, vmax = np.nanmin(data), np.nanmax(data)

            print(f"📊 栅格: {src.width}×{src.height} | 分辨率: {src.res}")
            print(f"📐 范围: ({src.bounds.left}, {src.bounds.bottom}) → ({src.bounds.right}, {src.bounds.top})")
            print(f"🔢 源 NoData: {src_nodata} | dtype: {raw.dtype}")
            print(f"📈 有效值范围: {vmin} ~ {vmax}")

            # —— 分类（仅在有效像元上判断），按习惯用 ≥
            classified = np.full_like(data, np.nan, dtype=np.float32)
            valid_mask = ~np.isnan(data)
            if np.any(valid_mask):
                classified[valid_mask] = np.where(data[valid_mask] >= float(threshold), 1.0, 0.0)
                total_valid = int(valid_mask.sum())
                suitable = int(np.nansum(classified == 1.0))
                print(f"✅ 有效像元: {total_valid}")
                print(f"🌱 适宜: {suitable}/{total_valid} ({(suitable/total_valid)*100:.2f}%)")
            else:
                print("⚠️ 警告: 没有有效像元！")

            # —— 写出为 ASCII Grid（AAIGrid）：把 nan 写回 -9999
            out_nodata = -9999.0
            meta = src.meta.copy()
            # AAIGrid 需要这些关键字段；去掉与 GTiff 相关的键（若有）
            for k in ("compress", "tiled", "interleave", "blockxsize", "blockysize", "driver"):
                meta.pop(k, None)
            meta.update({
                "driver": "AAIGrid",
                "dtype": "float32",
                "count": 1,
                "nodata": out_nodata
            })

            to_write = np.where(np.isnan(classified), out_nodata, classified).astype(np.float32)
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(to_write, 1)

            print(f"💾 已保存: {output_path}")
            return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback; traceback.print_exc()
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