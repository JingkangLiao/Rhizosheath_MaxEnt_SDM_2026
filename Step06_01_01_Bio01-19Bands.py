import os
import rasterio

# 输入和输出目录
input_dir = r"H:\585Clim"
output_dir = r"H:\585Clim"

# BIO 变量名（19 个）
bio_vars = [f"BIO{str(i).zfill(2)}" for i in range(1, 20)]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 遍历目录下的所有 tif 文件
for file in os.listdir(input_dir):
    if file.endswith(".tif"):
        file_path = os.path.join(input_dir, file)
        # 提取 GCM 名称，假设固定为倒数第3个字段
        gcm_name = file.split("_")[-3]
        out_folder = os.path.join(output_dir, gcm_name)
        ensure_dir(out_folder)

        print(f"\n正在处理: {file}")

        with rasterio.open(file_path) as src:
            band_count = src.count
            print(f"  波段数量: {band_count}")

            if band_count != 19:
                raise ValueError(f"❌ 文件 {file} 波段数={band_count}, 不等于 19，请检查数据来源！")

            for i in range(1, band_count + 1):
                band_data = src.read(i)
                out_meta = src.meta.copy()
                out_meta.update({
                    "count": 1,
                    "dtype": band_data.dtype
                })

                out_name = os.path.join(out_folder, f"{bio_vars[i - 1]}.tif")
                with rasterio.open(out_name, "w", **out_meta) as dst:
                    dst.write(band_data, 1)

            print(f"  ✅ 已完成拆分并保存到: {out_folder}")

print("\n🎉 全部完成！")
