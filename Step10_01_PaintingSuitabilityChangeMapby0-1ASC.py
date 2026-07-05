import os
import numpy as np
import rasterio
from rasterio.enums import ColorInterp

# ========= 参数（按需修改）=========
current_asc = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\GCM_CurrentlyClim_avg.asc"
future_asc  = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\01ASC\Rhizosheath_2040-2060_SSP245Prediction_Suitablity.asc"
output_tif  = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\g.ProjectionMapforMid-21st\DataInGis\Rhizosheath_SuitabilityChange.tif"

# 若输入不是严格 0/1，可设置阈值进行二值化
BIN_THRESHOLD = 0.5

def generate_change_geotiff(cur_path, fut_path, out_tif):
    # 读取并检查网格一致性
    with rasterio.open(cur_path) as sc, rasterio.open(fut_path) as sf:
        if not (sc.width == sf.width and sc.height == sf.height and sc.transform == sf.transform and sc.crs == sf.crs):
            raise ValueError("当前期与未来期栅格网格（尺寸/变换/CRS）不一致，请先对齐。")

        # 读取为 masked array（自动处理 nodata）
        cur = sc.read(1, masked=True).astype(np.float32)
        fut = sf.read(1, masked=True).astype(np.float32)

        # 二值化（若已是0/1则不影响）
        cur_bin = np.where(cur.mask, np.nan, cur.data)
        fut_bin = np.where(fut.mask, np.nan, fut.data)
        cur_bin = (cur_bin > BIN_THRESHOLD).astype(np.uint8)
        fut_bin = (fut_bin > BIN_THRESHOLD).astype(np.uint8)

        # 有效掩膜
        valid = (~cur.mask) & (~fut.mask)

        # 分类编码：0=持续不发生, 1=退缩, 2=稳定, 3=扩张, 255=nodata
        nodata_code = 255
        result = np.full(cur.shape, nodata_code, dtype=np.uint8)

        both0 = (cur_bin == 0) & (fut_bin == 0) & valid
        lost  = (cur_bin == 1) & (fut_bin == 0) & valid
        stab  = (cur_bin == 1) & (fut_bin == 1) & valid
        expa  = (cur_bin == 0) & (fut_bin == 1) & valid

        result[both0] = 0
        result[lost]  = 1
        result[stab]  = 2
        result[expa]  = 3

        # 写小体积 GeoTIFF（LZW 压缩 + 分块 + uint8）
        profile = sc.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            compress="LZW",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=nodata_code
        )
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(result, 1)
            # 可选调色板（大多数 GIS 支持）
            try:
                cmap = {
                    0: (200, 200, 200),  # 持续不发生
                    1: (230, 85,  13),   # 退缩
                    2: (49,  163, 84),   # 稳定
                    3: (49,  130, 189),  # 扩张
                }
                dst.write_colormap(1, cmap)
                dst.colorinterp = (ColorInterp.palette,)
            except Exception:
                pass

if __name__ == "__main__":
    generate_change_geotiff(current_asc, future_asc, output_tif)
    print(f"✅ 已输出小体积 GeoTIFF：{output_tif}")
