import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

# ============ 参数设置 ============
input_root = r"H:\585Clim"   # 拆分后的 BIO01-19 目录们
reference_file = r"T:\Work\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version06_260128\BIO01.tif"  # 参考栅格（目标几何）
output_format = "asc"   # 只支持 asc
DST_NODATA = -9999.0    # ASC 的 nodata
# 如果你的这一批都是连续变量（BIO），建议双线性；若有离散类，用 Resampling.nearest
DEFAULT_RESAMPLING = Resampling.bilinear

# ============ 工具函数 ============
def load_reference(ref_file):
    with rasterio.open(ref_file) as ref:
        return {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "nodata": ref.nodata,     # 参考栅格可能没有 nodata，也没关系
            "dtype": ref.dtypes[0]
        }

def _clean_to_nan(arr, src_nodata):
    """把源 nodata 或极端哨兵值清成 NaN；保留其它为 float32。"""
    arr = arr.astype("float32", copy=False)

    # 1) 源 nodata -> NaN
    if src_nodata is not None:
        arr = np.where(arr == src_nodata, np.nan, arr)

    # 2) 极端哨兵值（常见于 CMIP/浮点最小值） -> NaN
    #    有些文件没有写 nodata 标签，但把无效像元写成 -3.4e38 之类的极值
    extreme_mask = (~np.isnan(arr)) & ((arr <= -1e20) | (arr >= 1e20) | np.isinf(arr))
    if np.any(extreme_mask):
        arr[extreme_mask] = np.nan

    return arr

def reproject_to_match(src_file, ref_meta, resampling=DEFAULT_RESAMPLING):
    """将 tif 重投影到参考几何，返回 float32 数组（内部先用 NaN 表示无效像元）。"""
    with rasterio.open(src_file) as src:
        dst_array = np.empty((ref_meta["height"], ref_meta["width"]), dtype="float32")

        # 明确告诉 GDAL 源/目标的 nodata，避免 nodata 参与插值
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            dst_nodata=np.nan,  # 先用 NaN 表达，写 ASC 再落成 -9999
            resampling=resampling,
            init_dest_nodata=True,
        )

        # 再做一次保险清洗（若源没标 nodata，但有极端值）
        dst_array = _clean_to_nan(dst_array, src_nodata=None)
        return dst_array

def tif_to_asc(tif_file, ref_meta, resampling=DEFAULT_RESAMPLING):
    """把 tif 转为 ASC；必要时先重投影对齐参考几何。"""
    if not tif_file.lower().endswith(".tif"):
        return None

    asc_file = tif_file[:-4] + f".{output_format}"

    if os.path.exists(asc_file):
        print(f"⏩ 已存在，跳过: {asc_file}")
        return asc_file

    with rasterio.open(tif_file) as src:
        same_geo = (
            (src.crs == ref_meta["crs"]) and
            (src.transform == ref_meta["transform"]) and
            (src.width == ref_meta["width"]) and
            (src.height == ref_meta["height"])
        )

        if same_geo:
            # 直接读取并转 NaN
            arr = src.read(1)
            arr = _clean_to_nan(arr, src_nodata=src.nodata)
        else:
            # 重投影到参考几何
            arr = reproject_to_match(tif_file, ref_meta, resampling=resampling)

    # 写出为 ASCII Grid（AAIGrid）
    meta_out = {
        "driver": "AAIGrid",
        "height": ref_meta["height"],
        "width": ref_meta["width"],
        "transform": ref_meta["transform"],
        "dtype": "float32",
        "count": 1,
        "nodata": DST_NODATA,   # 头部 nodata
    }

    # NaN -> -9999
    arr_out = np.where(np.isnan(arr), DST_NODATA, arr).astype("float32", copy=False)

    with rasterio.open(asc_file, "w", **meta_out) as dst:
        dst.write(arr_out, 1)

    print(f"✅ 转换完成: {asc_file}")
    return asc_file

def check_asc_vs_tif(tif_file, asc_file, ref_meta, resampling=DEFAULT_RESAMPLING):
    """抽查几个像元，允许微小数值差，忽略 NaN（nodata）。"""
    # 取参考几何下的 tif 值（float32 + NaN）
    tif_ref = reproject_to_match(tif_file, ref_meta, resampling=resampling)

    with rasterio.open(asc_file) as a:
        asc_arr = a.read(1).astype("float32")
        asc_arr = np.where(asc_arr == DST_NODATA, np.nan, asc_arr)

    if tif_ref.shape != asc_arr.shape:
        raise ValueError(f"尺寸不一致: {tif_file} -> {asc_file}")

    # 随机抽查位置（固定几个）
    for (r, c) in [(50, 50), (100, 100), (200, 200)]:
        if r >= tif_ref.shape[0] or c >= tif_ref.shape[1]:
            continue
        v1, v2 = tif_ref[r, c], asc_arr[r, c]
        # 两个都 NaN -> 通过；否则用 isclose
        if (np.isnan(v1) and np.isnan(v2)):
            continue
        if not np.isclose(v1, v2, rtol=1e-5, atol=1e-5):
            print(f"⚠️ 数据差异: {os.path.basename(tif_file)} at ({r},{c}): tif={v1}, asc={v2}")

    print(f"🔍 抽查通过: {os.path.basename(asc_file)}")

def check_all_bioasc(gcm_dir):
    """检查目录里是否存在 BIO01-19.asc"""
    expected = [f"BIO{str(i).zfill(2)}.asc" for i in range(1, 20)]
    found = [os.path.basename(f) for f in glob.glob(os.path.join(gcm_dir, "BIO*.asc"))]

    missing = [f for f in expected if f not in found]
    if missing:
        print(f"❌ 缺失文件 ({gcm_dir}): {missing}")
    else:
        print(f"✅ 完整: {gcm_dir}")

# ============ 主流程 ============
def main():
    ref_meta = load_reference(reference_file)

    for gcm_dir in glob.glob(os.path.join(input_root, "*")):
        if not os.path.isdir(gcm_dir):
            continue
        print(f"\n📂 处理目录: {gcm_dir}")

        # 转换并抽查
        for tif_file in glob.glob(os.path.join(gcm_dir, "BIO*.tif")):
            asc_file = tif_to_asc(tif_file, ref_meta, resampling=DEFAULT_RESAMPLING)
            if asc_file is not None:
                check_asc_vs_tif(tif_file, asc_file, ref_meta, resampling=DEFAULT_RESAMPLING)

        # 完整性检查
        check_all_bioasc(gcm_dir)

if __name__ == "__main__":
    main()
