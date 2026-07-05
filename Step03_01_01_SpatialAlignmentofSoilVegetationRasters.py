# 因为我降低了map的精度
# 需要对植被类型数据和土壤数据进行匹配
"这段代码以指定目录下的任意 TIFF 文件为模板获取其坐标系、分辨率、尺寸等几何参数，分别对土壤数据（连续型）采用双线性重采样、植被类型数据（分类型）采用最近邻重采样，将所有土壤 TIFF 文件和指定植被 TIFF 文件重新投影并重采样至与模板文件完全匹配的空间参考和分辨率，输出带 LZW 压缩的 TIFF 文件并验证结果有效性。"
import os
import glob
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ========= 路径设置 =========
TEMPLATE_DIR = r"L:\Bioclim_1km_avg"   # 从这里挑一张 tif 做模板
SOIL_SRC_DIR = r"G:\BioclimaticVariables\Soil_HWSD2\Fixed"
SOIL_OUT_DIR = r"G:\BioclimaticVariables\Soil_HWSD2\version03_251027"

VEG_SRC_FILE = r"G:\BioclimaticVariables\Veg_GLC_FCS\version01\GLC_FCS1000_Global_2023_aligned.tif"
VEG_OUT_FILE = r"G:\BioclimaticVariables\Veg_GLC_FCS\version03_251027\GLC_FCS1000_Global_2023_to_template.tif"

RUN_SOIL = True
RUN_VEG  = True
# ===========================

# 建议的 GDAL 环境变量（可选）
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "2048")  # MB
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("align-resample")

# ---------- 工具函数 ----------

def pick_template(template_dir):
    tifs = glob.glob(os.path.join(template_dir, "*.tif"))
    if not tifs:
        raise FileNotFoundError(f"模板目录没有 .tif: {template_dir}")
    tpl = tifs[0]
    with rasterio.open(tpl) as src:
        profile = src.profile.copy()
        # 只取几何相关
        geom = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height
        }
    log.info(f"模板文件: {tpl}")
    return geom, profile

def _to_number_or_none(val):
    """把 tag 里的 nodata 字符串安全转成数字；失败返回 None"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.lower() in ("nan", ""):
        return np.nan
    try:
        return float(s)
    except Exception:
        return None

def guess_src_nodata(src):
    """稳健获取源 nodata（数字或 np.nan 或 None）"""
    nd = src.nodata
    if nd is not None:
        # rasterio 会把 float nodata 给成 float, int 给成 int
        try:
            return float(nd)
        except Exception:
            pass

    # 尝试从 tags 猜
    tags = src.tags()
    for k in ("NODATA", "NoData", "_FillValue", "missing_value", "nodata"):
        if k in tags:
            v = _to_number_or_none(tags.get(k))
            if v is not None:
                return v

    # 采样检测常见哨兵
    try:
        # 读一个小窗口（左上角 512x512 或尽量小）
        h = min(src.height, 512)
        w = min(src.width, 512)
        samp = src.read(1, window=rasterio.windows.Window(0, 0, w, h), masked=False)
        dt = np.dtype(src.dtypes[0])
        kind = dt.kind  # 'f' 浮点, 'i' 有符号整型, 'u' 无符号整型
        if kind == "f":
            # 浮点：检查是否有极端大绝对值
            if np.isfinite(samp).any():
                if np.nanmax(np.abs(samp)) > 1e30:
                    return -3.4028235e+38  # 常见 float32 nodata
        elif kind in ("i", "u"):
            # 整型：-9999 / -32768/ 65535 / 255 等
            uniq = np.unique(samp)
            for cand in (-9999, -32768, 65535, 255):
                if cand in uniq:
                    return float(cand)
    except Exception:
        pass

    return None  # 实在猜不到

def make_out_profile(template_geom, is_classmap, src_dtype, dst_nodata):
    """根据模板几何与数据类型生成输出 profile"""
    prof = {
        "driver": "GTiff",
        "crs": template_geom["crs"],
        "transform": template_geom["transform"],
        "width": template_geom["width"],
        "height": template_geom["height"],
        "count": 1,
        "tiled": True,
        "compress": "lzw",
        "BIGTIFF": "IF_SAFER"
    }
    if is_classmap:
        # 分类：保持整型（优先用源整型；若源是浮点，回落到 uint16）
        dt = np.dtype(src_dtype)
        if dt.kind in ("i", "u"):
            prof["dtype"] = src_dtype
        else:
            prof["dtype"] = "uint16"
        prof["nodata"] = int(dst_nodata) if dst_nodata is not None and not np.isnan(dst_nodata) else 255
    else:
        # 连续型：float32 + NaN
        prof["dtype"] = "float32"
        prof["nodata"] = np.nan
    return prof

def reproject_to_template(src_path, dst_path, template_geom, is_classmap):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.Env():  # 使用进程默认 GDAL 配置
        with rasterio.open(src_path) as src:
            src_dtype = src.dtypes[0]
            src_nd = guess_src_nodata(src)  # 可能是 float('nan')/数字/None
            if src_nd is not None and isinstance(src_nd, float) and not np.isfinite(src_nd):
                # 如果是 nan，当作 None 传给 src_nodata（GDAL 会按数据内的 NaN 识别）
                src_nd_for_gdal = None
            else:
                src_nd_for_gdal = src_nd

            # 目标 nodata
            if is_classmap:
                # 分类：优先用源的合法整型 nodata；否则 255
                dst_nd = src_nd if (src_nd is not None and (not isinstance(src_nd, float) or np.isfinite(src_nd))) else 255
                resamp = Resampling.nearest
            else:
                dst_nd = np.nan
                resamp = Resampling.bilinear

            out_profile = make_out_profile(template_geom, is_classmap, src_dtype, dst_nd)
            # reproject 的 num_threads 必须是 **整数**
            num_threads = max(1, (os.cpu_count() or 8) - 1)

            with rasterio.open(dst_path, "w", **out_profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nd_for_gdal,
                    dst_transform=template_geom["transform"],
                    dst_crs=template_geom["crs"],
                    dst_nodata=dst_nd,
                    resampling=resamp,
                    num_threads=num_threads,
                    init_dest_nodata=True
                )

            # 简单抽样统计，验证没有 ±3e38 残留
            with rasterio.open(dst_path) as chk:
                win = rasterio.windows.Window(0, 0, min(1024, chk.width), min(1024, chk.height))
                arr = chk.read(1, window=win, masked=False)
                if np.issubdtype(chk.dtypes[0], np.floating):
                    _min = float(np.nanmin(arr))
                    _max = float(np.nanmax(arr))
                else:
                    ndv = chk.nodata
                    if ndv is not None:
                        mask = arr != ndv
                        if mask.any():
                            _min = int(arr[mask].min())
                            _max = int(arr[mask].max())
                        else:
                            _min = _max = int(ndv)
                    else:
                        _min = int(arr.min())
                        _max = int(arr.max())
                log.info(f"[OK] {os.path.basename(src_path)} -> {os.path.basename(dst_path)} "
                         f"sample(min={_min}, max={_max})")

def process_soil(template_geom):
    files = sorted(glob.glob(os.path.join(SOIL_SRC_DIR, "*.tif")))
    if not files:
        log.warning(f"SOIL 源目录没有 .tif: {SOIL_SRC_DIR}")
        return
    os.makedirs(SOIL_OUT_DIR, exist_ok=True)
    for i, src in enumerate(files, 1):
        dst = os.path.join(SOIL_OUT_DIR, os.path.basename(src))
        try:
            log.info(f"[{i}/{len(files)}] Soil: {src} -> {dst}")
            reproject_to_template(src, dst, template_geom, is_classmap=False)
        except Exception as e:
            log.error(f"[失败] {src} -> {dst}: {e}")

def process_veg(template_geom):
    os.makedirs(os.path.dirname(VEG_OUT_FILE), exist_ok=True)
    try:
        log.info(f"Veg: {VEG_SRC_FILE} -> {VEG_OUT_FILE}")
        reproject_to_template(VEG_SRC_FILE, VEG_OUT_FILE, template_geom, is_classmap=True)
    except Exception as e:
        log.error(f"[失败] {VEG_SRC_FILE} -> {VEG_OUT_FILE}: {e}")

def main():
    template_geom, _ = pick_template(TEMPLATE_DIR)
    if RUN_SOIL:
        process_soil(template_geom)
    if RUN_VEG:
        process_veg(template_geom)

if __name__ == "__main__":
    main()
