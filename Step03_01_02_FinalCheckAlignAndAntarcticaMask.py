# 1. 用 wc2.1_1km_bioclim01_avg.tif 作为模板，检查同目录下所有 tif 是否同网格
# 2. 如果不对齐 -> 重投影/重采样到模板网格
# 3. 对齐后，强制把 南纬60° 以南 (lat < -60) 的像元设为 nodata
# 4. 将结果写到 version05_1028/final/ 下，文件名保持不变
# 5. 打印详细报告（对齐情况 & 是否在南极区域发现有效值）

import os
import glob
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

# ===================== 用户路径配置 =====================
BASE_DIR = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version06_260128"
TEMPLATE_NAME = "wc2.1_1km_bioclim01_avg.tif"  # 模板
FINAL_DIR = os.path.join(BASE_DIR, "final")

# 连续变量 -> 双线性。我们现在所有变量都按连续变量处理
DEFAULT_RESAMPLING = Resampling.bilinear

# 输出影像：统一 float32 + NaN nodata
OUT_DTYPE = "float32"
OUT_NODATA = np.nan

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("check-align-mask")


# ===================== 工具函数 =====================

def decompose_transform(transform):
    """
    把 rasterio 的 transform 解析成 (a,b,c,d,e,f)
    兼容:
    - Affine(a,b,c,d,e,f)
    - tuple/list 长度为6
    - tuple/list 长度为9 (3x3矩阵摊平)，取前6个
    """
    if hasattr(transform, "a") and hasattr(transform, "b") and hasattr(transform, "c") \
       and hasattr(transform, "d") and hasattr(transform, "e") and hasattr(transform, "f"):
        return (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )

    vals = list(transform)
    if len(vals) == 6:
        a, b, c, d, e, f = vals
        return float(a), float(b), float(c), float(d), float(e), float(f)
    elif len(vals) == 9:
        a, b, c, d, e, f = vals[:6]
        return float(a), float(b), float(c), float(d), float(e), float(f)

    raise ValueError(f"无法解析 transform，长度={len(vals)}, 内容={vals}")


def load_template(template_path):
    """
    读取模板，拿到标准网格信息。
    返回:
        template_geom: dict(crs, transform, width, height)
        template_profile: profile
    """
    with rasterio.open(template_path) as src:
        template_geom = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height
        }
        template_profile = src.profile.copy()

    log.info(f"[模板] {template_path}")
    log.info(f"  尺寸: {template_geom['width']} x {template_geom['height']}")
    log.info(f"  CRS: {template_geom['crs']}")
    log.info(f"  Transform: {template_geom['transform']}")
    return template_geom, template_profile


def guess_src_nodata(src):
    """
    获取源 nodata。
    优先 src.nodata，其次常见 tag。
    返回 float 或 None。
    """
    if src.nodata is not None:
        try:
            return float(src.nodata)
        except Exception:
            pass

    tags = src.tags()
    for k in ("NODATA", "NoData", "_FillValue", "missing_value", "nodata"):
        if k in tags:
            try:
                return float(tags[k])
            except Exception:
                pass

    return None


def same_grid(src, template_geom):
    """
    判断是否跟模板完全对齐（尺寸/transform/CRS 都一致）。
    """
    return (
        src.width == template_geom["width"] and
        src.height == template_geom["height"] and
        src.transform == template_geom["transform"] and
        src.crs == template_geom["crs"]
    )


def compute_lat_rows(transform, height):
    """
    给定整幅影像的仿射变换和高度，计算每一行像元中心的纬度。
    仿射：X = a*col + b*row + c
          Y = d*col + e*row + f
    我们用列=0.5, 行=row+0.5 估计该行中心纬度。
    返回 lat_rows: shape=(height,), float64
    """
    a, b, c, d, e, f = decompose_transform(transform)

    rows = np.arange(height, dtype=np.float64)
    col_center = 0.5       # 任取这一列的中心
    row_center = rows + 0.5

    # 计算每行中心点的纬度 (Y)
    # Y = d * col + e * row + f
    lat_vals = d * col_center + e * row_center + f
    return lat_vals.astype(np.float64)


def mask_antarctica_in_block(block_data, lat_rows, row_off, win_h):
    """
    block_data: (win_h, win_w) float32
    lat_rows:   (H,) 整幅图每一行的纬度
    row_off:    这个 window 在整幅图的起始行
    win_h:      block 的行数

    我们的逻辑是：如果某一整行的纬度 < -60，则整行全部设为 NaN。

    返回:
      cleaned_block          (win_h, win_w)
      changed_count          被我们改成 NaN 的像元数量
      had_valid_in_antarctica 这个窗口内南极区域是否原本有有效值
    """
    # 当前 block 行对应的全局纬度
    block_lats = lat_rows[row_off:row_off + win_h]  # shape (win_h,)
    rows_to_mask = block_lats < -60.0               # shape (win_h,), True 表示这整行在南极

    cleaned_block = block_data.copy()

    # 找出这些行里原本是有效值（非NaN）的像元
    if np.any(rows_to_mask):
        originally_valid = np.isfinite(block_data[rows_to_mask, :])
        changed_count = int(np.count_nonzero(originally_valid))
        had_valid_in_antarctica = bool(np.any(originally_valid))

        # 直接整行赋 NaN
        cleaned_block[rows_to_mask, :] = np.nan
    else:
        changed_count = 0
        had_valid_in_antarctica = False

    return cleaned_block, changed_count, had_valid_in_antarctica


def process_one_raster(src_path, template_geom, lat_rows, final_dir):
    """
    对单个 tif 做以下处理：
    1. 如果网格不同 -> 重投影/重采样到模板
    2. 把纬度<-60°的像元整行设为NaN
    3. 保存到 final_dir，文件名不变
    4. 返回报告信息
    """
    fname = os.path.basename(src_path)
    dst_path = os.path.join(final_dir, fname)

    with rasterio.open(src_path) as src:
        src_nd = guess_src_nodata(src)
        aligned = same_grid(src, template_geom)

        # 输出 profile（统一 float32+NaN nodata）
        out_profile = {
            "driver": "GTiff",
            "crs": template_geom["crs"],
            "transform": template_geom["transform"],
            "width": template_geom["width"],
            "height": template_geom["height"],
            "count": 1,
            "dtype": OUT_DTYPE,
            "nodata": OUT_NODATA,
            "tiled": True,
            "compress": "lzw",
            "BIGTIFF": "IF_SAFER"
        }

        os.makedirs(final_dir, exist_ok=True)

        with rasterio.open(dst_path, "w", **out_profile) as dst:

            total_changed = 0
            antarctica_had_data = False

            # 遍历目标网格的 block window
            for _, win in dst.block_windows(1):
                win_h = win.height
                win_w = win.width
                row_off = win.row_off
                col_off = win.col_off

                # 初始化为 NaN
                block = np.full((win_h, win_w), np.nan, dtype=np.float32)

                if aligned:
                    # 如果已经和模板同网格，可以直接按窗口读取
                    window = Window(col_off, row_off, win_w, win_h)
                    arr = src.read(1, window=window, out_dtype="float32", masked=False)

                    # 源 nodata -> NaN
                    if src_nd is not None and np.isfinite(src_nd):
                        mask_nodata = (arr == src_nd)
                        arr = arr.astype(np.float32)
                        arr[mask_nodata] = np.nan
                    else:
                        arr = arr.astype(np.float32)

                    block[:, :] = arr

                else:
                    # 需要重采样/重投影到模板的这一块
                    win_transform = window_transform(win, template_geom["transform"])

                    reproject(
                        source=rasterio.band(src, 1),
                        destination=block,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src_nd if (src_nd is not None and np.isfinite(src_nd)) else None,
                        dst_transform=win_transform,
                        dst_crs=template_geom["crs"],
                        dst_nodata=np.nan,
                        resampling=DEFAULT_RESAMPLING,
                        init_dest_nodata=True,
                        num_threads=max(1, (os.cpu_count() or 8) - 1)
                    )

                # 清除南极区域
                cleaned_block, changed_cnt, had_antarctic_vals = mask_antarctica_in_block(
                    block,
                    lat_rows,
                    row_off=row_off,
                    win_h=win_h
                )

                total_changed += changed_cnt
                antarctica_had_data = antarctica_had_data or had_antarctic_vals

                # 写出
                dst.write(cleaned_block.astype(np.float32), 1, window=win)

    summary = {
        "file": fname,
        "same_grid_as_template": aligned,
        "antarctica_had_data_before_fix": antarctica_had_data,
        "antarctica_pixels_cleared": total_changed,
        "output_path": dst_path
    }
    return summary


def main():
    # 1. 读模板
    template_path = os.path.join(BASE_DIR, TEMPLATE_NAME)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_path}")

    template_geom, _ = load_template(template_path)

    # 2. 预先算整幅图每一行对应的纬度
    lat_rows = compute_lat_rows(template_geom["transform"], template_geom["height"])

    # 3. 找所有 tif
    all_tifs = sorted(glob.glob(os.path.join(BASE_DIR, "*.tif")))
    if not all_tifs:
        log.error("未在指定目录中找到任何 .tif 文件")
        return

    # 避免把 final 目录里的东西再处理一遍
    all_tifs = [fp for fp in all_tifs if os.path.dirname(fp) != FINAL_DIR]

    log.info("==========================================")
    log.info(f"开始检查 & 修正，共 {len(all_tifs)} 个文件")
    log.info("==========================================")

    os.makedirs(FINAL_DIR, exist_ok=True)

    report_list = []

    # 4. 逐文件处理
    for i, tif_path in enumerate(all_tifs, start=1):
        log.info("------------------------------------------")
        log.info(f"[{i}/{len(all_tifs)}] 处理: {os.path.basename(tif_path)}")

        try:
            summary = process_one_raster(
                src_path=tif_path,
                template_geom=template_geom,
                lat_rows=lat_rows,
                final_dir=FINAL_DIR
            )

            log.info(f"  - same_grid_as_template: {summary['same_grid_as_template']}")
            log.info(f"  - antarctica_had_data_before_fix: {summary['antarctica_had_data_before_fix']}")
            log.info(f"  - antarctica_pixels_cleared: {summary['antarctica_pixels_cleared']}")
            log.info(f"  - output: {summary['output_path']}")

            report_list.append(summary)

        except Exception as e:
            log.error(f"  ❌ 处理失败: {tif_path}")
            log.error(f"     错误信息: {e}")

    # 5. 汇总报告
    log.info("==========================================")
    log.info("处理完成。汇总结果：")
    log.info("==========================================")

    n_ok = 0
    n_need_reproj = 0
    n_antarctic_fixed = 0

    for r in report_list:
        if r["same_grid_as_template"]:
            n_ok += 1
        else:
            n_need_reproj += 1

        if r["antarctica_pixels_cleared"] > 0:
            n_antarctic_fixed += 1

        log.info(
            f"[{r['file']}] "
            f"grid_ok={r['same_grid_as_template']}, "
            f"antarctica_had_data={r['antarctica_had_data_before_fix']}, "
            f"cleared_pixels={r['antarctica_pixels_cleared']}, "
            f"final='{r['output_path']}'"
        )

    log.info("------------------------------------------")
    log.info(f"总文件数: {len(report_list)}")
    log.info(f"✔ 已与模板同网格的文件数: {n_ok}")
    log.info(f"↻ 需要重采样/重投影的文件数: {n_need_reproj}")
    log.info(f"🧊 在南极(<-60°)检测到并清除值的文件数: {n_antarctic_fixed}")
    log.info("------------------------------------------")
    log.info(f"所有修正后文件保存在: {FINAL_DIR}")
    log.info("完成 ✅")


if __name__ == "__main__":
    main()

