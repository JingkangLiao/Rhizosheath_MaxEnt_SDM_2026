# -*- coding: utf-8 -*-
import os
import glob
import math
import rasterio
from rasterio.errors import RasterioIOError

# ===== 用户配置 =====
FOLDER = r"D:\DOCTOR\Work\根鞘综述\Figures\Figure_Location&MaxEnt\c.LocationMaxEntFigure\VariablesIn\version05_1028"
# ====================

def near(a, b, tol=1e-7):
    return abs(a - b) <= tol

def res_label(px):
    """把经纬度像元度数转为常见名称"""
    cand = [
        (0.5,      "0.5° (~50 km)"),
        (0.25,     "0.25° (~25 km)"),
        (1/12,     "5′ (~10 km)"),
        (1/24,     "2.5′ (~5 km)"),
        (1/120,    "30″ (~1 km)"),
    ]
    for v, label in cand:
        if abs(px - v) < 5e-5:  # 给较宽容差，避免小数误差
            return label
    return f"{px:.8f}° (非常见网格)"

def fmt_crs(crs):
    try:
        if crs and crs.to_epsg():
            return f"EPSG:{crs.to_epsg()}"
        return str(crs) if crs else "None"
    except Exception:
        return str(crs) if crs else "None"

def human_size_bytes(nbytes):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    val = float(nbytes)
    while val >= 1024 and i < len(units)-1:
        val /= 1024.0
        i += 1
    return f"{val:.2f} {units[i]}"

def main():
    tifs = sorted(glob.glob(os.path.join(FOLDER, "*.tif")))
    if not tifs:
        print(f"⚠️ 目录下没有 .tif：{FOLDER}")
        return

    print(f"📂 目标目录：{FOLDER}")
    print(f"🧾 共发现 {len(tifs)} 个 .tif 文件\n")

    groups = {}  # key = (w,h,px,py,crs_str)
    any_error = False

    for i, fp in enumerate(tifs, 1):
        fn = os.path.basename(fp)
        try:
            with rasterio.open(fp) as ds:
                w, h = ds.width, ds.height
                tr = ds.transform
                # 像元大小（经纬度网格常见：a>0, e<0）
                px = float(abs(tr.a))
                py = float(abs(tr.e))
                # 旋转项
                rot_b = float(tr.b)
                rot_d = float(tr.d)
                rotated = (abs(rot_b) > 1e-9) or (abs(rot_d) > 1e-9)
                # CRS
                crs_str = fmt_crs(ds.crs)
                # dtype / nodata
                dtype = ds.dtypes[0]
                nodata = ds.nodata
                # 空间范围
                bounds = ds.bounds  # left, bottom, right, top
                # 文件大小
                try:
                    fsize = os.path.getsize(fp)
                    fsize_h = human_size_bytes(fsize)
                except Exception:
                    fsize_h = "未知"

                # 打印逐文件信息
                print(f"[{i}/{len(tifs)}] {fn}")
                print(f"  - 尺寸: {w} × {h} 像元")
                print(f"  - 分辨率: {px:.8f}° × {py:.8f}°  ({res_label(px)})")
                print(f"  - 旋转项: b={rot_b:.3e}, d={rot_d:.3e}  => {'有旋转⚠️' if rotated else '无旋转'}")
                print(f"  - CRS: {crs_str}")
                print(f"  - dtype: {dtype} | nodata: {nodata}")
                print(f"  - 范围: left={bounds.left:.6f}, right={bounds.right:.6f}, "
                      f"bottom={bounds.bottom:.6f}, top={bounds.top:.6f}")
                print(f"  - 文件大小: {fsize_h}")

                # 分组键（用四舍五入避免微小误差）
                key = (w, h, round(px, 8), round(py, 8), crs_str)
                groups.setdefault(key, []).append(fn)

                # 简要规范性提示
                if rotated:
                    print("  ⚠️ 提示：存在旋转项（b/d ≠ 0），严格对齐前建议先重投影到无旋转格网。")
                if (bounds.bottom < -90) or (bounds.top > 90):
                    print("  ⚠️ 提示：纬度范围越界，请检查 CRS/transform。")
                if ds.crs is None:
                    print("  ⚠️ 提示：缺少 CRS（未地理参照）。")

                print("")

        except RasterioIOError as e:
            any_error = True
            print(f"[{i}/{len(tifs)}] {fn}")
            print(f"  ❌ 无法打开: {e}\n")
        except Exception as e:
            any_error = True
            print(f"[{i}/{len(tifs)}] {fn}")
            print(f"  ❌ 解析失败: {e}\n")

    # 分组汇总
    print("\n================= 汇总（按 尺寸×分辨率×CRS 分组）=================")
    for idx, (key, files) in enumerate(groups.items(), 1):
        w, h, px, py, crs_str = key
        px_lab = res_label(px)
        print(f"[组 {idx}] {w}×{h} | {px:.8f}°×{py:.8f}° ({px_lab}) | {crs_str} | 共 {len(files)} 个：")
        # 估算每层 float32 未压缩体量
        est_bytes = w * h * 4
        print(f"       粗估未压缩单层体量: {human_size_bytes(est_bytes)}")
        # 列举部分文件名
        for fn in files[:10]:
            print(f"       - {fn}")
        if len(files) > 10:
            print(f"       ... 以及另外 {len(files) - 10} 个")
        print("------------------------------------------------------------------")

    if len(groups) == 1 and not any_error:
        print("\n✅ 结论：该文件夹内的 GeoTIFF 在“尺寸×分辨率×CRS”上完全一致，规范良好。")
    else:
        print("\n🔎 结论：存在至少 2 种不同的网格/CRS 或有文件解析失败。请根据上面的分组决定是否需要重采样对齐。")

if __name__ == "__main__":
    main()
