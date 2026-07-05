import os
import glob
import numpy as np
import rasterio


# ================== 可修改区 ==================
IN_DIR  = r"G:\BioclimaticVariables\Soil_HWSD2\Raw"
OUT_DIR = r"G:\BioclimaticVariables\Soil_HWSD2\Fixed"

PATTERN = "**/*.tif"     # 递归搜索所有 tif
NEG_THRESHOLD = 0        # < 0 视为缺失
DEFAULT_NODATA = -9999   # 若原 tif 未定义 nodata，则使用该值

COMPRESS = "LZW"
TILED = True             # 写出 tiled tif（更快）
BIGTIFF = "IF_SAFER"     # 大文件自动 BigTIFF
# =============================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_nodata_for_dtype(dtype, preferred):
    """
    确保 nodata 能被 dtype 表示（避免 uint16 / int8 等报错）。
    - float: 直接返回 preferred（或 -9999.0）
    - int: 若 preferred 超范围，则：
          有符号整型 -> info.min
          无符号整型 -> info.max
    """
    if np.issubdtype(dtype, np.floating):
        return float(preferred) if preferred is not None else -9999.0

    info = np.iinfo(dtype)
    if preferred is None:
        preferred = DEFAULT_NODATA

    if info.min <= preferred <= info.max:
        return int(preferred)

    # preferred 超出范围：给一个 dtype 可表示的“极值”兜底
    return int(info.min if info.min < 0 else info.max)


def fix_one_tif(in_path: str, out_path: str) -> dict:
    """
    将单波段tif中所有 < NEG_THRESHOLD 的像元替换为 nodata。
    nodata：若原有则沿用，否则设为 DEFAULT_NODATA（并适配 dtype）。
    同时继承/尊重 GeoTIFF 内部 mask band（若存在），并将无效区显式写成 nodata。
    输出保留原始 dtype / crs / transform / extent，并使用 LZW 压缩。
    返回处理统计信息。
    """
    with rasterio.open(in_path) as src:
        if src.count != 1:
            raise ValueError(f"[非单波段] {in_path} 有 {src.count} 个波段（你要求都是1个波段）。")

        profile = src.profile.copy()

        # ✅ 关键：masked=True 会把 nodata + 内部 mask band 一并转成 mask
        arr_m = src.read(1, masked=True)   # np.ma.MaskedArray
        arr = arr_m.data                   # ndarray（保持原 dtype）
        mask_valid = ~arr_m.mask           # True 表示有效像元（已排除内部 mask/nodata）

        # 读取原 nodata；若无则用 DEFAULT_NODATA，并确保适配 dtype
        src_nodata = src.nodata
        raw_nodata = src_nodata if (src_nodata is not None) else DEFAULT_NODATA
        nodata = choose_nodata_for_dtype(arr.dtype, raw_nodata)

        # ✅ 修复：若源 nodata 是 NaN，masked=True 未必能正确 mask NaN，需要显式排除
        if src_nodata is not None and np.issubdtype(arr.dtype, np.floating):
            if np.isnan(src_nodata):
                mask_valid &= ~np.isnan(arr)

        # 只在有效像元里找负值
        neg_mask = (arr < NEG_THRESHOLD) & mask_valid
        n_neg = int(np.count_nonzero(neg_mask))
        n_total = int(arr.size)

        arr_fixed = arr.copy()

        # 替换负值为 nodata（并把原本无效区也显式写成 nodata，避免 mask 丢失）
        if np.issubdtype(arr_fixed.dtype, np.integer):
            nd = np.array(nodata, dtype=arr_fixed.dtype)
            arr_fixed[neg_mask] = nd
            arr_fixed[~mask_valid] = nd
        else:
            nd = float(nodata)
            arr_fixed[neg_mask] = nd
            arr_fixed[~mask_valid] = nd

        # 更新写出 profile：保持空间参考等不动，只改 nodata/压缩等
        profile.update(
            nodata=nodata,
            compress=COMPRESS,
            tiled=TILED,
            bigtiff=BIGTIFF,
            count=1
        )

        ensure_dir(os.path.dirname(out_path))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr_fixed, 1)

    return {
        "in_path": in_path,
        "out_path": out_path,
        "dtype": str(arr.dtype),
        "src_nodata": src_nodata,
        "raw_nodata": raw_nodata,
        "used_nodata": nodata,
        "neg_count": n_neg,
        "total_pixels": n_total
    }


def main():
    in_paths = sorted(glob.glob(os.path.join(IN_DIR, PATTERN), recursive=True))
    if not in_paths:
        raise FileNotFoundError(f"未找到任何 tif：{IN_DIR}\\{PATTERN}")

    print(f"[INFO] 输入 tif 数量：{len(in_paths)}")
    print(f"[INFO] 输出目录：{OUT_DIR}")

    sum_neg = 0
    changed_files = 0

    for i, in_path in enumerate(in_paths, start=1):
        rel = os.path.relpath(in_path, IN_DIR)  # 保持相对目录结构
        out_path = os.path.join(OUT_DIR, rel)

        stats = fix_one_tif(in_path, out_path)

        sum_neg += stats["neg_count"]
        if stats["neg_count"] > 0:
            changed_files += 1

        print(
            f"[{i}/{len(in_paths)}] {os.path.basename(in_path)} | "
            f"neg={stats['neg_count']} | "
            f"nodata(src={stats['src_nodata']}, raw={stats['raw_nodata']}, used={stats['used_nodata']}) | "
            f"dtype={stats['dtype']}"
        )

    print(f"[DONE] 共处理 {len(in_paths)} 个文件；含负值并被修正的文件：{changed_files}")
    print(f"[DONE] 总计被置为 NoData 的像元数（所有文件合计）：{sum_neg}")


if __name__ == "__main__":
    main()
